import os
import random
import shutil
from dataclasses import dataclass, field
import cv2
import clip
import torch
from PIL import Image
import shutil
import numpy as np
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.misc import get_rank, get_device, load_module_weights
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.draw_laion import DrawLaion


@threestudio.register("IDsculpt-geo-system")
class ImageConditionDreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"
        freq: dict = field(default_factory=dict)

        use_mixed_camera_config: bool = False

        visualize_samples: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        self.draw_laion = DrawLaion()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "texture":
            render_out = self.renderer(**batch, render_mask=True)
        else:
            render_out = self.renderer(**batch, render_rgb=False)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.pearson = PearsonCorrCoef().to(self.device)

    def training_substep(self, batch, batch_idx, guidance: str, render_type="rgb"):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "guidance"
        """
        gt_mask = batch["mask"] > 0.5   # float to bool
        gt_normal = batch["ref_normal"]
        mvp_mtx_ref = batch["mvp_mtx"]
        c2w_ref = batch["c2w4x4"]
        ldmk = batch["face_landmarks"].squeeze(0)

        if guidance == "guidance":
            batch = batch["random_camera"]
            mvp = batch["mvp_mtx"].clone()
            azimuth = batch["azimuth"].clone().item()
            control_img = self.draw_laion.draw(azimuth, mvp, ldmk, gt_normal.shape[1], gt_normal.shape[2])

        if guidance == "ref_guidance":
            batch = batch['ref_control_img']
            gt_normal = batch["normal"]
            depth_mask = batch["depth_for_mask"] > 0.2
            gt_mask = batch["mask"] > 0.5

        # Support rendering visibility mask
        batch["mvp_mtx_ref"] = mvp_mtx_ref
        batch["c2w_ref"] = c2w_ref

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
                guidance == "guidance"
                and self.cfg.freq.guidance_eval > 0
                and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        if guidance == "ref":
            # normal sds loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                        1 - 2 * gt_normal[gt_mask.squeeze(-1)]
                )  # [B, 3]

                pred_normal = out["comp_normal"]
                pred_normal = pred_normal[..., [0, 2, 1]]
                valid_pred_normal = (
                        1 - 2 * pred_normal[gt_mask.squeeze(-1)]
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                    # F.mse_loss(valid_pred_normal, valid_gt_normal),
                    # F.l1_loss(valid_pred_normal, valid_gt_normal),
                )

        elif guidance == "ref_guidance":
            valid_gt_normal = (
                    1 - 2 * gt_normal[gt_mask.squeeze(-1).squeeze(1)]
            )  # [B, 3]
            
            pred_normal = out["comp_normal"][..., [0, 2, 1]]
            valid_pred_normal = (
                    1 - 2 * pred_normal[gt_mask.squeeze(-1).squeeze(1)]
            )  # [B, 3]

            set_loss("normal", F.mse_loss(valid_pred_normal, valid_gt_normal))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                batch_depth = batch["depth"]  # [B, 1, H, W]
                out_depth = out["depth"].permute(0, 3, 1, 2)  # [B, 1, H, W]

                gt_mask_expanded = depth_mask
                valid_gt_depth = batch_depth[gt_mask_expanded].unsqueeze(1)  # [B, ]
                valid_pred_depth = out_depth[gt_mask_expanded].unsqueeze(1)  # [B, ]
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                # gt_mask_expanded = depth_mask
                valid_gt_depth = batch_depth[gt_mask_expanded].unsqueeze(1)  # [B, ]
                valid_pred_depth = out_depth[gt_mask_expanded].unsqueeze(1)  # [B, ]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # mask loss
            if self.C(self.cfg.loss.lambda_mask) > 0:
                set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"].squeeze(-1)))
            
            # mask binary cross loss
            if self.C(self.cfg.loss.lambda_mask_binary) > 0:
                set_loss("mask_binary", F.binary_cross_entropy(
                    out["opacity"].clamp(1.0e-5, 1.0 - 1.0e-5),
                    batch["mask"].squeeze(0).float(), ))


        elif guidance == "guidance" and self.true_global_step > self.cfg.freq.no_diff_steps:
            if self.cfg.stage == "geometry" and render_type == "normal":
                guidance_inp = out["comp_normal_viewspace"]
            else:
                guidance_inp = out["comp_rgb"]

            normal_cond = out["comp_normal_viewspace"]

            guidance_out = self.guidance(
                guidance_inp,
                control_img,
                self.prompt_utils,
                **batch,
                normal_condition=normal_cond,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
                mask=out["mask"] if "mask" in out else None,
            )

            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    set_loss(name.split("_")[-1], value)

        # Regularization
        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

        elif self.cfg.stage == "geometry":
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        if self.cfg.stage == "geometry":
            render_type = "normal"
            if self.true_global_step < self.cfg.freq.fit_step:
                do_ref = True
                do_guidance = False
                do_ref_guidance = False
            else:   # alternate
                do_ref = False
                do_ref_guidance = (
                        self.true_global_step < self.cfg.freq.guidance_only_steps
                        or self.true_global_step % self.cfg.freq.n_ref == 0
                )
                do_guidance = not do_ref_guidance
        else:
            raise Exception("The stage must be \"geometry\" ")

        total_loss = 0.0
        if do_guidance:
            out = self.training_substep(batch, batch_idx, guidance="guidance", render_type=render_type)
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref", render_type=render_type)
            total_loss += out["loss"]

        if do_ref_guidance:
            out = self.training_substep(batch, batch_idx, guidance="ref_guidance", render_type=render_type)
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {}
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],

            name="validation_step",
            step=self.true_global_step,
        )

        if self.cfg.stage == "texture" and self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"

        try:
            self.save_img_sequence(
                filestem,
                filestem,
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="validation_epoch_end",
                step=self.true_global_step,
            )
            shutil.rmtree(
                os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
            )
        except:
            pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale", "img": out["depth"][0], "kwargs": {}
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale", "img": out["opacity_vis"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)}
                    }
                ]
                if "opacity_vis" in out
                else []
            )
            ,
            name="test_step",
            step=self.true_global_step,
        )

        # FIXME: save camera extrinsics
        c2w = batch["c2w"]
        save_path = os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test/{batch['index'][0]}.npy")
        np.save(save_path, c2w.detach().cpu().numpy()[0])

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

    def on_before_optimizer_step(self, optimizer) -> None:
        # print("on_before_opt enter")
        # for n, p in self.geometry.named_parameters():
        #     if p.grad is None:
        #         print(n)
        # print("on_before_opt exit")

        pass

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance." + k: v for (k, v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return