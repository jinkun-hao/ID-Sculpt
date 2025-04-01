import os
import random
import shutil
from dataclasses import dataclass, field
import cv2
import clip
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import time
import tqdm
import shutil
import numpy as np
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
# import kiui

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.misc import get_rank, get_device, load_module_weights
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.draw_laion import DrawLaion
from threestudio.utils.tvloss import tv_loss
from threestudio.utils.sr import sr
from threestudio.utils.inpaint import mipmap_linear_grid_put_2d, orbit_camera, OrbitCamera, dilate_image


@threestudio.register("head3d-inpaint-style-system")
class ImageConditionDreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture'].
        # Note that in the paper we consolidate 'coarse' and 'geometry' into a single phase called 'geometry-sculpting'.
        stage: str = "coarse"
        freq: dict = field(default_factory=dict)

        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)

        guidance_ref_type: str = ""
        guidance_ref: dict = field(default_factory=dict)

        guidance_clip_type: str = ""
        guidance_clip: dict = field(default_factory=dict)

        prompt_processor_ref_type: str = ""
        prompt_processor_ref: dict = field(default_factory=dict)

        prompt_processor_clip_type: str = ""
        prompt_processor_clip: dict = field(default_factory=dict)

        use_mixed_camera_config: bool = False

        control_guidance_type: str = ""
        control_guidance: dict = field(default_factory=dict)
        control_prompt_processor_type: str = ""
        control_prompt_processor: dict = field(default_factory=dict)

        visualize_samples: bool = False

        opt: dict = field(default_factory=dict)
        # camera_path: str = 'default'

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        self.cfg.guidance_3d_type = ""  # try to not use zero123
        if self.cfg.guidance_3d_type != "":
            self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(
                self.cfg.guidance_3d
            )
        else:
            self.guidance_3d = None

        if not (self.cfg.guidance_clip_type == ""):
            self.clip_guidance = threestudio.find(self.cfg.guidance_clip_type)(self.cfg.guidance_clip)
            self.prompt_processor_clip = threestudio.find(self.cfg.prompt_processor_clip_type)(
                self.cfg.prompt_processor_clip
            )
            self.prompt_clip_utils = self.prompt_processor_clip()
        else:
            self.clip_guidance = None
            self.prompt_clip_utils = None

        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        # self.prompt_processor_clip = threestudio.find(self.cfg.prompt_processor_clip_type)(
        #     self.cfg.prompt_processor_clip
        # )
        # self.prompt_clip_utils = self.prompt_processor_clip()

        if not (self.cfg.guidance_ref_type == ""):
            self.guidance_ref = threestudio.find(self.cfg.guidance_ref_type)(self.cfg.guidance_ref)
            self.prompt_processor_ref = threestudio.find(self.cfg.prompt_processor_ref_type)(
                self.cfg.prompt_processor_ref
            )
            self.prompt_ref_utils = self.prompt_processor_ref()
        else:
            self.guidance_ref = None
            self.prompt_processor_ref = None

        # p_config = {}
        # self.perceptual_loss = threestudio.find("perceptual-loss")(p_config)

        self.draw_laion = DrawLaion()

        if not (self.cfg.control_guidance_type == ""):
            self.control_guidance = threestudio.find(self.cfg.control_guidance_type)(self.cfg.control_guidance)
            self.control_prompt_processor = threestudio.find(self.cfg.control_prompt_processor_type)(
                self.cfg.control_prompt_processor
            )
            self.control_prompt_utils = self.control_prompt_processor()

        self.cam = OrbitCamera(self.cfg.opt.width, self.cfg.opt.height, r=self.cfg.opt.camera_distance,
                               fovy=self.cfg.opt.fovy_deg)

    def configure_optimizers(self):
        dummy_param = nn.Parameter(torch.empty(0), requires_grad=True)
        # import pdb; pdb.set_trace()
        optimizer = torch.optim.AdamW([dummy_param], betas=[0.9, 0.99], eps=0.0001)
        # optim = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            "optimizer": optimizer,
        }
        return ret

    def add_param_to_optimizer(self, optimizer, new_param):
        optimizer.add_param_group(new_param)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # import pdb
        # pdb.set_trace()
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

    @torch.no_grad()
    def deblur(self, ratio=2):
        h = w = int(self.cfg.opt.texture_size)
        # overall deblur by LR then SR

        cur_albedo = self.renderer.mesh.albedo.detach().cpu().numpy()
        cur_albedo = (cur_albedo * 255).astype(np.uint8)
        cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
        cur_albedo = sr(cur_albedo, scale=ratio, cache_dir=self.cfg.opt.sr_dir)
        cur_albedo = cur_albedo.astype(np.float32) / 255

        cur_albedo = torch.from_numpy(cur_albedo).to(self.device)

        # enhance quality by SD refine...
        # kiui.vis.plot_image(albedo.detach().cpu().numpy())
        # text_embeds = self.guidance_embeds if not self.opt.text_dir else self.guidance_embeds['default']
        # albedo = self.guidance.refine(text_embeds, albedo.permute(2,0,1).unsqueeze(0).contiguous(), strength=0.8).squeeze(0).permute(1,2,0).contiguous()
        # kiui.vis.plot_image(albedo.detach().cpu().numpy())

        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def dilate_texture(self):
        h = w = int(self.cfg.opt.texture_size)

        mask = self.cnt.squeeze(-1) > 0

        ## dilate texture
        mask = mask.view(h, w)
        mask = mask.detach().cpu().numpy()

        self.albedo = dilate_image(self.albedo, mask, iterations=int(h * 0.2))
        self.cnt = dilate_image(self.cnt, mask, iterations=int(h * 0.2))

        self.update_mesh_albedo()

    @torch.no_grad()
    def update_mesh_albedo(self):
        mask = self.cnt.squeeze(-1) > 0
        cur_albedo = self.albedo.clone()
        cur_albedo[mask] /= self.cnt[mask].repeat(1, 3)
        self.renderer.mesh.albedo = cur_albedo

    def save_model(self, save_dir=None, filename='head.obj'):
        os.makedirs(self.cfg.opt.outdir, exist_ok=True)
        # path = os.path.join(save_dir, self.cfg.opt.save_path)
        path = os.path.join(save_dir, filename)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
        return path

    @torch.no_grad()
    def inpaint_view(self, batch, is_ref=False):

        h = w = 1024  # hardcoded
        H = batch['height']
        W = batch['width']

        out = self(batch)

        # valid crop region with fixed aspect ratio
        valid_pixels = out['alpha'].squeeze(-1).nonzero()  # [N, 2]
        min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
        min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()

        size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
        h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        min_h = int(h_start)
        min_w = int(w_start)
        max_h = int(min_h + size)
        max_w = int(min_w + size)

        # crop region is outside rendered image: do not crop at all.
        if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
            min_h = 0
            min_w = 0
            max_h = H
            max_w = W

        def _zoom(x, mode='bilinear', size=(H, W)):
            return F.interpolate(x[..., min_h:max_h + 1, min_w:max_w + 1], size, mode=mode)

        image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous())  # [1, 3, H, W]

        image_np = image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        image_render = (image_np * 255).astype(np.uint8)  # 将图像张量从 [0,1] 范围映射到 [0,255] 并转为整数类型
        # image_pil = Image.fromarray(image_np)
        # image_pil.save('vis/rendered_image.png')

        # trimap: generate, refine, keep
        mask_generate = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest')  # [1, 1, H, W]
        # mask_generate_inpaint = _zoom(out['cnt_inpaint'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest')  # [1, 1, H, W]
        # import pdb
        # pdb.set_trace()
        mask_generate = mask_generate < 0.2
        # mask_generate_inpaint = mask_generate_inpaint < 0.2
        # mask_add = _zoom(out['alpha'].permute(2, 0, 1).unsqueeze(0).contiguous()) == 0
        # mask_generate = mask_generate | mask_add

        # viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous())  # [1, 1, H, W]
        # viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous())  # [1, 1, H, W]
        # mask_refine = ((viewcos_old < viewcos) & ~mask_generate)

        # mask_keep = (~mask_generate & ~mask_refine)
        mask_keep = ~mask_generate
        # mask_keep_inpaint = ~mask_generate_inpaint

        mask_generate = mask_generate.float()
        # mask_refine = mask_refine.float()
        mask_keep = mask_keep.float()

        # dilate and blur mask
        # blur_size = 9
        # mask_generate_blur = dilation(mask_generate, kernel=torch.ones(blur_size, blur_size, device=mask_generate.device))
        # mask_generate_blur = gaussian_blur(mask_generate_blur, kernel_size=blur_size, sigma=5) # [1, 1, H, W]
        # mask_generate[mask_generate > 0.5] = 1 # do not mix any inpaint region
        mask_generate_blur = mask_generate

        # weight map for mask_generate
        # mask_weight = (mask_generate > 0.5).float().cpu().numpy().squeeze(0).squeeze(0)
        # mask_weight = ndimage.distance_transform_edt(mask_weight)#.clip(0, 30) # max pixel dist hardcoded...
        # mask_weight = (mask_weight - mask_weight.min()) / (mask_weight.max() - mask_weight.min() + 1e-20)
        # mask_weight = torch.from_numpy(mask_weight).to(self.device).unsqueeze(0).unsqueeze(0)

        # mask_refine_np = mask_refine.squeeze().cpu().detach().numpy()
        mask_generate_np = mask_generate.squeeze().cpu().detach().numpy()
        mask_keep_np = mask_keep.squeeze().cpu().detach().numpy()
        # mask_keep_inpaint_np = mask_keep_inpaint.squeeze().cpu().detach().numpy()
        # 将mask的值从[0, 1]缩放到[0, 255]并转换为uint8
        # mask_refine_np = (mask_refine_np * 255).astype(np.uint8)
        mask_generate_np = (mask_generate_np * 255).astype(np.uint8)
        mask_keep_np = (mask_keep_np * 255).astype(np.uint8)
        mask_keep_np = np.repeat(mask_keep_np[:, :, np.newaxis], 3, axis=2)
        # mask_keep_inpaint_np = (mask_keep_inpaint_np * 255).astype(np.uint8)
        # mask_keep_inpaint_np = np.repeat(mask_keep_inpaint_np[:, :, np.newaxis], 3, axis=2)
        # # 合并三个mask
        # mask_np = np.concatenate((mask_refine_np, mask_generate_np, mask_keep_np), axis=1)
        # mask_np_expanded = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)

        # 转换为PIL Image，注意这里假设mask是单通道的
        # mask_image = Image.fromarray(mask_np, mode='L') # 'L'模式表示单通道图像

        # 保存为PNG文件
        # mask_image.save('vis/mask_image.png')
        # import pdb; pdb.set_trace()
        # kiui.vis.plot_matrix(mask_generate, mask_refine, mask_keep)

        if not (mask_generate > 0.5).any():
            return

        control_images = {}
        # construct normal control
        if 'normal' in self.cfg.guidance.control_type:
            rot_normal = out['rot_normal']  # [H, W, 3]
            # import pdb
            # pdb.set_trace()
            # control_img = rot_normal.detach().cpu().numpy() * 255
            # control_img = control_img.astype('uint8')
            # from PIL import Image
            # pil_image = Image.fromarray(control_img)
            # pil_image.save('vis/test_norm.png')
            rot_normal[..., 0] *= -1  # align with normalbae: blue = front, red = left, green = top
            # control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5, size=(512, 512)) # [1, 3, H, W]
            control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5, size=(512, 512))

        # construct depth control
        if 'depth' in self.cfg.guidance.control_type:
            depth = out['depth']
            control_images['depth'] = _zoom(depth.view(1, 1, H, W), size=(512, 512)).repeat(1, 3, 1,
                                                                                                1)  # [1, 3, H, W]

        if not is_ref:
            # construct inpaint control
            image_generate = image.clone()  # torch.Size([1, 3, 512, 512])
            image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1  # -1 is inpaint region
            image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
            control_images['inpaint'] = image_generate
            # masked_image = image * (mask_generate < 0.5)
            # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
            latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear')  # [1, 1, 64, 64]
            # latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear')  # [1, 1, 64, 64]
            latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear')  # [1, 1, 64, 64]
            # control_images['masked_image'] = masked_image
            control_images['latents_mask'] = latents_mask
            # control_images['latents_mask_refine'] = latents_mask_refine
            control_images['latents_mask_keep'] = latents_mask_keep

            # image = image + torch.randn_like(image) * (1 - mask_keep)
            control_images['latents_original'], _ = self.guidance.get_latents(
                F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(
                    self.guidance.weights_dtype))  # [1, 4, 64, 64]

            # prompt to reject & regenerate
            # rgbs = self.guidance(text_embeds, height=512, width=512, control_images=control_images, refine_strength=self.opt.refine_strength).float()

            rgbs_raw = self.guidance(
                img=image,
                control_images=control_images,
                prompt_utils=self.prompt_utils,
                **batch,
            ).float()
        else:
            # mask_keep_temp = batch["mask"].permute(0, 3, 1, 2)
            # latents_mask_keep = F.interpolate(mask_keep_temp, size=(64, 64), mode='bilinear')
            # control_images['latents_mask_keep'] = latents_mask_keep
            # image = batch['rgb'].permute(0, 3, 1, 2)
            # control_images['latents_original'], _ = self.guidance.get_latents(
            #     F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(
            #         self.guidance.weights_dtype))  # [1, 4, 64, 64]
            # rgbs_raw = self.guidance(
            #     img=image,
            #     control_images=control_images,
            #     prompt_utils=self.prompt_utils,
            #     **batch,
            # ).float()
            rgbs_raw = batch['rgb'].permute(0, 3, 1, 2)

        # # performing upscaling (assume 2/4/8x)
        # if rgbs.shape[-1] != W or rgbs.shape[-2] != H:
        #     scale = W // rgbs.shape[-1]
        rgbs_np = rgbs_raw.detach().cpu().squeeze(0).permute(1, 2, 0).contiguous().numpy()
        rgbs_np = (rgbs_np * 255).astype(np.uint8)
        #     rgbs = kiui.sr.sr(rgbs, scale=scale)
        #     rgbs = rgbs.astype(np.float32) / 255
        #     rgbs = torch.from_numpy(rgbs).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)

        # apply mask to make sure non-inpaint region is not changed
        rgbs = rgbs_raw * (1 - mask_keep) + image * mask_keep
        # rgbs = rgbs * mask_generate_blur + image * (1 - mask_generate_blur)

        # project-texture mask

        cos_np = out['viewcos'].squeeze().cpu().numpy()
        cos_np = (cos_np * 255).astype(np.uint8)  # 将图像张量从 [0,1] 范围映射到 [0,255] 并转为整数类
        cos_np_expanded = np.repeat(cos_np[:, :, np.newaxis], 3, axis=2)
        # 将numpy数组转换为PIL Image
        # image_pil = Image.fromarray(image_np, mode='L')
        # 保存为PNG文件
        # image_pil.save('vis/viewcos1.png')

        # proj_mask = (out['alpha'] > 0) & (out['viewcos'] > self.cfg.guidance.cos_thresh)  # [H, W, 1]
        proj_mask_ori = (out['alpha'] > 0) & (out['viewcos'] > self.cfg.guidance.cos_thresh)  # [H, W, 1]
        # import pdb;pdb.set_trace()

        proj_mask_np = proj_mask_ori.cpu().numpy().squeeze()  # 去掉单通道维度，得到(512, 512)的二维数组
        kernel = np.ones((5, 5), np.uint8)
        # 使用OpenCV的erode函数进行腐蚀操作
        eroded_mask_np = cv2.erode(proj_mask_np.astype(np.uint8), kernel, iterations=8)
        # 将腐蚀后的NumPy数组转换回PyTorch张量，并确保它的dtype是布尔类型
        proj_mask = torch.from_numpy(eroded_mask_np.astype(bool)).to(dtype=torch.bool, device=proj_mask_ori.device)[:, :, None]
        proj_mask_temp = proj_mask
        backproj_mask = proj_mask.float() * (1 - mask_keep.squeeze(0).permute(1, 2, 0)) # H, W, C

        proj_mask_ori_np = proj_mask_ori.squeeze().cpu().detach().numpy()
        proj_mask_np = proj_mask.squeeze().cpu().detach().numpy()
        # 将mask的值从[0, 1]缩放到[0, 255]并转换为uint8
        proj_mask_ori_np = (proj_mask_ori_np * 255).astype(np.uint8)
        proj_mask_np = (proj_mask_np * 255).astype(np.uint8)
        # 合并三个mask
        mask_np = np.concatenate((proj_mask_ori_np, proj_mask_np), axis=1)
        mask_np_expanded = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)

        # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())

        rgb_inpaint = ((rgbs_raw.squeeze(0).permute(1, 2, 0) * proj_mask) * 255).cpu().detach().numpy().astype(np.uint8)

        proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
        proj_mask_ori = _zoom(proj_mask_ori.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
        backproj_mask = backproj_mask.view(1, 1, H, W).float().view(-1).bool()

        uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')

        uvs_real = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[backproj_mask]
        rgbs_real = rgbs_raw.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[backproj_mask]

        mask_keep_inpaint = (1 - mask_keep).view(-1).bool()


        uvs_inpaint = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[mask_keep_inpaint]
        rgbs_inpaint = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[mask_keep_inpaint]

        cur_albedo_real, cur_cnt_real = mipmap_linear_grid_put_2d(h, w, uvs_real[..., [1, 0]] * 2 - 1, rgbs_real,
                                                                  min_resolution=256,
                                                                  return_count=True)
        cur_albedo_inpaint, cur_cnt_inpaint = mipmap_linear_grid_put_2d(h, w, uvs_inpaint[..., [1, 0]] * 2 - 1,
                                                                        rgbs_inpaint, min_resolution=256,
                                                                        return_count=True)

        mask = cur_cnt_real.squeeze(-1) > 0
        mask_inpaint = cur_cnt_inpaint.squeeze(-1) > 0

        self.albedo[mask] = cur_albedo_real[mask]
        self.albedo_inpaint[mask_inpaint] = cur_albedo_inpaint[mask_inpaint]
        self.cnt[mask] = cur_cnt_real[mask]
        self.cnt_inpaint[mask_inpaint] = cur_cnt_inpaint[mask_inpaint]

        # update mesh texture for rendering
        mask = self.cnt.squeeze(-1) > 0
        cur_albedo = self.albedo.clone()
        cur_albedo[mask] /= self.cnt[mask].repeat(1, 3)

        mask_inpaint = self.cnt_inpaint.squeeze(-1) > 0
        cur_albedo_inpaint = self.albedo_inpaint.clone()
        cur_albedo_inpaint[mask_inpaint] /= self.cnt_inpaint[mask_inpaint].repeat(1, 3)

        self.renderer.mesh.albedo_inpaint = cur_albedo_inpaint
        self.renderer.mesh.albedo = cur_albedo

        cnt_np = self.cnt.squeeze().cpu().detach().numpy()
        cnt_np = (cnt_np * 255).astype(np.uint8)
        cnt_np_expanded = np.repeat(cnt_np[:, :, np.newaxis], 3, axis=2)

        albedo_np = cur_albedo.squeeze().cpu().numpy()
        albedo_np = (albedo_np * 255).astype(np.uint8)  # 将图像张量从 [0,1] 范围映射到 [0,255] 并转为整数类型

        cur_albedo_real_np = cur_albedo_real.squeeze().cpu().numpy()
        cur_albedo_real_np = (cur_albedo_real_np * 255).astype(np.uint8)

        # import pdb; pdb.set_trace()
        # proj_mask_expend = np.repeat(proj_mask_np[:, :, np.newaxis], 3, axis=2)
        backproj = proj_mask_temp.float() * (1 - mask_keep.squeeze(0).permute(1, 2, 0))
        backproj_np = backproj.cpu().numpy().squeeze()
        backproj_np = (backproj_np * 255).astype(np.uint8)
        backproj_np_expend = np.repeat(backproj_np[:, :, np.newaxis], 3, axis=2)

        # combined_image_np = np.concatenate(
        #     (mask_np_expanded, backproj_np_expend, image_render, mask_keep_np, rgb_inpaint, rgbs_np, cos_np_expanded), axis=1)
        # combined_tex_np = np.concatenate((cnt_np_expanded, albedo_np, cur_albedo_real_np), axis=1)
        # image_pil = Image.fromarray(combined_image_np)
        # combined_tex_pil = Image.fromarray(combined_tex_np)
        # combined_tex_pil.save('vis/tex.png')
        # image_pil.save('vis/exp_out.png')
        # kiui.vis.plot_image(cur_albedo.detach().cpu().numpy())

        # update viewcos cache
        # viewcos = viewcos.view(-1, 1)[proj_mask]
        # cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs_real[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.cnt = self.cnt.clone()
        self.renderer.mesh.cnt_inpaint = self.cnt_inpaint.clone()
        # self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)

    def initialize_texture(self, batch):
        h = w = int(self.cfg.opt.texture_size)
        self.albedo = 0.5 * torch.ones((h, w, 3), device=self.device, dtype=torch.float32)
        self.albedo_inpaint = 0.5 * torch.ones((h, w, 3), device=self.device, dtype=torch.float32)
        self.cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)
        self.cnt_inpaint = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

        if self.cfg.opt.camera_path == 'default':
            vers = [-15] * 10  + [45] * 5
            hors = [0, -45, 45, -90, 90, 120, -120, 150, -150, 180] + [0, 90, -90, 135, -135]
            # vers = [-15]
            # hors = [25]
        elif self.cfg.opt.camera_path == 'front':
            vers = [0] * 8 + [-89.9, 89.9] + [45]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif self.cfg.opt.camera_path == 'top':
            vers = [0, -45, 45, -89.9, 89.9] + [0] + [0] * 6
            hors = [0] * 5 + [180] + [45, -45, 90, -90, 135, -135]
        elif self.cfg.opt.camera_path == 'side':
            vers = [0, 0, 0, 0, 0] + [-45, 45, -89.9, 89.9] + [-45, 0]
            hors = [0, 45, -45, 90, -90] + [0, 0, 0, 0] + [180, 180]
        else:
            raise NotImplementedError(f'camera path {self.opt.camera_path} not implemented!')

        start_t = time.time()

        print(f'[INFO] start generation...')
        for ver, hor in tqdm.tqdm(zip(vers, hors), total=len(vers)):
            # render image
            pose = orbit_camera(ver, hor, self.cfg.opt.camera_distance)
            mvp = self.cam.perspective @ np.linalg.inv(pose)
            mvp = torch.from_numpy(mvp).float().unsqueeze(0).to(self.device)
            batch['mvp_mtx'] = mvp
            batch['azimuth'] = torch.tensor([hor]).float().to(self.device)
            batch['elevation'] = torch.tensor([ver]).float().to(self.device)

            self.inpaint_view(batch)

        self.dilate_texture()
        self.deblur()

        torch.cuda.synchronize()
        end_t = time.time()
        print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')

        os.makedirs(self.get_save_dir(), exist_ok=True)
        self.save_model(save_dir=self.get_save_dir(), filename=self.cfg.opt.save_path.split('.')[0] + '_init.obj')

    def training_substep(self, batch, batch_idx, guidance: str, render_type="rgb"):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "guidance"
        """
        # import pdb
        # pdb.set_trace()
        gt_mask = batch["mask"] > 0.5  # float to bool
        gt_rgb = batch["rgb"]
        gt_depth = batch["ref_depth"]
        gt_normal = batch["ref_normal"]
        mvp_mtx_ref = batch["mvp_mtx"]
        c2w_ref = batch["c2w4x4"]
        ldmk = batch["face_landmarks"].squeeze(0)

        if guidance == "guidance":
            batch = batch["random_camera"]
            mvp = batch["mvp_mtx"].clone()
            azimuth = batch["azimuth"].clone().item()
            control_img = self.draw_laion.draw(azimuth, mvp, ldmk, gt_rgb.shape[1], gt_rgb.shape[2])

        if guidance == "ref_guidance":
            batch = batch['ref_control_img']
            gt_normal = batch["normal"]
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
            if render_type == "rgb":
                # color loss. Use l2 loss in coarse and geometry satge; use l1 loss in texture stage.
                if self.C(self.cfg.loss.lambda_rgb) > 0:
                    gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                            1 - gt_mask.float()
                    )
                    pred_rgb = out["image"].unsqueeze(0)

                    
                    # grow_mask = F.max_pool2d(1 - gt_mask.float().permute(0, 3, 1, 2), (9, 9), 1, 4)
                    grow_mask = gt_mask.float()
                    grow_mask_uint8 = (grow_mask.detach().cpu() * 255).to(torch.float32).clamp(0, 255).squeeze().to(torch.uint8)
                    grow_mask_expanded = np.repeat(grow_mask_uint8[:, :, np.newaxis], 3, axis=2)

                    gt_rgb_uint8 = (gt_rgb.detach().cpu() * 255).to(torch.float32).clamp(0, 255)[0].to(torch.uint8)
                    pred_rgb_uint8 = (pred_rgb.detach().cpu() * 255).to(torch.float32).clamp(0, 255)[0].to(torch.uint8)
                    concate_img = np.concatenate((gt_rgb_uint8, pred_rgb_uint8, grow_mask_expanded), axis=1)
                    to_pil = transforms.ToPILImage()
                    pil_image = to_pil(concate_img)
                    pil_image.save("vis/normal_test_normal79.png")

                    set_loss('tv', tv_loss(out["image"].unsqueeze(0).permute(0, 3, 1, 2)))

                    if self.cfg.stage in ["coarse", "geometry"]:
                        set_loss("rgb", F.mse_loss(gt_rgb, pred_rgb))
                    else:
                        if self.cfg.stage == "texture":
                            # grow_mask = F.max_pool2d(1 - gt_mask.float().permute(0, 3, 1, 2), (9, 9), 1, 4)
                            # grow_mask = 
                            grow_mask = gt_mask.float()
                            set_loss("rgb", F.l1_loss(gt_rgb * grow_mask, pred_rgb * grow_mask))
                            # set_loss("rgb", F.mse_loss((gt_rgb * grow_mask).float(), (pred_rgb * grow_mask).float()))
                            # print(f"loss_rgb:{F.mse_loss((gt_rgb * grow_mask).float(), (pred_rgb * grow_mask).float())}")
                        else:
                            set_loss("rgb", F.l1_loss(gt_rgb, pred_rgb))

                # mask loss
                if self.C(self.cfg.loss.lambda_mask) > 0:
                    set_loss("mask", F.mse_loss(gt_mask.float(), out["alpha"]))

                # mask binary cross loss
                if self.C(self.cfg.loss.lambda_mask_binary) > 0:
                    set_loss("mask_binary", F.binary_cross_entropy(
                        out["alpha"].clamp(1.0e-5, 1.0 - 1.0e-5),
                        batch["mask"].float(), ))

            # normal loss
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

            # mask loss
            if self.C(self.cfg.loss.lambda_mask) > 0:
                set_loss("mask", F.mse_loss(gt_mask.float(), out["alpha"].squeeze(-1)))

            # mask binary cross loss
            if self.C(self.cfg.loss.lambda_mask_binary) > 0:
                set_loss("mask_binary", F.binary_cross_entropy(
                    out["alpha"].clamp(1.0e-5, 1.0 - 1.0e-5),
                    batch["mask"].squeeze(0).float(), ))


        elif guidance == "guidance" and self.true_global_step > self.cfg.freq.no_diff_steps:
            # if self.cfg.stage == "geometry" and render_type == "normal":
            #     # guidance_inp = out["comp_normal"][..., [1, 2, 0]]
            #     # normal_cond = None
            #     guidance_inp = out["comp_normal_viewspace"]
            # else:
            #     guidance_inp = out["comp_rgb"]

            control_images = {}
            rot_normal = out['rot_normal']  # [H, W, 3]
            rot_normal[..., 0] *= -1  # align with normalbae: blue = front, red = left, green = top
            control_images['normal'] = rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous()
            image = out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, 3, H, W]
            # image_fixed = out['image_fixed'].permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, 3, H, W]

            control_images['latents_original'], _ = self.guidance.get_latents(
                F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(
                    self.guidance.weights_dtype))  # [1, 4, 64, 64]
            # control_images['latents_fixed'], _ = self.guidance.get_latents(
            #     F.interpolate(image_fixed, (512, 512), mode='bilinear', align_corners=False).to(
            #         self.guidance.weights_dtype))  # [1, 4, 64, 64]

            guidance_out = self.guidance(
                img=image,
                img_fixed=None,
                control_images=control_images,
                prompt_utils=self.prompt_utils,
                is_inpaint=False,
                **batch,
            )

            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    set_loss(name.split("_")[-1], value)

            # if self.cfg.stage == "texture" and self.clip_guidance is not None:
            #     # azimuth = batch["azimuth"]
            #     # if azimuth < -90 or azimuth > 90:
            #     guidance_clip_loss = self.clip_guidance(
            #         guidance_inp,
            #         gt_rgb,
            #         self.prompt_clip_utils,
            #         embedding_type='both',
            #         **batch,
            #     )
            #     self.log(f"train/loss_clip", guidance_clip_loss)
            #     set_loss('clip', guidance_clip_loss)

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

        elif self.cfg.stage == "texture":
            if self.C(self.cfg.loss.lambda_reg) > 0 and guidance == "guidance" and self.true_global_step % 5 == 0:
                rgb = out["comp_rgb"]
                rgb = F.interpolate(rgb.permute(0, 3, 1, 2), (512, 512), mode='bilinear').permute(0, 2, 3, 1)
                control_prompt_utils = self.control_prompt_processor()
                with torch.no_grad():
                    control_dict = self.control_guidance(
                        rgb=rgb,
                        cond_rgb=rgb,
                        prompt_utils=control_prompt_utils,
                        mask=out["mask"] if "mask" in out else None,
                    )

                    edit_images = control_dict["edit_images"]
                    temp = (edit_images.detach().cpu()[0].numpy() * 255).astype(np.uint8)
                    cv2.imwrite(".threestudio_cache/control_debug.jpg", temp[:, :, ::-1])

                loss_reg = (rgb.shape[1] // 8) * (rgb.shape[2] // 8) * self.perceptual_loss(
                    edit_images.permute(0, 3, 1, 2), rgb.permute(0, 3, 1, 2)).mean()
                set_loss("reg", loss_reg)
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
        # init texture
        if self.global_step == 0:
            self.renderer.mesh = self.geometry.isosurface()
            self.initialize_texture(batch)
            self.renderer.perpare_albedo()
            self.add_param_to_optimizer(self.trainer.optimizers[0], self.renderer.get_params()[0])
            # self.add_param_to_optimizer(self.trainer.optimizers[0], self.renderer.get_params()[1])
        # self.mesh = 
        # self.renderer.mesh = self.geometry.isosurface()

        # 先试batchsize为1的情况
        if self.cfg.freq.ref_or_guidance == "accumulate":
            do_ref = True
            do_guidance = True
        elif self.cfg.freq.ref_or_guidance == "alternate":
            do_ref = (
                    self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_guidance = not do_ref
            if hasattr(self.guidance.cfg, "only_pretrain_step"):
                if (self.guidance.cfg.only_pretrain_step > 0) and (
                        self.global_step % self.guidance.cfg.only_pretrain_step) < (
                        self.guidance.cfg.only_pretrain_step // 5):
                    do_guidance = True
                    do_ref = False
        render_type = "rgb"

        total_loss = 0.0
        if do_guidance:
            out = self.training_substep(batch, batch_idx, guidance="guidance", render_type=render_type)
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref", render_type=render_type)
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
                        "img": batch["image"],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "image" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["image"],
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
                        "img": out["normal"],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["rot_normal"],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "rot_normal" in out
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
                    "img": out["alpha"][:, :, 0],
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
                        "img": batch["image"],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "image" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["image"],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "image" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["normal"],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["rot_normal"],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "rot_normal" in out
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
                    "img": out["alpha"][:, :, 0],
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
        self.save_model(save_dir=self.get_save_dir(), filename=self.cfg.opt.save_path.split('.')[0] + '_final.obj')

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