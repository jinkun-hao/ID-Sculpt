import os
import random
import copy
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
# from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule, BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from threestudio.utils.perceptual import PerceptualLoss

import sys

from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("sd-controlnet-vsd-ip-inpaint-guidance")
class ControlNetVSDGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "../../../sd_model/models--SG161222--Realistic_Vision_V4.0_noVAE/snapshots/980a16dee9c208293b990744e245efcf1fb54035"
        pretrained_model_name_or_path_lora: str = "../../../sd_model/models--SG161222--Realistic_Vision_V4.0_noVAE/snapshots/980a16dee9c208293b990744e245efcf1fb54035"
        vae_model_path: str = "../../../sd_model/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493"
        ip_ckpt: str = "../../../IP_Adapter/models/ip-adapter-faceid-portrait-v11_sd15.bin"

        ddim_scheduler_name_or_path: str = "../../../sd_model/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
        control_type: str = "landmark"  # normal/canny

        control_image_path: str = ''

        # ip adapter params
        insightface_root: str = None
        pil_image_path: str = ''
        IP_scale: float = 1.0

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        guidance_scale_lora: float = 1.0
        condition_scale: float = 1.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        sqrt_anneal: bool = False  # sqrt anneal proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        trainer_max_steps: int = 25000
        use_img_loss: bool = False  # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/

        diffusion_steps: int = 20

        use_sds: bool = False

        cos_thresh: float = 0.0

        # Canny threshold
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        controlnet_name_or_path: str
        if self.cfg.control_type in ("normal", "input_normal"):
            controlnet_name_or_path = "../../../sd_model/models--lllyasviel--control_v11p_sd15_normalbae/snapshots/cb7296e6587a219068e9d65864e38729cd862aa8"
        elif self.cfg.control_type == "canny":
            controlnet_name_or_path = "../../../sd_model/models--lllyasviel--control_v11p_sd15_canny/snapshots/115a470d547982438f70198e353a921996e2e819"
        elif self.cfg.control_type == "landmark":
            controlnet_name_or_path = "../../../sd_model/models--CrucibleAI--ControlNetMediaPipeFace/snapshots/f6ed75cc495674bea8bf7409ef3d0e5bfb7d8c90"
        elif self.cfg.control_type == "depth":
            controlnet_name_or_path = "../../../sd_model/models--lllyasviel--control_v11f1p_sd15_depth/snapshots/539f99181d33db39cf1af2e517cd8056785f0a87"
        

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        pipe_lora_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        controlnet = ControlNetModel.from_pretrained(
            controlnet_name_or_path,
            torch_dtype=self.weights_dtype,
        )

        vae = AutoencoderKL.from_pretrained(self.cfg.vae_model_path).to(dtype=torch.float16)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler_lora = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )

        @dataclass
        class SubModules:
            pipe: StableDiffusionControlNetPipeline
            pipe_lora: StableDiffusionControlNetPipeline
            ip_pipe: IPAdapterFaceID = None

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet,
            scheduler=self.scheduler,
            vae=vae,
            **pipe_kwargs
        ).to(self.device)
        # pipe_lora = StableDiffusionControlNetPipeline.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path_lora, controlnet=controlnet,
        #      scheduler=self.scheduler_lora, vae=vae, **pipe_kwargs
        # )
        # del pipe_lora.vae
        # cleanup()
        # pipe_lora.vae = pipe.vae

        if (
                self.cfg.pretrained_model_name_or_path
                == self.cfg.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        else:
            self.single_model = False
            pipe_lora = StableDiffusionControlNetPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path_lora, controlnet=controlnet,
                scheduler=self.scheduler_lora, vae=vae, **pipe_kwargs
            ).to(self.device)
            del pipe_lora.vae
            cleanup()
            pipe_lora.vae = pipe.vae

        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

        del self.pipe.text_encoder
        if not self.single_model:
            del self.pipe_lora.text_encoder
        cleanup()

        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        self.scheduler_lora.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        pipe_new = copy.deepcopy(self.pipe)
        self.ip = IPAdapterFaceID(pipe_new, self.cfg.ip_ckpt, self.device, num_tokens=16)
        # self.ip_lora = IPAdapterFaceID(pipe_new, self.cfg.ip_ckpt, self.device, num_tokens=16)
        self.ip.set_scale(self.cfg.IP_scale)
        # self.ip_lora.set_scale(self.cfg.IP_scale)
        self.submodules.ip_pipe = self.ip.pipe
        # self.submodules.ip_pipe_lora = self.ip_lora.pipe

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.ip_unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)
        # for p in self.ip_unet_lora.parameters():
        #     p.requires_grad_(False)

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            ).to(self.device)

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.image_prompt_embeds, self.uncond_image_prompt_embeds = self.prepare_ip_adapter()

        # self.control_image = cv2.imread(self.cfg.control_image_path)
        self.perceptual_loss = PerceptualLoss().to(self.device)

        threestudio.info(f"Loaded ControlNet!")

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_ip_adapter(self):

        # num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                           root=self.cfg.insightface_root)
        app.prepare(ctx_id=0, det_size=(224, 224))
        image = cv2.imread(self.cfg.pil_image_path)
        faces = app.get(image)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        n_cond = faceid_embeds.shape[1]

        image_prompt_embeds, uncond_image_prompt_embeds = self.ip.get_image_embeds(faceid_embeds)
        # [1, 16, 768]

        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.50):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def ip_pipe(self):
        return self.submodules.ip_pipe

    # @property
    # def ip_pipe_lora(self):
    #     return self.submodules.ip_pipe_lora

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def ip_unet(self):
        return self.submodules.ip_pipe.unet

    # @property
    # def ip_unet_lora(self):
    #     return self.submodules.ip_pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @property
    def controlnet(self):
        return self.submodules.pipe.controlnet

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
            self,
            controlnet: ControlNetModel,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            image_cond: Float[Tensor, "..."],
            condition_scale: float,
            encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
            self,
            unet: UNet2DConditionModel,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            encoder_hidden_states: Float[Tensor, "..."],
            cross_attention_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
            self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
            self,
            latents: Float[Tensor, "B 4 H W"],
            latent_height: int = 64,
            latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def edit_latents(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            control_images,
            t: Int[Tensor, "B"],
            is_inpaint: bool = True,
            use_ip: bool = False,
    ) -> Float[Tensor, "B 4 64 64"]:

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        threestudio.debug("Start editing...")

        self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)
            for i, t in enumerate(self.scheduler.timesteps):

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                (
                    down_block_res_samples,
                    mid_block_res_sample,
                ) = self.forward_controlnet(
                    self.controlnet,
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    image_cond=control_images[self.cfg.control_type],
                    condition_scale=self.cfg.condition_scale,
                )

                noise_pred = self.forward_control_unet(
                    self.ip_unet if use_ip else self.unet,
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

                # perform classifier-free guidance
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if is_inpaint:
                    mask_keep = control_images['latents_mask_keep']

                    latents_original = control_images['latents_original']

                    init_latents_proper = latents_original
                    if i < len(self.scheduler.timesteps) - 1:
                        noise_timestep = self.scheduler.timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )
                    # latents_original_noisy = self.scheduler.add_noise(latents_original, torch.randn_like(latents_original), t)
                    latents = latents * (1 - mask_keep) + init_latents_proper * mask_keep

            threestudio.debug("Editing finished.")
        return latents

    def prepare_image_cond(self, cond_rgb: Float[Tensor, "B H W C"]):
        if self.cfg.control_type == "normal":
            control = cond_rgb.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "canny":
            cond_rgb = (
                (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            )
            blurred_img = cv2.blur(cond_rgb, ksize=(5, 5))
            detected_map = self.preprocessor(
                blurred_img, self.cfg.canny_lower_bound, self.cfg.canny_upper_bound
            )
            control = (
                    torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            )
            control = control.unsqueeze(-1).repeat(1, 1, 3)
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif self.cfg.control_type == 'landmark':
            control = (
                    torch.from_numpy(np.array(cond_rgb)).float().to(self.device) / 255.0
            )
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unknown control type: {self.cfg.control_type}")

        return F.interpolate(control, (512, 512), mode="bilinear", align_corners=False)

    def compute_grad_vsd(
            self,
            latents: Float[Tensor, "B 4 64 64"],
            text_embeddings: Float[Tensor, "BB 77 768"],
            image_cond: Float[Tensor, "B 3 512 512"],
            use_ip: bool = False,
    ):
        B = latents.shape[0]

        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                self.controlnet,
                latent_model_input,  # torch.Size([2, 4, 64, 64])
                t,  # torch.Size([1])
                encoder_hidden_states=text_embeddings,  # torch.Size([2, 1, 77, 1024])
                image_cond=image_cond,  # torch.Size([1, 3, 512, 512])
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred_pretrain = self.forward_control_unet(
                self.ip_unet if use_ip else self.unet,
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            noise_pred_est = self.forward_control_unet(
                self.unet_lora,
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
                noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        # TODO: more general cases
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_text,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
                noise_pred_est_text - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_est)

        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            latents_denoised_pretrain = (latents_noisy - sigma * noise_pred_pretrain) / alpha
            latents_denoised_est = (latents_noisy - sigma * noise_pred_est) / alpha
            image_denoised_pretrain = self.decode_latents(latents_denoised_pretrain)
            image_denoised_est = self.decode_latents(latents_denoised_est)
            grad_img = (
                    w * (image_denoised_est - image_denoised_pretrain) * alpha / sigma
            )
        else:
            grad_img = None
        return grad, grad_img

    def train_lora(
            self,
            latents: Float[Tensor, "B 4 64 64"],
            text_embeddings: Float[Tensor, "BB 77 768"],
            image_cond: Float[Tensor, "B 3 512 512"],
            use_ip: bool = False,
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.cfg.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # use view-independent text embeddings in LoRA
        text_embeddings, _ = text_embeddings.chunk(2)

        down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
            self.controlnet,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            image_cond=image_cond,
            condition_scale=self.cfg.condition_scale,
        )

        noise_pred = self.forward_control_unet(
            self.unet_lora,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def get_latents(
            self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents, rgb_BCHW_512

    def __call__(
            self,
            img: Float[Tensor, "B C H W"],
            control_images, # dict
            # cond_rgb: Float[Tensor, "B H W C"],
            prompt_utils: PromptProcessorOutput,
            elevation: Float[Tensor, "B"],
            azimuth: Float[Tensor, "B"],
            camera_distances: Float[Tensor, "B"],
            # normal_condition=None,
            img_fixed=None,
            current_step_ratio=None,
            mask: Float[Tensor, "B H W 1"] = None,
            strength=0,
            refine_strength=0.0,
            rgb_as_latents=False,
            is_inpaint=True,
            **kwargs,
    ):
        batch_size, _, H, W = img.shape
        assert batch_size == 1

        for k in control_images:
            control_images[k] = control_images[k].to(self.weights_dtype)

        # rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents, rgb_BCHW_512 = self.get_latents(img, rgb_as_latents=rgb_as_latents)

        # temp = torch.zeros(1).to(rgb.device)
        # azimuth = kwargs.get("azimuth", temp)
        camera_distance = kwargs.get("camera_distance", camera_distances)
        view_dependent_prompt = kwargs.get("view_dependent_prompt", True)
        # text_embeddings, uncond_text_embeddings

        prompt_embeds_, negative_prompt_embeds_ = prompt_utils.get_text_embeddings(elevation, azimuth, camera_distance,
                                                                                   view_dependent_prompt,
                                                                                   if_seperate=True)

        use_ip = False

        if len(prompt_embeds_.shape) == 4:
            prompt_embeds_ = prompt_embeds_.squeeze(0)
            negative_prompt_embeds_ = negative_prompt_embeds_.squeeze(0)

        text_embeddings = torch.cat([prompt_embeds_, negative_prompt_embeds_], dim=0)

        prompt_embeds = torch.cat([prompt_embeds_, self.image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, self.uncond_image_prompt_embeds], dim=1)
        ip_text_embeddings = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)  # [2, 93, 768]

        if azimuth > -90 and azimuth < 90:
            use_ip = True

        use_ip = True

        text_embeddings = ip_text_embeddings if use_ip else text_embeddings


        if (
                self.cfg.use_sds
        ):  # did not change to vsd for backward compatibility in config files
            grad, grad_img = self.compute_grad_vsd(latents, text_embeddings, image_cond, use_ip=use_ip)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            loss_lora = self.train_lora(latents, text_embeddings, image_cond, use_ip=use_ip)
            loss_dict = {
                "loss_sd": loss_vsd,
                "loss_lora": loss_lora,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }

            if self.cfg.use_img_loss:
                grad_img = torch.nan_to_num(grad_img)
                if self.grad_clip_val is not None:
                    grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
                target_img = (rgb_BCHW_512 - grad_img).detach()
                loss_vsd_img = (
                        0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
                )
                loss_dict["loss_vsd_img"] = loss_vsd_img

            return loss_dict

        elif is_inpaint:
            latents = control_images['latents_original']

            t = torch.randint(
                950,
                951,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

            # if self.cfg.control_type == "ip_controlnet_inpaint":
            gen_latent = self.edit_latents(text_embeddings, latents, control_images, t, is_inpaint=True, use_ip=True)

            imgs = self.decode_latents(gen_latent)
            images = F.interpolate(imgs, (H, W), mode="bilinear")

            return images

        else:
            if img_fixed is not None:
                latents = control_images['latents_fixed']
            else:
                latents = control_images['latents_original']
            rgb_BCHW_512 = img
            t = torch.randint(
                50,
                200,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

            # if self.cfg.control_type == "ip_controlnet_inpaint":
            gen_latent = self.edit_latents(text_embeddings, latents, control_images, t, is_inpaint=False, use_ip=True)

            imgs = self.decode_latents(gen_latent)
            gt_rgb_BCHW = F.interpolate(imgs, (H, W), mode="bilinear")

            # loss_l1 = torch.nn.functional.l1_loss(rgb_BCHW_512, gt_rgb_BCHW.detach(), reduction='mean') / batch_size
            loss_l2 = (0.5 * F.mse_loss(rgb_BCHW_512, gt_rgb_BCHW.detach().float(), reduction='sum') / batch_size)

            # import pdb; pdb.set_trace()
            loss_p = self.perceptual_loss(rgb_BCHW_512, gt_rgb_BCHW.detach()).sum() / batch_size
            # print(f"loss_l2:{loss_l2}, loss_p:{loss_p}")
            return {
                "loss_l2": loss_l2.float(),
                "loss_p": loss_p.float(),
            }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.sqrt_anneal:
            percentage = (
                                 float(global_step) / self.cfg.trainer_max_steps
                         ) ** 0.5  # progress percentage
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                                   max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
                           ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

        

