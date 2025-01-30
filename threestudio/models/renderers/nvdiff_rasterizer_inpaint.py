from dataclasses import dataclass

import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


@threestudio.register("nvdiff-rasterizer-inpaint")
class NVDiffRasterizer(Rasterizer, nn.Module):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"
        texture_lr: float = 0.02

    cfg: Config

    def __init__(self, *args, **kwargs):
        Rasterizer.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

    def configure(
            self,
            geometry: BaseImplicitGeometry,
            material: BaseMaterial,
            background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        # default as blender grey
        # self.bg = 0.807 * torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.geometry.device)
        self.bg_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.geometry.device)
        # self.mesh = self.geometry.isosurface()
        self.mesh = None
        self.raw_albedo = None

    @torch.no_grad()
    def export_mesh(self, path):
        if self.raw_albedo is not None:
            self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
        self.mesh.write(path)

    def perpare_albedo(self):
        self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(self.mesh.albedo))

    def get_params(self):
        params = [
            {'params': self.raw_albedo, 'lr': self.cfg.texture_lr},
        ]

        return params

    def forward(
            self,
            mvp_mtx: Float[Tensor, "B 4 4"],
            camera_positions: Float[Tensor, "B 3"],
            light_positions: Float[Tensor, "B 3"],
            height: int,
            width: int,
            render_rgb: bool = True,
            render_mask: bool = False,
            **kwargs
    ) -> Dict[str, Any]:

        results = {}

        batch_size = mvp_mtx.shape[0]
        # mesh = self.geometry.isosurface()   # marchingcube

        v_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            self.mesh.v, mvp_mtx
        )

        rast, rast_db = self.ctx.rasterize(v_clip, self.mesh.f, (height, width))
        alpha = rast[..., 3:] > 0

        # rgb texture
        texc, texc_db = self.ctx.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft,
                                             rast_db=rast_db, diff_attrs='all')
        
        # pose = kwargs["c2w"][0]
        # import pdb
        # pdb.set_trace()
        # v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)


        # albedo = self.ctx.texture(self.mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear') # [1, H, W, 3]
        if self.raw_albedo is None:
            # print('self.raw_albedo is None')
            albedo = self.ctx.texture(self.mesh.albedo_inpaint.unsqueeze(0), texc, uv_da=texc_db,
                                      filter_mode='linear-mipmap-linear')  # [1, H, W, 3]
        else:
            albedo = self.ctx.texture(self.raw_albedo.unsqueeze(0), texc, uv_da=texc_db,
                                      filter_mode='linear-mipmap-linear')  # [1, H, W, 3]
            albedo = torch.sigmoid(albedo)
        vn = self.mesh.vn
        normal, _ = self.ctx.interpolate_one(vn, rast,
                                             self.mesh.fn)  # torch.Size([169370, 3]) torch.Size([241400, 3]) 120700

        normal = F.normalize(normal, dim=-1)
        # normal_aa = torch.lerp(
        #     torch.zeros_like(normal), (normal + 1.0) / 2.0, alpha.float()
        # )
        normal_aa = self.ctx.antialias(
            normal, rast, v_clip, self.mesh.f
        )

        # rotated normal (where [0, 0, 1] always faces camera)
        w2c = kwargs["c2w"][:, :3, :3].inverse()
        rot_normal = torch.einsum("bij,bhwj->bhwi", w2c, normal)
        rot_normal = F.normalize(rot_normal, dim=-1)
        bg_normal = torch.zeros_like(rot_normal)
        bg_normal[..., 2] = 1
        rot_normal_aa = torch.lerp(
            (bg_normal + 1.0) / 2.0,
            (rot_normal + 1.0) / 2.0,
            alpha.float(),
        ).contiguous()
        rot_normal_aa = self.ctx.antialias(
            rot_normal_aa, rast, v_clip, self.mesh.f
        )

        rot_normal_aa = rot_normal_aa.squeeze(0)  # [H, W, 3]
        normal_aa = normal_aa.squeeze(0)  # [H, W, 3]

        # rot normal z axis is exactly viewdir-normal cosine
        viewcos = rot_normal.squeeze(0)[..., [2]].abs()  # double-sided

        # antialias
        albedo = self.ctx.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1)  # [H, W, 3]
        alpha = self.ctx.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1)  # [H, W, 3]

        # replace background
        albedo = alpha * albedo + (1 - alpha) * self.bg
        normal_aa = alpha * normal_aa + (1 - alpha) * self.bg_normal
        rot_normal_aa = alpha * rot_normal_aa + (1 - alpha) * self.bg_normal
        # rot_normal = self.ctx.antialias(
        #     rot_normal.unsqueeze(0).contiguous(), rast, v_clip, self.mesh.f
        # ).squeeze(0) # [H, W, 3]

        if hasattr(self.mesh, 'cnt'):
            cnt = self.ctx.texture(self.mesh.cnt.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            cnt = self.ctx.antialias(cnt, rast, v_clip, self.mesh.f).squeeze(0)  # [H, W, 3]
            cnt = alpha * cnt + (1 - alpha) * 1  # 1 means no-inpaint in background
            results['cnt'] = cnt

        # if hasattr(self.mesh, 'cnt_inpaint'):
        #     cnt_inpaint = self.ctx.texture(self.mesh.cnt_inpaint.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
        #     cnt_inpaint = self.ctx.antialias(cnt_inpaint, rast, v_clip, self.mesh.f).squeeze(0)  # [H, W, 3]
        #     cnt_inpaint = alpha * cnt_inpaint + (1 - alpha) * 1  # 1 means no-inpaint in background
        #     results['cnt_inpaint'] = cnt_inpaint

        if hasattr(self.mesh, 'viewcos_cache'):
            viewcos_cache = self.ctx.texture(self.mesh.viewcos_cache.unsqueeze(0), texc, uv_da=texc_db,
                                             filter_mode='linear-mipmap-linear')
            viewcos_cache = self.ctx.antialias(viewcos_cache, rast, v_clip, self.mesh.f).squeeze(0)  # [H, W, 3]
            results['viewcos_cache'] = viewcos_cache

        # all shaped as [H, W, C]
        results['image'] = albedo
        results['alpha'] = alpha
        results['normal'] = normal_aa  # in [-1, 1]
        results['rot_normal'] = rot_normal_aa  # in [-1, 1]
        results['viewcos'] = viewcos
        results['uvs'] = texc.squeeze(0)
        results['comp_rgb_bg'] = self.bg

        return results