# ==================================================================================
# OverLoCK Model Implementation
#
# This code is sourced from the official OverLoCK repository:
# https://github.com/LMMMEng/OverLoCK
#
# The code is licensed under the Apache License 2.0.
# A copy of the license is included in the original repository (LICENSE.md).
#
# In accordance with the license, this file retains the original structure
# and functionality for integration into the PRISM project.
#
# MODIFIED: Removed hard dependency on 'natten'.
# MODIFIED: Fixed fallback logic for 'depthwise_conv2d_implicit_gemm'.
# MODIFIED: Fixed device mismatch error in '_apply_rpb' by moving index tensors to the correct device.
# ==================================================================================

'''
This is an official implementation of OverLoCK model proposed in the paper:
https://arxiv.org/abs/2502.20087
'''
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from timm.models.layers import DropPath, to_2tuple
import warnings

# Try to import natten, but make it optional.
try:
    from natten.functional import na2d_av

    HAS_NATTEN = True
    print("--- INFO: 'natten' library found. Using NATTEN for optimized neighborhood attention. ---")
except ImportError:
    HAS_NATTEN = False
    warnings.warn(
        "--- WARNING: 'natten' library not found or failed to import. "
        "Falling back to a slower PyTorch implementation for neighborhood attention. "
        "Performance will be impacted. ---"
    )

from torch.utils.checkpoint import checkpoint

# Try to import mmdet/mmseg specific modules, but make them optional
# so the model can be used in a standalone script.
try:
    from mmdet.models.builder import MODELS
    from mmdet.utils import get_root_logger
    from mmcv.runner import load_checkpoint

    IS_MM_ENV = True
except ImportError:
    IS_MM_ENV = False


    # Define dummy decorators if not in a mm-style environment
    class MODELS:
        @staticmethod
        def register_module():
            def decorator(cls):
                return cls

            return decorator


    def get_root_logger():
        import logging
        return logging.getLogger()


    def load_checkpoint(model, filename, map_location='cpu', logger=None):
        if logger is None:
            logger = get_root_logger()
        logger.info(f"Loading checkpoint from {filename}")
        state_dict = torch.load(filename, map_location=map_location)
        # Handle potential 'state_dict' key in checkpoint
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)


def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2,
                                                                                              kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        except ImportError:
            # Set to None on failure, so the next 'if' block is skipped
            DepthWiseConv2dImplicitGEMM = None
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (
                conv_bias - bn.running_mean) * bn.weight / std


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim)
    )


def downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
    )


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()


class GRN(nn.Module):
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class DilatedReparamBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size >= 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size >= 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size >= 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size >= 5:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(f'dil_conv_k{k}_{r}',
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__(f'dil_bn_k{k}_{r}', get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{k}_{r}')
            bn = self.__getattr__(f'dil_bn_k{k}_{r}')
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__(f'dil_conv_k{k}_{r}')
                bn = self.__getattr__(f'dil_bn_k{k}_{r}')
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__(f'dil_conv_k{k}_{r}')
                self.__delattr__(f'dil_bn_k{k}_{r}')


class CTXDownsample(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim)
        )
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim // 4, h_dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim // 4)
        )

    def forward(self, x, ctx):
        x = self.x_proj(x)
        ctx = self.h_proj(ctx)
        return (x, ctx)


class ResDWConv(nn.Conv2d):
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super().forward(x)


class RepConvBlock(nn.Module):
    def __init__(self, dim=64, kernel_size=7, mlp_ratio=4, ls_init_value=None, res_scale=False,
                 drop_path=0, norm_layer=LayerNorm2d, use_gemm=False, deploy=False, use_checkpoint=False):
        super().__init__()
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        mlp_dim = int(dim * mlp_ratio)
        self.dwconv = ResDWConv(dim, kernel_size=3)
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False,
                                attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )
        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward_features(self, x):
        x = self.dwconv(x)
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))
        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self.forward_features, x, use_reentrant=False)
        return self.forward_features(x)


class DynamicConvBlock(nn.Module):
    def __init__(self, dim=64, ctx_dim=32, kernel_size=7, smk_size=5, num_heads=2, mlp_ratio=4, ls_init_value=None,
                 res_scale=False, drop_path=0, norm_layer=LayerNorm2d, is_first=False, is_last=False, use_gemm=False,
                 deploy=False, use_checkpoint=False, **kwargs):
        super().__init__()
        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim * mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        self.scale = (dim // self.num_heads) ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint
        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)
        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1), GRN(dim),
        )
        self.weight_query = nn.Sequential(nn.Conv2d(dim, dim // 2, 1, bias=False), nn.BatchNorm2d(dim // 2))
        self.weight_key = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.Conv2d(ctx_dim, dim // 2, 1, bias=False),
                                        nn.BatchNorm2d(dim // 2))
        self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, 1)
        self.dyconv_proj = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim))
        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size, deploy, False, use_gemm),
            nn.BatchNorm2d(dim)
        )
        self.se_layer = SEModule(dim)
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU())
        self.proj = nn.Sequential(nn.BatchNorm2d(dim), nn.Conv2d(dim, out_dim, 1))
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, 1), nn.GELU(),
            ResDWConv(mlp_dim, 3), GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, 1)
        )
        self.ls1 = LayerScale(out_dim, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self._get_rpb()

    def _get_rpb(self):
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, 2 * self.smk_size - 1, 2 * self.smk_size - 1))
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, 2 * self.kernel_size - 1, 2 * self.kernel_size - 1))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def _generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def _apply_rpb(self, attn, rpb, h, w, k, idx_h, idx_w, idx_k):
        # --- MODIFICATION START ---
        # Move index tensors to the correct device
        device = attn.device
        idx_h = idx_h.to(device)
        idx_w = idx_w.to(device)
        idx_k = idx_k.to(device)
        # --- MODIFICATION END ---

        num_repeat_h = torch.ones(k, dtype=torch.long, device=attn.device)
        num_repeat_w = torch.ones(k, dtype=torch.long, device=attn.device)
        num_repeat_h[k // 2] = h - (k - 1)
        num_repeat_w[k // 2] = w - (k - 1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2 * k - 1)) + idx_w.repeat_interleave(
            num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = torch.flip(bias_idx.reshape(-1, k ** 2), [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx].reshape(1, self.num_heads, h, w, k ** 2)
        return attn + rpb

    def _forward_inner(self, x, h_x, h_r):
        input_res = x.shape[2:]
        B, C, H, W = x.shape
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)
        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x)
        lepe = self.lepe(x)
        is_pad = False
        if min(H, W) < self.kernel_size:
            is_pad = True
            size = (self.kernel_size, int(self.kernel_size / H * W)) if H < W else (int(self.kernel_size / W * H),
                                                                                    self.kernel_size)
            x, x_f = (F.interpolate(t, size=size, mode='bilinear', align_corners=False) for t in (x, x_f))
            H, W = size
        query, key = torch.split(x_f, [C, x_f.shape[1] - C], dim=1)
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        q = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        k = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = rearrange(einsum(q, k, 'b g c n, b g c l -> b g n l'), 'b g n l -> b l g n')
        weight = self.weight_proj(weight).reshape(B, self.num_heads, H, W, -1)
        a1, a2 = torch.split(weight, [self.smk_size ** 2, self.kernel_size ** 2], -1)
        a1 = self._apply_rpb(a1, self.rpb1, H, W, self.smk_size, *self._generate_idx(self.smk_size))
        a2 = self._apply_rpb(a2, self.rpb2, H, W, self.kernel_size, *self._generate_idx(self.kernel_size))
        a1, a2 = torch.softmax(a1, -1), torch.softmax(a2, -1)
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)

        if HAS_NATTEN:
            x1 = na2d_av(a1, value[0], self.smk_size)
            x2 = na2d_av(a2, value[1], self.kernel_size)
        else:
            # Fallback implementation without natten
            pad1 = self.smk_size // 2
            pad2 = self.kernel_size // 2

            v1 = rearrange(value[0], 'b g h w c -> b (g c) h w')
            v2 = rearrange(value[1], 'b g h w c -> b (g c) h w')

            v1 = F.unfold(v1, kernel_size=self.smk_size, padding=pad1)
            v2 = F.unfold(v2, kernel_size=self.kernel_size, padding=pad2)

            v1 = v1.reshape(B, self.num_heads, -1, self.smk_size ** 2, H, W)
            v2 = v2.reshape(B, self.num_heads, -1, self.kernel_size ** 2, H, W)

            v1 = rearrange(v1, 'b g c k h w -> b g c h w k')
            v2 = rearrange(v2, 'b g c k h w -> b g c h w k')

            a1 = a1.unsqueeze(2)
            a2 = a2.unsqueeze(2)

            x1 = einsum(a1, v1, 'b g c h w k, b g c h w k -> b g h w c')
            x2 = einsum(a2, v2, 'b g c h w k, b g c h w k -> b g h w c')

        x = rearrange(torch.cat([x1, x2], 1), 'b g h w c -> b (g c) h w', h=H, w=W)
        if is_pad:
            x = F.adaptive_avg_pool2d(x, input_res)
        x = self.dyconv_proj(x) + lepe
        x = self.se_layer(x) * gate
        x = self.proj(x)
        x = identity + self.drop_path(self.ls1(x)) if not self.res_scale else self.ls1(identity) + self.drop_path(x)
        x = self.dwconv2(x)
        x_mlp = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.ls2(x_mlp)) if not self.res_scale else self.ls2(x) + self.drop_path(x_mlp)
        if self.is_last:
            return x, None
        return torch.split(x, [C, x.shape[1] - C], dim=1)

    def forward(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        return self._forward_inner(x, h_x, h_r)


class OverLoCK(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2], sub_depth=[4, 2], in_chans=3, embed_dim=[96, 192, 384, 768],
                 kernel_size=[7, 7, 7, 7], mlp_ratio=[4, 4, 4, 4], sub_mlp_ratio=[4, 4], sub_num_heads=[4, 8],
                 ls_init_value=[None, None, 1, 1], res_scale=True, smk_size=5, deploy=False, use_gemm=True,
                 use_ds=True, drop_rate=0, drop_path_rate=0, norm_layer=LayerNorm2d, projection=1024,
                 num_classes=1000, use_checkpoint=[0, 0, 0, 0]):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = downsample(embed_dim[2], embed_dim[3])
        self.high_level_proj = nn.Conv2d(embed_dim[-1], embed_dim[-1] // 4, 1)
        self.patch_embedx = CTXDownsample(embed_dim[2], embed_dim[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]
        self.blocks1 = nn.ModuleList([RepConvBlock(embed_dim[0], kernel_size[0], mlp_ratio[0], ls_init_value[0],
                                                   res_scale, dpr[i], norm_layer, use_gemm, deploy,
                                                   i < use_checkpoint[0]) for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([RepConvBlock(embed_dim[1], kernel_size[1], mlp_ratio[1], ls_init_value[1],
                                                   res_scale, dpr[i + depth[0]], norm_layer, use_gemm, deploy,
                                                   i < use_checkpoint[1]) for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([RepConvBlock(embed_dim[2], kernel_size[2], mlp_ratio[2], ls_init_value[2],
                                                   res_scale, dpr[i + sum(depth[:2])], norm_layer, use_gemm, deploy,
                                                   i < use_checkpoint[2]) for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([RepConvBlock(embed_dim[3], kernel_size[3], mlp_ratio[3], ls_init_value[3],
                                                   res_scale, dpr[i + sum(depth[:3])], norm_layer, use_gemm, deploy,
                                                   i < use_checkpoint[3]) for i in range(depth[3])])
        self.sub_blocks3 = nn.ModuleList([DynamicConvBlock(embed_dim[2], embed_dim[-1], kernel_size[2], smk_size,
                                                           sub_num_heads[0], sub_mlp_ratio[0], ls_init_value[2],
                                                           res_scale, dpr[i + sum(depth)], norm_layer, i == 0, False,
                                                           use_gemm, deploy, i < use_checkpoint[2]) for i in
                                          range(sub_depth[0])])
        self.sub_blocks4 = nn.ModuleList([DynamicConvBlock(embed_dim[3], embed_dim[-1], kernel_size[-1], smk_size,
                                                           sub_num_heads[1], sub_mlp_ratio[1], ls_init_value[3],
                                                           res_scale, dpr[i + sum(depth) + sub_depth[0]], norm_layer,
                                                           False, i == sub_depth[1] - 1, use_gemm, deploy,
                                                           i < use_checkpoint[3]) for i in range(sub_depth[1])])
        self.h_proj = nn.Sequential(nn.Conv2d(embed_dim[-1], embed_dim[-1] + embed_dim[-1] // 4, 1),
                                    LayerScale(embed_dim[-1] + embed_dim[-1] // 4, 1e-5))
        self.extra_norm = nn.ModuleList(
            [norm_layer(d) if i < 2 else norm_layer(d + embed_dim[-1] // 4) for i, d in enumerate(embed_dim)])
        self.extra_norm.append(norm_layer(embed_dim[-1]))
        # These layers are for classification, not needed for feature extraction as a backbone
        del self.extra_norm[3]  # remove last norm as h_proj has norm-like layer
        self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _convert_sync_batchnorm(self):
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward_features(self, x):
        outs = []
        x = self.patch_embed1(x)
        for blk in self.blocks1: x = blk(x)
        outs.append(self.extra_norm[0](x))
        x = self.patch_embed2(x)
        for blk in self.blocks2: x = blk(x)
        outs.append(self.extra_norm[1](x))
        x = self.patch_embed3(x)
        for blk in self.blocks3: x = blk(x)
        ctx = self.patch_embed4(x)
        for blk in self.blocks4: ctx = blk(ctx)
        ctx_cls = self.extra_norm[-1](ctx)
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=x.shape[2:], mode='bilinear', align_corners=False)
        h_r = None  # Initialize h_r to avoid UnboundLocalError
        for i, blk in enumerate(self.sub_blocks3):
            x, ctx_up = blk(x, ctx_up, ctx_up if i == 0 else h_r)
            h_r = ctx_up
        outs.append(self.extra_norm[2](torch.cat([x, ctx_up], dim=1)))
        x, ctx_up = self.patch_embedx(x, ctx_up)
        for blk in self.sub_blocks4:
            x, _ = blk(x, ctx_up, ctx_ori)
        x = x + self.h_proj(ctx_cls)
        outs.append(x)
        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


# Factory functions
@MODELS.register_module()
def overlock_xt(pretrained=False, **kwargs):
    model = OverLoCK(depth=[2, 2, 3, 2], sub_depth=[6, 2], embed_dim=[56, 112, 256, 336], kernel_size=[17, 15, 13, 7],
                     sub_num_heads=[4, 6], sub_mlp_ratio=[3, 3], **kwargs)
    if pretrained:
        url = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_xt_in1k_224.pth'
        load_checkpoint(model, url, map_location='cpu', logger=get_root_logger())
    model._convert_sync_batchnorm()
    return model


@MODELS.register_module()
def overlock_t(pretrained=False, **kwargs):
    model = OverLoCK(depth=[4, 4, 6, 2], sub_depth=[12, 2], embed_dim=[64, 128, 256, 512], kernel_size=[17, 15, 13, 7],
                     sub_num_heads=[4, 8], sub_mlp_ratio=[3, 3], **kwargs)
    if pretrained:
        url = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_t_in1k_224.pth'
        load_checkpoint(model, url, map_location='cpu', logger=get_root_logger())
    model._convert_sync_batchnorm()
    return model


@MODELS.register_module()
def overlock_s(pretrained=False, **kwargs):
    model = OverLoCK(depth=[6, 6, 8, 3], sub_depth=[16, 3], embed_dim=[64, 128, 320, 512], kernel_size=[17, 15, 13, 7],
                     sub_num_heads=[8, 16], sub_mlp_ratio=[3, 3], **kwargs)
    if pretrained:
        url = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_s_in1k_224.pth'
        load_checkpoint(model, url, map_location='cpu', logger=get_root_logger())
    model._convert_sync_batchnorm()
    return model


@MODELS.register_module()
def overlock_b(pretrained=False, **kwargs):
    model = OverLoCK(depth=[8, 8, 10, 4], sub_depth=[20, 4], embed_dim=[80, 160, 384, 576], kernel_size=[17, 15, 13, 7],
                     sub_num_heads=[6, 9], sub_mlp_ratio=[3, 3], **kwargs)
    if pretrained:
        url = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_b_in1k_224.pth'
        load_checkpoint(model, url, map_location='cpu', logger=get_root_logger())
    model._convert_sync_batchnorm()
    return model