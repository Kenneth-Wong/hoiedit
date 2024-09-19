import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .masactrl_utils import AttentionBase
from .masactrl import MutualSelfAttentionControl

from torchvision.utils import save_image


class MutualRelControlMaskAuto(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.1,
                 ref_subj_token_idx=[0], ref_obj_token_idx=[2],
                 cur_subj_token_idx=[0], cur_obj_token_idx=[2],
                 mask_save_dir=None, model_type="SD", attn_way='sob'):
        """
        MasaCtrl with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        print("Using MutualRelAttentionControlMaskAuto")
        self.thres = thres
        self.ref_subj_token_idx = ref_subj_token_idx
        self.ref_obj_token_idx = ref_obj_token_idx
        self.cur_subj_token_idx = cur_subj_token_idx
        self.cur_obj_token_idx = cur_obj_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

        self.attn_way = attn_way


    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask
            thres = self.thres
            mask[mask >= thres] = 1
            mask[mask < thres] = 0
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg])

        attn = sim.softmax(-1)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def attn_so_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask    # 2, res*res
            thres = self.thres
            mask[mask >= thres] = 1
            mask[mask < thres] = 0
            sim_subj_fg = sim + mask[0].masked_fill(mask[0] == 0, torch.finfo(sim.dtype).min)
            # sim_subj_bg = sim + mask[0].masked_fill(mask[0] == 1, torch.finfo(sim.dtype).min)
            sim_obj_fg = sim + mask[1].masked_fill(mask[1] == 0, torch.finfo(sim.dtype).min)
            # sim_obj_bg = sim + mask[1].masked_fill(mask[1] == 1, torch.finfo(sim.dtype).min)
            sim_bg = sim + (mask[0] + mask[1]).masked_fill((mask[0] + mask[1]) > 0, torch.finfo(sim.dtype).min)
            # sim = torch.cat([sim_subj_fg, sim_obj_fg, sim_subj_bg, sim_obj_bg])  # 8x4, res*res, res*res
            sim = torch.cat([sim_subj_fg, sim_obj_fg, sim_bg])  # 8x4, res*res, res*res
        attn = sim.softmax(-1)

        if len(attn) == 3 * len(v):
            v = torch.cat([v] * 3)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)  # 8x4, res*res, 80
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)  # 4, res*res,640
        return out

    def general_attn_so_batch(self, q, k, v, kt, vt, sim, attn, is_cross, place_in_unet, num_heads,  **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        kt = rearrange(kt, "(b h) n d -> h (b n) d", h=num_heads)
        vt = rearrange(vt, "(b h) n d -> h (b n) d", h=num_heads)

        sim_st = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        sim_tt = torch.einsum("h i d, h j d -> h i j", q, kt) * kwargs.get("scale")

        # binarize the mask
        source_mask = self.self_attns_mask  # 2, res*res
        thres = self.thres
        source_mask[source_mask >= thres] = 1
        source_mask[source_mask < thres] = 0

        if self.attn_way == 'sob':
            mask1, mask2 = source_mask[0], source_mask[1]
            sim_subj_fg = sim_st + mask1.masked_fill(mask1 == 0, torch.finfo(sim.dtype).min)
            sim_obj_fg = sim_st + mask2.masked_fill(mask2 == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim_st + (mask1 + mask2).masked_fill((mask1 + mask2) > 0, torch.finfo(sim.dtype).min)
            v = torch.cat([v] * 3)

        elif self.attn_way == 'so':
            mask1, mask2 = source_mask[0], source_mask[1]
            sim_subj_fg = sim_st + mask1.masked_fill(mask1 == 0, torch.finfo(sim.dtype).min)
            sim_obj_fg = sim_st + mask2.masked_fill(mask2 == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim_tt
            v = torch.cat([v, v, vt])

        elif self.attn_way == 'sb':
            mask1, mask2 = source_mask[0], source_mask[1]
            sim_subj_fg = sim_st + mask1.masked_fill(mask1 == 0, torch.finfo(sim.dtype).min)
            sim_obj_fg = sim_tt
            sim_bg = sim_st + (mask1 + mask2).masked_fill((mask1 + mask2) > 0, torch.finfo(sim.dtype).min)
            v = torch.cat([v, vt, v])

        elif self.attn_way == 'ob':
            mask1, mask2 = source_mask[0], source_mask[1]
            sim_subj_fg = sim_tt
            sim_obj_fg = sim_st + mask2.masked_fill(mask2 == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim_st + (mask1 + mask2).masked_fill((mask1 + mask2) > 0, torch.finfo(sim.dtype).min)
            v = torch.cat([vt, v, v])

        elif self.attn_way == 's':
            mask1, mask2 = source_mask[0], source_mask[1]
            sim_subj_fg = sim_st + mask1.masked_fill(mask1 == 0, torch.finfo(sim.dtype).min)
            sim_obj_fg = sim_tt
            sim_bg = sim_tt
            v = torch.cat([v, vt, vt])

        elif self.attn_way == 'o':
            mask1, mask2 = source_mask[0], source_mask[1]
            sim_subj_fg = sim_tt
            sim_obj_fg = sim_st + mask2.masked_fill(mask2 == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim_tt
            v = torch.cat([vt, v, vt])

        elif self.attn_way == 'b':
            mask1, mask2 = source_mask[0], source_mask[1]
            sim_subj_fg = sim_tt
            sim_obj_fg = sim_tt
            sim_bg = sim_st + (mask1 + mask2).masked_fill((mask1 + mask2) > 0, torch.finfo(sim.dtype).min)
            v = torch.cat([vt, vt, v])

        elif self.attn_way == '':
            sim_subj_fg = sim_tt
            sim_obj_fg = sim_tt
            sim_bg = sim_tt
            v = torch.cat([vt, vt, vt])

        else:
            raise NotImplementedError

        sim = torch.cat([sim_subj_fg, sim_obj_fg, sim_bg])  # 8x3, res*res, res*res
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)  # 8x3, res*res, 80
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)  # 3, res*res,640

        return out

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def aggregate_so_cross_attn_map(self, subj_idx, obj_idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        subj_image, obj_image = attn_map[..., subj_idx], attn_map[..., obj_idx]
        if isinstance(subj_idx, list):
            subj_image = subj_image.sum(-1)
        if isinstance(obj_idx, list):
            obj_image = obj_image.sum(-1)
        image = torch.stack((subj_image, obj_image), -1)  # B, res, res, 2
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]   # B, 1, 1, 2
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]   # B, 1, 1, 2
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            # save cross attention map with res 16 * 16
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if len(self.cross_attns) == 0:
            self.self_attns_mask = None
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            mask = self.aggregate_so_cross_attn_map(self.ref_subj_token_idx, self.ref_obj_token_idx)  # B, 16, 16, 2
            mask_source = mask[-2]  # 16, 16, 2
            res = int(np.sqrt(q.shape[1]))
            self.self_attns_mask = F.interpolate(mask_source.permute(2, 0, 1).unsqueeze(0), (res, res)).flatten(2).squeeze(0)  # 2, res*res
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(self.self_attns_mask.shape[1]))
                mask_image = self.self_attns_mask.reshape(-1, H, W).unsqueeze(1)  # 2, 1, res, res  for B,C,H,W saving
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_s_{self.cur_step}_{self.cur_att_layer}.png"), pad_value=0.5)
                mask_image[mask_image >= self.thres] = 1
                mask_image[mask_image < self.thres] = 0
                save_image(mask_image, os.path.join(self.mask_save_dir, f"binarized_mask_s_{self.cur_step}_{self.cur_att_layer}.png"), pad_value=0.5)

        # if self.self_attns_mask is not None:
            mask = self.aggregate_so_cross_attn_map(self.cur_subj_token_idx, self.cur_obj_token_idx)  # B, 16, 16, 2
            mask_target = mask[-1]  # 16, 16, 2  target
            res = int(np.sqrt(q.shape[1]))
            spatial_mask = F.interpolate(mask_target.permute(2, 0, 1).unsqueeze(0), (res, res)).flatten(2).squeeze(0).unsqueeze(-1) #F.interpolate(mask_target.unsqueeze(0).unsqueeze(0), (res, res)).reshape(-1, 1)
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(spatial_mask.shape[1]))
                mask_image = spatial_mask.squeeze(-1).reshape(-1, H, W).unsqueeze(1)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_t_{self.cur_step}_{self.cur_att_layer}.png"), pad_value=0.5)
                mask_image[mask_image >= self.thres] = 1
                mask_image[mask_image < self.thres] = 0
                save_image(mask_image, os.path.join(self.mask_save_dir, f"binarized_mask_t_{self.cur_step}_{self.cur_att_layer}.png"), pad_value=0.5)

            # binarize the mask
            thres = self.thres
            spatial_mask[spatial_mask >= thres] = 1
            spatial_mask[spatial_mask < thres] = 0

            out_u_target = self.general_attn_so_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], ku[-num_heads:], vu[-num_heads:], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.general_attn_so_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], kc[-num_heads:], vc[-num_heads:], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)


            out_u_target_subj_fg, out_u_target_obj_fg, out_u_target_bg = out_u_target.chunk(3)
            out_c_target_subj_fg, out_c_target_obj_fg, out_c_target_bg= out_c_target.chunk(3)

            out_u_target = out_u_target_obj_fg * spatial_mask[1] + out_u_target_subj_fg * spatial_mask[0] + out_u_target_bg * (spatial_mask.sum(0) == 0)  #+ #
            out_c_target = out_c_target_obj_fg * spatial_mask[1] + out_c_target_subj_fg * spatial_mask[0] + out_c_target_bg * (spatial_mask.sum(0) == 0)  #+   #

            # set self self-attention mask to None
            self.self_attns_mask = None

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out
