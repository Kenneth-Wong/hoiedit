import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler

from masactrl.diffuser_utils_null_text import MasaCtrlNullTextPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

torch.cuda.set_device(0)  # set the GPU device


# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#model_path = '../huggingface/CompVis/stable-diffusion-v1-4'
#model_path = "xyn-ai/anything-v4.0"
# model_path = "runwayml/stable-diffusion-v1-5"
model_path = '../ReVersion-master/experiments/ride_on_sd4'
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = MasaCtrlNullTextPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)


from masactrl.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlMask, \
    MutualSelfAttentionControlMaskAuto
from masactrl.marectrl import MutualRelControlMaskAuto
from masactrl.marectrl_gsam import MutualRelGSAMControlMaskAuto
from grounded_sam import GSAM

seed = 43 #44
seed_everything(seed)

prompts = ["girl feeding horse, in the snow", "girl <R> horse, in the snow"]  #["a real photo of boy feeding horse", "a real photo of boy <R> horse"] #["a real photo of boy feeding horse, in the snow","a real photo of boy <R> horse, in the snow"]
subj_id, subj_name = 5, "girl"
obj_id, obj_name = 7, "zebra"

out_dir = "./workdir/masactrl_debug2/" #"./workdir/masactrl_exp/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}_seed_{seed}_{prompts[0]}")
os.makedirs(out_dir, exist_ok=True)

# prompts = [
#     "1boy, casual, outdoors, sitting",  # source prompt
#     "1boy, casual, outdoors, standing"  # target prompt
# ]

# initialize the noise map
start_code = torch.randn([1, 4, 64, 64], device=device)
start_code = start_code.expand(len(prompts), -1, -1, -1)

# inference the synthesized image without MasaCtrl
editor = AttentionBase()
regiter_attention_editor_diffusers(model, editor)
image_ori = model(prompts, latents=start_code, guidance_scale=7.5)

# inference the synthesized image with MasaCtrl
STEP = 4
LAYPER = 10

masa_save_dir = os.path.join(out_dir, "masa_masks")
mare_save_dir = os.path.join(out_dir, "mare_masks")
os.makedirs(masa_save_dir, exist_ok=True)
os.makedirs(mare_save_dir, exist_ok=True)

# hijack the attention module
editor = MutualSelfAttentionControlMaskAuto(STEP, LAYPER, ref_token_idx=[1, 3], cur_token_idx=[1, 3], mask_save_dir=masa_save_dir)
regiter_attention_editor_diffusers(model, editor)

# inference the synthesized image
image_masactrl = model(prompts, latents=start_code, guidance_scale=7.5)[-1:]

editor2 = MutualRelControlMaskAuto(STEP, LAYPER, ref_subj_token_idx=[1], ref_obj_token_idx=[3],
                 cur_subj_token_idx=[1], cur_obj_token_idx=[3], mask_save_dir=mare_save_dir, thres=0.1)
regiter_attention_editor_diffusers(model, editor2)
image_marectrl = model(prompts, latents=start_code, guidance_scale=7.5)[-1:]

# save the synthesized image
out_image = torch.cat([image_ori, image_masactrl, image_marectrl], dim=0)
save_image(out_image, os.path.join(out_dir, f"all_step{STEP}_layer{LAYPER}.png"))
save_image(out_image[0], os.path.join(out_dir, f"source_step{STEP}_layer{LAYPER}.png"))
save_image(out_image[1], os.path.join(out_dir, f"without_step{STEP}_layer{LAYPER}.png"))
save_image(out_image[2], os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYPER}.png"))
save_image(out_image[3], os.path.join(out_dir, f"marectrl_step{STEP}_layer{LAYPER}.png"))


##################### GSAM refinement
gsam_model = GSAM()
source_detections = gsam_model.run_single_image(os.path.join(out_dir, f"source_step{STEP}_layer{LAYPER}.png"),
                                                classes=[subj_name, obj_name],
                                                dsize=(512, 512),
                                                de_duplicated=True,
                                                save_name="source",
                                                save_box_path=out_dir,
                                                save_mask_path=out_dir)

mare_detections = gsam_model.run_single_image(os.path.join(out_dir, f"marectrl_step{STEP}_layer{LAYPER}.png"),
                                                classes=[subj_name, obj_name],
                                                dsize=(512, 512),
                                                de_duplicated=True,
                                                save_name="marectrl",
                                                save_box_path=out_dir,
                                                save_mask_path=out_dir)

source_masks = source_detections.mask
mare_masks = mare_detections.mask

editor3 = MutualRelGSAMControlMaskAuto(source_masks, mare_masks, STEP, LAYPER, mask_save_dir=out_dir, thres=0.1)
regiter_attention_editor_diffusers(model, editor3)
image_marectrl_gsam = model(prompts, latents=start_code, guidance_scale=7.5)[-1:]

save_image(image_marectrl_gsam, os.path.join(out_dir, f"marectrl_gsam_step{STEP}_layer{LAYPER}.png"))



print("Syntheiszed images are saved in", out_dir)
