import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

torch.cuda.set_device(0)  # set the GPU device


# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = '../huggingface/CompVis/stable-diffusion-v1-4'
#model_path = "xyn-ai/anything-v4.0"
# model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)


from masactrl.masactrl import MutualSelfAttentionControl


seed = 42
seed_everything(seed)

out_dir = "./workdir/masactrl_exp/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

prompts = [
    "1boy, casual, outdoors, sitting",  # source prompt
    "1boy, casual, outdoors, standing"  # target prompt
]

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

# hijack the attention module
editor = MutualSelfAttentionControl(STEP, LAYPER)
regiter_attention_editor_diffusers(model, editor)

# inference the synthesized image
image_masactrl = model(prompts, latents=start_code, guidance_scale=7.5)[-1:]

# save the synthesized image
out_image = torch.cat([image_ori, image_masactrl], dim=0)
save_image(out_image, os.path.join(out_dir, f"all_step{STEP}_layer{LAYPER}.png"))
save_image(out_image[0], os.path.join(out_dir, f"source_step{STEP}_layer{LAYPER}.png"))
save_image(out_image[1], os.path.join(out_dir, f"without_step{STEP}_layer{LAYPER}.png"))
save_image(out_image[2], os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYPER}.png"))

print("Syntheiszed images are saved in", out_dir)