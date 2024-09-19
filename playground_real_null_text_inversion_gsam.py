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
# model_path = "xyn-ai/anything-v4.0"
# model_path = '../huggingface/CompVis/stable-diffusion-v1-4'
# model_path = "../huggingface/runwayml/stable-diffusion-v1-5"
model_path = '../ReVersion-master/experiments/back2back_sd4'
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False)
model = MasaCtrlNullTextPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

from masactrl.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlMask, \
    MutualSelfAttentionControlMaskAuto
from masactrl.marectrl import MutualRelControlMaskAuto
from torchvision.io import read_image
from masactrl.marectrl_gsam import MutualRelGSAMControlMaskAuto
from grounded_sam import GSAM


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


seed = 42
seed_everything(seed)

source_prompt = "man shake hands with man" # #  #"monkey back to back monkey" #
target_prompt = "man <R> man"  ## #  # "A photo of a person, black t-shirt, raising hand" #"a photo of jumping monkey" #"a photo of a running corgi" #"child <R> father"  #
prompts = [source_prompt, target_prompt]
subj_id, subj_name = 1, "man"
obj_id, obj_name = 3, "man"

out_dir = "./workdir/debug/"  # "./workdir/masactrl_real_exp/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}_seed_{seed}_{prompts[0]}")
os.makedirs(out_dir, exist_ok=True)

# source image
SOURCE_IMAGE_PATH = "./gradio_app/images/shake_hands2.png"  #"./gradio_app/images/feeding.png"  # "/T2021047/code/ReVersion-master/experiments/back2back_sd4/inference/monkey <R> monkey/samples/0009.png"  # #"./gradio_app/images/shake_hands.jpg"  # # # "gradio_app/images/person.png"# #"./gradio_app/images/corgi.jpg" #'/T2021047/code/ReVersion/reversion_benchmark_v1/ride_on/9.jpg'  #"./gradio_app/images/holding.jpg" #'/T2021047/code/ReVersion/reversion_benchmark_v1/hug/6.jpg' # ##
source_image = load_image(SOURCE_IMAGE_PATH, device)



# invert the source image with null text inverion
start_code, uncond_embeddings = model.invert_with_null_text_optimize(source_image,
                                                                     source_prompt,
                                                                     guidance_scale=7.5,
                                                                     num_inference_steps=50,
                                                                     num_inner_steps=10,
                                                                     )
start_code = start_code.expand(len(prompts), -1, -1, -1)

# inference the synthesized image with MasaCtrl
STEP = 4
LAYER = 10

# results of direct synthesis
editor = AttentionBase()
regiter_attention_editor_diffusers(model, editor)
image_fixed = model([target_prompt],
                    latents=start_code[-1:],
                    unconditioning=uncond_embeddings,
                    num_inference_steps=50,
                    guidance_scale=7.5)

# hijack the attention module
# editor = MutualSelfAttentionControl(STEP, LAYER)
masa_save_dir = os.path.join(out_dir, "masa_masks")
mare_save_dir = os.path.join(out_dir, "mare_masks")
os.makedirs(masa_save_dir, exist_ok=True)
os.makedirs(mare_save_dir, exist_ok=True)
editor = MutualSelfAttentionControlMaskAuto(STEP, LAYER, ref_token_idx=[1, 5], cur_token_idx=[1, 5],
                                            mask_save_dir=masa_save_dir)
editor2 = MutualRelControlMaskAuto(STEP, LAYER, ref_subj_token_idx=[1], ref_obj_token_idx=[5],
                                   cur_subj_token_idx=[1], cur_obj_token_idx=[3], mask_save_dir=mare_save_dir)

regiter_attention_editor_diffusers(model, editor)

# inference the synthesized image
image_masactrl = model(prompts,
                       latents=start_code,
                       unconditioning=uncond_embeddings,
                       guidance_scale=7.5)

regiter_attention_editor_diffusers(model, editor2)
image_marectrl = model(prompts,
                       latents=start_code,
                       unconditioning=uncond_embeddings,
                       guidance_scale=7.5)

# save the synthesized image
out_image = torch.cat([source_image * 0.5 + 0.5,
                       image_masactrl[0:1],
                       image_fixed,
                       image_masactrl[-1:],
                       #image_marectrl[0:1],
                       image_marectrl[-1:]], dim=0)
save_image(out_image, os.path.join(out_dir, f"all_step{STEP}_layer{LAYER}.png"))
save_image(out_image[0], os.path.join(out_dir, f"source_step{STEP}_layer{LAYER}.png"))
save_image(out_image[1], os.path.join(out_dir, f"masa_reconstructed_source_step{STEP}_layer{LAYER}.png"))
save_image(out_image[2], os.path.join(out_dir, f"without_step{STEP}_layer{LAYER}.png"))
save_image(out_image[3], os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYER}.png"))
#save_image(out_image[3], os.path.join(out_dir, f"mare_reconstructed_source_step{STEP}_layer{LAYER}.png"))
save_image(out_image[4], os.path.join(out_dir, f"marectrl_step{STEP}_layer{LAYER}.png"))


##################### GSAM refinement
gsam_model = GSAM()
source_detections = gsam_model.run_single_image(os.path.join(out_dir, f"source_step{STEP}_layer{LAYER}.png"),
                                                classes=[subj_name, obj_name],
                                                dsize=(512, 512),
                                                de_duplicated=True,
                                                save_name="source",
                                                save_box_path=out_dir,
                                                save_mask_path=out_dir)

mare_detections = gsam_model.run_single_image(os.path.join(out_dir, f"marectrl_step{STEP}_layer{LAYER}.png"),
                                                classes=[subj_name, obj_name],
                                                dsize=(512, 512),
                                                de_duplicated=True,
                                                save_name="marectrl",
                                                save_box_path=out_dir,
                                                save_mask_path=out_dir)

source_masks = source_detections.mask
mare_masks = mare_detections.mask

editor3 = MutualRelGSAMControlMaskAuto(source_masks, mare_masks, STEP, LAYER, mask_save_dir=out_dir, thres=0.1)
regiter_attention_editor_diffusers(model, editor3)
image_marectrl_gsam = model(prompts, latents=start_code, unconditioning=uncond_embeddings, guidance_scale=7.5)[-1:]

save_image(image_marectrl_gsam, os.path.join(out_dir, f"marectrl_gsam_step{STEP}_layer{LAYER}.png"))

print("Syntheiszed images are saved in", out_dir)