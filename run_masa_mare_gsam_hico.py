import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
import pickle
from PIL import Image
import os.path as osp
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

from masactrl.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlMask, \
    MutualSelfAttentionControlMaskAuto
from masactrl.marectrl import MutualRelControlMaskAuto
from masactrl.marectrl_gsam import MutualRelGSAMControlMaskAuto
from grounded_sam import GSAM
from torch.utils.data import Dataset
from tqdm import tqdm


# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#model_path = '../huggingface/CompVis/stable-diffusion-v1-4'
#model_path = "xyn-ai/anything-v4.0"
model_path = "../huggingface/runwayml/stable-diffusion-v1-5"
#model_path = '../ReVersion-master/experiments/ride_on_sd4'
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = MasaCtrlNullTextPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)


edit_dict = {'bicycle': ['hold', 'ride', 'inspect', 'jump', 'repair', 'straddle', 'wash'], 
'motorcycle': ['hold', 'ride', 'sit_on', 'inspect', 'jump', 'walk', 'wash'],
             'horse': ['feed', 'hold', 'kiss',  'ride', 'walk', 'wash'],
             'dog': ['carry', 'feed', 'hold', 'hug', 'kiss', 'run', 'straddle', 'walk', 'wash'],
             'sheep': ['carry', 'feed', 'kiss', 'hold', 'shear'],
             'cat': ['feed', 'hold', 'kiss', 'scratch'],
             }




def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image



seed = 42 #44
seed_everything(seed)


run_image_dir = 'data/hico_keep'
run_target_obj = 'sheep'
run_img_files = os.listdir(osp.join(run_image_dir, run_target_obj))
run_img_files = [f for f in run_img_files if f.endswith('.png') or f.endswith('.jpg')]


out_dir = f"./workdir/visualization_for_cvpr24/run_dataset/{run_target_obj}"
os.makedirs(out_dir, exist_ok=True)

# inference the synthesized image with MasaCtrl
STEP = 4
LAYPER = 10

for img_file in tqdm(run_img_files):
    img_path = osp.join(run_image_dir, run_target_obj, img_file)
    image_name = img_file[:-4]
    #print(image_name)
    source_image = load_image(img_path, torch.device('cuda'))
    source_rel, source_obj = image_name.split('*')[:2]
    #print(source_rel, source_obj)
    source_prompt = ' '.join(['human', source_rel, source_obj])
    source_subj_idx = [1]
    source_obj_idx = [2 + len(source_rel.split('_'))]
    
    # editing according the target_prompt
    # inversion
    start_code, uncond_embeddings = model.invert_with_null_text_optimize(source_image,
                                                                         source_prompt,
                                                                         guidance_scale=7.5,
                                                                         num_inference_steps=50,
                                                                         num_inner_steps=10,
                                                                         )
    start_code = start_code.expand(2, -1, -1, -1)
    rand_start_code = torch.randn([1, 4, 64, 64], device=device)
    rand_start_code = rand_start_code.expand(2, -1, -1, -1)

    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)
    gen_image_ori = model([source_prompt], latents=rand_start_code[:1], guidance_scale=7.5)  # use the source prompt, synthesize one
    save_image(gen_image_ori, os.path.join(out_dir, f"{image_name}_GEN_source_step{STEP}_layer{LAYPER}_seed{seed}_{source_prompt}.png"))
    
    # get target prompt
    cand_rels = edit_dict[run_target_obj]
    cand_rels = list(set(cand_rels).difference(set([source_rel])))
    for tgt_rel in cand_rels:
        tgt_prompt = ' '.join(['human', tgt_rel, source_obj])
        tgt_subj_idx = [1]
        tgt_obj_idx = [2 + len(tgt_rel.split('_'))]
        
        # hijack the attention module
        editor = MutualSelfAttentionControlMaskAuto(STEP, LAYPER, ref_token_idx=source_subj_idx+source_obj_idx, cur_token_idx=tgt_subj_idx+tgt_obj_idx)
        regiter_attention_editor_diffusers(model, editor)
        # inference the synthesized image
        image_masactrl = model([source_prompt, tgt_prompt], latents=start_code, unconditioning=uncond_embeddings, guidance_scale=7.5)[-1:]
        
        gen_image_masactrl = model([source_prompt, tgt_prompt], latents=rand_start_code, guidance_scale=7.5)[-1:]

        editor2 = MutualRelControlMaskAuto(STEP, LAYPER, ref_subj_token_idx=source_subj_idx, ref_obj_token_idx=source_obj_idx,
                         cur_subj_token_idx=tgt_subj_idx, cur_obj_token_idx=tgt_obj_idx, thres=0.1)
        regiter_attention_editor_diffusers(model, editor2)
        image_marectrl = model([source_prompt, tgt_prompt], latents=start_code, unconditioning=uncond_embeddings, guidance_scale=7.5)[-1:]
        
        gen_image_marectrl = model([source_prompt, tgt_prompt], latents=rand_start_code, guidance_scale=7.5)[-1:]
        
        
        out_image = torch.cat([source_image * 0.5 + 0.5, image_masactrl, image_marectrl], dim=0)
        save_image(out_image, os.path.join(out_dir, f"{image_name}_all_step{STEP}_layer{LAYPER}_seed{seed}_{source_prompt}+{tgt_prompt}.png"))
        save_image(out_image[1], os.path.join(out_dir, f"{image_name}_masactrl_step{STEP}_layer{LAYPER}_seed{seed}_{source_prompt}+{tgt_prompt}.png"))
        save_image(out_image[2], os.path.join(out_dir, f"{image_name}_marectrl_step{STEP}_layer{LAYPER}_seed{seed}_{source_prompt}+{tgt_prompt}.png"))
        
        gen_out_image = torch.cat([gen_image_ori, gen_image_masactrl, gen_image_marectrl], dim=0)
        save_image(gen_out_image, os.path.join(out_dir, f"{image_name}_GEN_all_step{STEP}_layer{LAYPER}_seed{seed}_{source_prompt}+{tgt_prompt}.png"))
        save_image(gen_out_image[1], os.path.join(out_dir, f"{image_name}_GEN_masactrl_step{STEP}_layer{LAYPER}_seed{seed}_{source_prompt}+{tgt_prompt}.png"))
        save_image(gen_out_image[2], os.path.join(out_dir, f"{image_name}_GEN_marectrl_step{STEP}_layer{LAYPER}_seed{seed}_{source_prompt}+{tgt_prompt}.png"))
    