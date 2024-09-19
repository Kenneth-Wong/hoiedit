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

from masactrl.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlMask, \
    MutualSelfAttentionControlMaskAuto
from masactrl.marectrl import MutualRelControlMaskAuto
from torchvision.io import read_image
from masactrl.marectrl_gsam import MutualRelGSAMControlMaskAuto
from grounded_sam import GSAM

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything
import json
import os.path as osp
from tqdm import tqdm
from data.reversion_benchmark_scenarios import inference_templates
import argparse

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def find_subjobj(text):
    tokens = text.split(' ')
    mid_idx = tokens.index('{}')
    subj_idx = list(range(0, mid_idx))
    obj_idx = list(range(mid_idx+1, len(tokens)))
    subj = ' '.join([tokens[i] for i in subj_idx])
    obj = ' '.join([tokens[i] for i in obj_idx])
    prompt = subj + ', ' + obj
    return prompt, subj, obj, subj_idx, obj_idx



parser = argparse.ArgumentParser()
parser.add_argument('--target_rel', type=str, default="walk")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_sub_dir', type=str, default='hico_experiments') #'experiments')
parser.add_argument('--data_dir', type=str, default='data/hico_actions') #'../ReVersion-master/reversion_benchmark_v1/')
parser.add_argument('--output_dir', type=str, default='./workdir/visualization_for_cvpr24/hico_actions')#'./workdir/visualization_for_cvpr24/reversion_benchmark_v1')
args = parser.parse_args()


torch.cuda.set_device(0)  # set the GPU device
# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_path = f'../ReVersion-master/{args.model_sub_dir}/{args.target_rel}_sd4'
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False)
model = MasaCtrlNullTextPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

seed_everything(args.seed)


out_dir = osp.join(args.output_dir, args.target_rel)
os.makedirs(out_dir, exist_ok=True)

# inference the synthesized image with MasaCtrl
STEP = 4
LAYER = 10

records = {}

if args.data_dir == '../ReVersion-master/reversion_benchmark_v1/':
    img2prompt = json.load(open(osp.join(args.data_dir, 'all_prompts.json')))
    prompt2img = json.load(open(osp.join(args.data_dir, 'prompt2img.json')))

    # 1. query from template: what object combiantions are suitable for target relationships:
    cand_pairs = inference_templates[args.target_rel]
    num = 0
    for cand_pair in tqdm(cand_pairs):
        if cand_pair in prompt2img:
            cand_imgs = prompt2img[cand_pair]
            for cand_img in cand_imgs:
                if args.target_rel not in cand_img:
                    num += 1
    print(num)

    for cand_pair in tqdm(cand_pairs):
        if cand_pair in prompt2img:
            cand_imgs = prompt2img[cand_pair]
            for cand_img in cand_imgs:
                if args.target_rel not in cand_img:
                    # num += 1
                    # for source prompt, do not add rels,
                    source_prompt, subj_name, obj_name, subj_id, obj_id = find_subjobj(cand_pair)
                    target_prompt = cand_pair.format('<R>')
                    prompts = [source_prompt, target_prompt]

                    # source image
                    SOURCE_IMAGE_PATH = osp.join(args.data_dir, cand_img)
                    source_image = load_image(SOURCE_IMAGE_PATH, device)

                    # start_code = torch.randn([1, 4, 64, 64], device=device)
                    # start_code = start_code.expand(len(prompts), -1, -1, -1)
                    # uncond_embeddings = torch.randn([1, 77, 768], device=device)

                    key = f"{source_prompt}*{target_prompt}*{subj_name}*{obj_name}*{args.target_rel}*{cand_img.replace('/', '-')}"



                    # invert the source image with null text inverion
                    start_code, uncond_embeddings = model.invert_with_null_text_optimize(source_image,
                                                                                         source_prompt,
                                                                                         guidance_scale=7.5,
                                                                                         num_inference_steps=50,
                                                                                         num_inner_steps=10,
                                                                                         )
                    start_code = start_code.expand(len(prompts), -1, -1, -1)

                    editor = MutualSelfAttentionControl(STEP, LAYER)
                    # editor = MutualSelfAttentionControlMaskAuto(STEP, LAYER, ref_token_idx=subj_id+obj_id, cur_token_idx=subj_id+obj_id)
                    editor2 = MutualRelControlMaskAuto(STEP, LAYER, ref_subj_token_idx=subj_id, ref_obj_token_idx=obj_id,
                                                       cur_subj_token_idx=subj_id, cur_obj_token_idx=obj_id)

                    regiter_attention_editor_diffusers(model, editor)

                    # inference the synthesized image
                    image_masactrl_ori_token = model([source_prompt, cand_pair.format(args.target_rel)],
                                                     latents=start_code,
                                                     unconditioning=uncond_embeddings,
                                                     guidance_scale=7.5
                                                     )
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
                                           image_masactrl_ori_token[-1:],
                                           image_masactrl[-1:],
                                           image_marectrl[-1:]], dim=0)
                    save_image(out_image, os.path.join(out_dir, f"all*{key}.png"))
                    save_image(out_image[0], os.path.join(out_dir, f"source*{key}.png"))
                    save_image(out_image[2], os.path.join(out_dir,f"masaori*{key}.png"))
                    save_image(out_image[3], os.path.join(out_dir, f"masa*{key}.png"))
                    save_image(out_image[4], os.path.join(out_dir, f"mare*{key}.png"))

                    records[key] = [start_code, uncond_embeddings, subj_name, obj_name, prompts]

                    #break
        #break

    #################### GSAM refinement
    gsam_model = GSAM()
    for k, v in records.items():
        start_code, uncond_embeddings, subj_name, obj_name, prompts = v
        try:
            source_detections = gsam_model.run_single_image(os.path.join(out_dir, f"source*{k}.png"),
                                                            classes=[subj_name, obj_name],
                                                            dsize=(512, 512),
                                                            de_duplicated=True)

            mare_detections = gsam_model.run_single_image(os.path.join(out_dir, f"mare*{k}.png"),
                                                          classes=[subj_name, obj_name],
                                                          dsize=(512, 512),
                                                          de_duplicated=True)

            source_masks = source_detections.mask
            mare_masks = mare_detections.mask

            editor3 = MutualRelGSAMControlMaskAuto(source_masks, mare_masks, STEP, LAYER, thres=0.1)
            regiter_attention_editor_diffusers(model, editor3)
            image_marectrl_gsam = model(prompts, latents=start_code, unconditioning=uncond_embeddings, guidance_scale=7.5)[
                                  -1:]
            save_image(image_marectrl_gsam, os.path.join(out_dir, f"gsam*{k}.png"))
        except:
            continue


elif args.data_dir == 'data/hico_actions':
    act2obj = json.load(open(osp.join(args.data_dir, 'act2obj.json')))
    obj2img = json.load(open(osp.join(args.data_dir, 'obj2img.json')))

    # 1. query from template: what object combiantions are suitable for target relationships:
    cand_objs = act2obj[args.target_rel]
    num = 0
    for cand_obj in tqdm(cand_objs):
        if cand_obj in obj2img:
            cand_imgs = obj2img[cand_obj]
            for cand_img in cand_imgs:
                if args.target_rel not in cand_img:
                    num += 1
    print(num)

    for cand_obj in tqdm(cand_objs):
        if cand_obj in obj2img:
            cand_imgs = obj2img[cand_obj]
            for cand_img in cand_imgs:
                if args.target_rel not in cand_img:
                    # num += 1
                    # for source prompt, do not add rels,
                    source_prompt, subj_name, obj_name, subj_id, obj_id = 'human, '+cand_obj, 'human', cand_obj, 1, 3
                    target_prompt = f'human <R> {cand_obj}'
                    prompts = [source_prompt, target_prompt]

                    # source image
                    SOURCE_IMAGE_PATH = osp.join(args.data_dir, cand_img)
                    source_image = load_image(SOURCE_IMAGE_PATH, device)

                    # start_code = torch.randn([1, 4, 64, 64], device=device)
                    # start_code = start_code.expand(len(prompts), -1, -1, -1)
                    # uncond_embeddings = torch.randn([1, 77, 768], device=device)

                    key = f"{source_prompt}*{target_prompt}*{subj_name}*{obj_name}*{args.target_rel}*{cand_img.replace('/', '-')}"



                    # invert the source image with null text inverion
                    start_code, uncond_embeddings = model.invert_with_null_text_optimize(source_image,
                                                                                         source_prompt,
                                                                                         guidance_scale=7.5,
                                                                                         num_inference_steps=50,
                                                                                         num_inner_steps=10,
                                                                                         )
                    start_code = start_code.expand(len(prompts), -1, -1, -1)

                    editor = MutualSelfAttentionControl(STEP, LAYER)
                    # editor = MutualSelfAttentionControlMaskAuto(STEP, LAYER, ref_token_idx=subj_id+obj_id, cur_token_idx=subj_id+obj_id)
                    editor2 = MutualRelControlMaskAuto(STEP, LAYER, ref_subj_token_idx=subj_id, ref_obj_token_idx=obj_id,
                                                       cur_subj_token_idx=subj_id, cur_obj_token_idx=obj_id)

                    regiter_attention_editor_diffusers(model, editor)

                    # inference the synthesized image
                    image_masactrl_ori_token = model([source_prompt, f'human {args.target_rel} {cand_obj}'],
                                                     latents=start_code,
                                                     unconditioning=uncond_embeddings,
                                                     guidance_scale=7.5
                                                     )
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
                                           image_masactrl_ori_token[-1:],
                                           image_masactrl[-1:],
                                           image_marectrl[-1:]], dim=0)
                    save_image(out_image, os.path.join(out_dir, f"all*{key}.png"))
                    save_image(out_image[0], os.path.join(out_dir, f"source*{key}.png"))
                    save_image(out_image[2], os.path.join(out_dir,f"masaori*{key}.png"))
                    save_image(out_image[3], os.path.join(out_dir, f"masa*{key}.png"))
                    save_image(out_image[4], os.path.join(out_dir, f"mare*{key}.png"))

                    records[key] = [start_code, uncond_embeddings, subj_name, obj_name, prompts]

                    #break
        #break

    #################### GSAM refinement
    gsam_model = GSAM()
    for k, v in records.items():
        start_code, uncond_embeddings, subj_name, obj_name, prompts = v
        try:
            source_detections = gsam_model.run_single_image(os.path.join(out_dir, f"source*{k}.png"),
                                                            classes=[subj_name, obj_name],
                                                            dsize=(512, 512),
                                                            de_duplicated=True)

            mare_detections = gsam_model.run_single_image(os.path.join(out_dir, f"mare*{k}.png"),
                                                          classes=[subj_name, obj_name],
                                                          dsize=(512, 512),
                                                          de_duplicated=True)

            source_masks = source_detections.mask
            mare_masks = mare_detections.mask

            editor3 = MutualRelGSAMControlMaskAuto(source_masks, mare_masks, STEP, LAYER, thres=0.1)
            regiter_attention_editor_diffusers(model, editor3)
            image_marectrl_gsam = model(prompts, latents=start_code, unconditioning=uncond_embeddings, guidance_scale=7.5)[
                                  -1:]
            save_image(image_marectrl_gsam, os.path.join(out_dir, f"gsam*{k}.png"))
        except:
            continue