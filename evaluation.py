import numpy as np
import os
import os.path as osp
from PIL import Image
import json
import glob
import clip
import glob
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='./workdir/visualization_for_cvpr24/reversion_benchmark_v1_run2_index_correct')  # ./workdir/visualization_for_cvpr24/hico_actions
args = parser.parse_args()

model, preprocess = clip.load('../cache_models/clip/ViT-B-16.pt', device="cuda")

trans_dict = {"back2back": "back to back with", "hug": "hug", "shake_hands": "shake hands with"}

all_rels_dir = glob.glob(osp.join(args.output_dir, '*'))

text_sims_all_rel = torch.zeros((0, 3)).float()
img_sims_all_rel = torch.zeros((0, 3)).float()
with torch.no_grad():
    for rel_dir in all_rels_dir:
        target_rel = rel_dir.split('/')[-1]
        source_img_paths = glob.glob(osp.join(rel_dir, 'source*'))
        source_img_names = [p.split('/')[-1] for p in source_img_paths]
        keys = ['*'.join(p.split('*')[1:])[:-4] for p in source_img_names]
        masa_img_paths = [osp.join(rel_dir, "masa*"+k+".png") for k in keys]
        mare_img_paths = [osp.join(rel_dir, "mare*"+k+".png") for k in keys]
        masaori_img_paths = [osp.join(rel_dir, "masaori*"+k+".png") for k in keys]

        ##
        #print(masa_img_paths, mare_img_paths, masaori_img_paths)
        ##
        text_sims_cur_rel = torch.zeros((0, 3)).float()
        img_sims_cur_rel = torch.zeros((0, 3)).float()

        for k, source_img_path, masa_img_path, mare_img_path, masaori_img_path in tqdm(zip(keys, source_img_paths, masa_img_paths, mare_img_paths, masaori_img_paths)):
            source_img = preprocess(Image.open(source_img_path)).unsqueeze(0).cuda()
            masa_img = preprocess(Image.open(masa_img_path)).unsqueeze(0).cuda()
            mare_img = preprocess(Image.open(mare_img_path)).unsqueeze(0).cuda()
            masaori_img = preprocess(Image.open(masaori_img_path)).unsqueeze(0).cuda()
            all_imgs = torch.cat([masaori_img, masa_img, mare_img], 0)

            source_feat = model.encode_image(source_img)
            source_feat = source_feat / torch.norm(source_feat)

            edited_feat = model.encode_image(all_imgs)
            edited_feat = edited_feat / torch.norm(edited_feat)

            target_text = k.split('*')[2] + ' ' + trans_dict[k.split('*')[4]] + ' ' + k.split('*')[3] if k.split('*')[4] in trans_dict else k.split('*')[2] + ' ' + k.split('*')[4] + ' ' + k.split('*')[3]
            #target_text = trans_dict[k.split('*')[4]]
            print(target_text)
            target_text = clip.tokenize([target_text]).cuda()
            text_feat = model.encode_text(target_text)
            text_feat = text_feat / torch.norm(text_feat)

            img_sims = torch.mm(source_feat, edited_feat.transpose(0, 1)).cpu()
            text_sims = torch.mm(text_feat, edited_feat.transpose(0, 1)).cpu()
            #print(img_sims, text_sims)

            text_sims_cur_rel = torch.cat((text_sims_cur_rel, text_sims))
            img_sims_cur_rel = torch.cat((img_sims_cur_rel, img_sims))
            text_sims_all_rel = torch.cat((text_sims_all_rel, text_sims))
            img_sims_all_rel = torch.cat((img_sims_all_rel, img_sims))
            #break

        print(target_rel, text_sims_cur_rel.mean(0), img_sims_cur_rel.mean(0))
        #break

print(text_sims_all_rel.mean(0), img_sims_all_rel.mean(0))
        
        
        
        
    
    