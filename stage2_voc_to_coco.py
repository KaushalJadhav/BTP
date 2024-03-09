import os
import sys
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import data.transforms_bbox as Tr
from data.coco import COCO_box
from configs.defaults import _C
from models.ClsNet import Labeler, pad_for_grid
from utils.densecrf import DENSE_CRF

logger = logging.getLogger("stage2")

#format -> coco_id : ['class_name',voc_id]
coco_to_voc_map =  {0  : ['unlabelled',0],
                    5  : ['aeroplane', 1],
                    2  : ['bicycle', 2],
                    16 : ['bird',  3],
                    9  : ['boat',  4],
                    44 : ['bottle',  5],
                    6  : ['bus', 6],
                    3  : ['car', 7],
                    17 : ['cat', 8],
                    62 : ['chair', 9],
                    21 : ['cow', 10],
                    67 : ['dining table',  11],
                    18 : ['dog', 12],
                    19 : ['horse', 13],
                    4  : ['motorcycle',14],
                    1  : ['person',15],
                    64 : ['potted plant',16],
                    20 : ['sheep', 17],
                    63 : ['couch', 18],
                    7  : ['train', 19],
                    72 : ['tv',  20]}  

def main(cfg):
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    ann_path =  os.path.join(cfg.DATA.ROOT,'annotations/instances_train2017.json')
    data_root = os.path.join(cfg.DATA.ROOT,'train2017')
    
    tr_transforms = Tr.Normalize_Caffe()
    trainset = COCO_box(data_root,ann_path,cfg,tr_transforms)
    train_loader = DataLoader(trainset, batch_size=1)
    
    model = Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE).cuda()
    model_stage_1 = wandb.restore(cfg.WAND.RESTORE_NAME, run_path=cfg.WANDB.RESTORE_RUN_PATH) #restoring weights from stage1
    model.load_state_dict(torch.load(model_stage_1.name))
    WEIGHTS = torch.clone(model.classifier.weight.data)
    model.eval()
    
    bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = cfg.MODEL.DCRF
    dCRF = DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
    
    if cfg.SAVE_PSEUDO_LABLES:
        folder_name = os.path.join(cfg.DATA.ROOT, cfg.NAME)
        if os.path.isdir(folder_name)==0:
            os.mkdir(folder_name)
        save_paths = []
        for txt in ("Y_crf_COCO", "Y_ret_COCO"):
            sub_folder = folder_name + f"/{txt}"
            if(os.path.isdir(sub_folder)==0):    
                os.mkdir(sub_folder)
            save_paths += [os.path.join(sub_folder, "{}.png")]
            
    logger.info(f"START {cfg.NAME} -->")
    with torch.no_grad():
        for it, (img, bboxes, bg_mask) in enumerate(tqdm(train_loader)):
            '''
            img     : (1,3,H,W) float32
            bboxes  : (1,K,5)   float32
            bg_mask : (1,H,W)   float32
            '''
            fn,rgb_img_path = trainset.filename(it)
            rgb_img = np.array(Image.open(rgb_img_path))

            bboxes = bboxes[0] # (1,K,5) --> (K,5)
            bg_mask = bg_mask[None] # (1,H,W) --> (1,1,H,W)
            img_H, img_W = img.shape[-2:]
            norm_H, norm_W = (img_H-1)/2, (img_W-1)/2
            bboxes[:,[0,2]] = bboxes[:,[0,2]]*norm_W + norm_W
            bboxes[:,[1,3]] = bboxes[:,[1,3]]*norm_H + norm_H
            bboxes = bboxes.long()
            gt_labels = bboxes[:,4].unique() #bboxes : (wmin, hmin, wmax, hmax, cls) 1 x 5 numpy float32
            
            features = model.get_features(img.cuda())
            features = F.interpolate(features, img.shape[-2:], mode='bilinear', align_corners=True)
            padded_features = pad_for_grid(features, cfg.MODEL.GRID_SIZE)
            padded_bg_mask = pad_for_grid(bg_mask.cuda(), cfg.MODEL.GRID_SIZE)
            grid_bg, valid_gridIDs = model.get_grid_bg_and_IDs(padded_bg_mask, cfg.MODEL.GRID_SIZE)
            bg_protos = model.get_bg_prototypes(padded_features, padded_bg_mask, grid_bg, cfg.MODEL.GRID_SIZE)
            bg_protos = bg_protos[0,valid_gridIDs] # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
            normed_bg_p = F.normalize(bg_protos)
            normed_f = F.normalize(features)
            bg_attns = F.relu(torch.sum(normed_bg_p*normed_f, dim=1))
            bg_attn = torch.mean(bg_attns, dim=0, keepdim=True) # (len(valid_gridIDs),H,W) --> (1,H,W)
            bg_attn[bg_attn < cfg.MODEL.BG_THRESHOLD * bg_attn.max()] = 0
            Bg_unary = torch.clone(bg_mask[0]) # (1,H,W)
            region_inside_bboxes = Bg_unary[0]==0 # (H,W)
          
            Fg_unary = []
            for uni_cls in gt_labels: #gt_labels contains all unique class indices
                orig_cat_id = trainset.cat_id_map.index(uni_cls)
                if(orig_cat_id in coco_to_voc_map):                 # weights present for given class
                  voc_id = coco_to_voc_map[orig_cat_id][1]
                  w_c = WEIGHTS[voc_id][None]     
                  raw_cam = F.relu(torch.sum(w_c*features, dim=1)) # (1,H,W)
                  normed_cam = torch.zeros_like(raw_cam)
                  for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                      denom = raw_cam[:,hmin:hmax,wmin:wmax].amax() + 1e-12
                      normed_cam[:,hmin:hmax,wmin:wmax] = raw_cam[:,hmin:hmax,wmin:wmax] / denom
                  Fg_unary += [normed_cam]
                else:                                               # no weights present, use 1-u0
                  class_mask = torch.zeros_like(Bg_unary)
                  for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                      class_mask[:,hmin:hmax,wmin:wmax] = 1
                  temp_attention_map = (1 - Bg_unary) * class_mask 
                  Fg_unary += [temp_attention_map.cuda()]
                                 
            Fg_unary = torch.cat(Fg_unary, dim=0).detach().cpu()           
            unary = torch.cat((Bg_unary,Fg_unary), dim=0)
            unary[:,region_inside_bboxes] = torch.softmax(unary[:,region_inside_bboxes], dim=0)
            refined_unary = dCRF.inference(rgb_img, unary.numpy())
            
            # (Out of bboxes) reset Fg scores to zero
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                mask = np.zeros((img_H,img_W))
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                    mask[hmin:hmax,wmin:wmax] = 1
                refined_unary[idx_cls] *= mask

            # Y_crf
            tmp_mask = refined_unary.argmax(0)
            Y_crf = np.zeros_like(tmp_mask, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                Y_crf[tmp_mask==idx_cls] = uni_cls
            Y_crf[tmp_mask==0] = 0
               
            # Y_ret
            tmp_Y_crf = torch.from_numpy(Y_crf) # (H,W)
            gt_labels_with_Bg = [0] + gt_labels.tolist()
            corr_maps = []
            for idx_cls, uni_cls in enumerate(gt_labels_with_Bg):
                orig_cat_id=0
                if uni_cls!=0:
                  orig_cat_id = trainset.cat_id_map.index(uni_cls)
                
                indices = tmp_Y_crf==uni_cls
                if indices.sum():
                    normed_p = F.normalize(features[...,indices].mean(dim=-1))   # (1,dims)
                    corr = F.relu((normed_f*normed_p[...,None,None]).sum(dim=1)) # (1,H,W)
                else:
                    if (orig_cat_id in coco_to_voc_map):
                        voc_id = coco_to_voc_map[orig_cat_id][1]
                        normed_w = F.normalize(WEIGHTS[voc_id][None])
                        corr = F.relu((normed_f*normed_w).sum(dim=1)) # (1,H,W)
                    else:
                        _,ht,wid = unary.shape 
                        corr = F.normalize(unary[idx_cls].reshape((1,ht,wid))).cuda()
                corr_maps.append(corr)
            corr_maps = torch.cat(corr_maps) # shape : (1+len(gt_labels),H,W)

            # (Out of bboxes) reset Fg correlations to zero
            for idx_cls, uni_cls in enumerate(gt_labels_with_Bg):
                if uni_cls == 0:
                    corr_maps[idx_cls, ~region_inside_bboxes] = 1
                else:
                    mask = torch.zeros(img_H,img_W).type_as(corr_maps)
                    for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                        mask[hmin:hmax,wmin:wmax] = 1
                    corr_maps[idx_cls] *= mask

            tmp_mask = corr_maps.argmax(0).detach().cpu().numpy()
            Y_ret = np.zeros_like(tmp_mask, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                Y_ret[tmp_mask==idx_cls] = uni_cls
            Y_ret[tmp_mask==0] = 0
            
            if cfg.SAVE_PSEUDO_LABLES:
                for pseudo, save_path in zip([Y_crf, Y_ret], save_paths):
                    Image.fromarray(pseudo).save(save_path.format(fn))

    logger.info(f"END {cfg.NAME} -->")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)
