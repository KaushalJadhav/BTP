import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data.transforms_bbox as Tr
from data.voc import VOC_box
from configs.defaults import _C
from models.ClsNet import Labeler

import wandb
from utils.wandb import init_wandb, wandb_log
from tqdm import tqdm


def my_collate(batch):
    '''
    This is to assign a batch-wise index for each box.
    '''
    sample = {}
    img = []
    bboxes = []
    bg_mask = []
    batchID_of_box = []
    for batch_id, item in enumerate(batch):
        img.append(item[0])
        bboxes.append(item[1]) 
        bg_mask.append(item[2])
        for _ in range(len(item[1])):
            batchID_of_box += [batch_id]
    sample["img"] = torch.stack(img, dim=0)
    sample["bboxes"] = torch.cat(bboxes, dim=0)
    sample["bg_mask"] = torch.stack(bg_mask, dim=0)[:,None]
    sample["batchID_of_box"] = torch.tensor(batchID_of_box, dtype=torch.long)
    return sample


def main(cfg):
    """
    Main function

    Create dataloaders, train the model, and save the trained model.

    Inputs:
    - cfg: config file

    Outputs:
    - Trained model saved locally and on wandb
    """
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    tr_transforms = Tr.Compose([
        Tr.GaussianNoise(),
        Tr.RandomScale(0.5, 1.5),
        Tr.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
        Tr.RandomHFlip(0.5), 
        Tr.ColorJitter(0.5,0.5,0.5,0),
        Tr.Normalize_Caffe(),
    ])
    trainset = VOC_box(cfg, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, collate_fn=my_collate, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     model.to(device)
    #     model.module.backbone.load_state_dict(torch.load(f"./weights/{cfg.MODEL.WEIGHTS}"), strict=False)
    #     params = model.module.get_params()
    # else:
    #     model.to(device)
    #     model.backbone.load_state_dict(torch.load(f"./weights/{cfg.MODEL.WEIGHTS}"), strict=False)
    #     params = model.get_params()

    model.to(device)
    model.backbone.load_state_dict(torch.load(f"./weights/{cfg.MODEL.WEIGHTS}"), strict=False)
    params = model.get_params()

    lr = cfg.SOLVER.LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    optimizer = optim.SGD(
        [{"params":params[0], "lr":lr,    "weight_decay":wd},
         {"params":params[1], "lr":2*lr,  "weight_decay":0 },
         {"params":params[2], "lr":10*lr, "weight_decay":wd},
         {"params":params[3], "lr":20*lr, "weight_decay":0 }], 
        momentum=cfg.SOLVER.MOMENTUM
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Initializing W&B
    init_wandb(cfg)
    
    model = model.train()
    iterator = iter(train_loader)

    for it in tqdm(range(1, cfg.SOLVER.MAX_ITER+1)):

        try:
            sample = next(iterator)
        except:
            iterator = iter(train_loader)
            sample = next(iterator)

        img = sample["img"]
        bboxes = sample["bboxes"]
        bg_mask = sample["bg_mask"]
        batchID_of_box = sample["batchID_of_box"]
        ind_valid_bg_mask = bg_mask.mean(dim=(1,2,3)) > 0.125 # This is because VGG16 has output stride of 8.
        
        img = img.to(device)
        bg_mask = bg_mask.to(device)
        logits = model(img, bboxes, batchID_of_box, bg_mask, ind_valid_bg_mask, GAP=cfg.MODEL.GAP)
        logits = logits[...,0,0]
        fg_t = bboxes[:,-1][:,None].expand(bboxes.shape[0], np.prod(cfg.MODEL.ROI_SIZE))
        fg_t = fg_t.flatten().long()
        target = torch.zeros(logits.shape[0], dtype=torch.long)
        target[:fg_t.shape[0]] = fg_t

        loss = criterion(logits, target.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Iteration {it} Loss {loss.item()}")
        # Logging on W&B
        wandb_log(loss.item(), optimizer.param_groups[0]["lr"], it)

    torch.save(model.state_dict(), f"./weights/{cfg.NAME}.pt")
    wandb.save(f"./weights/{cfg.NAME}.pt")

    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)