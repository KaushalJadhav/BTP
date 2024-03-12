import numpy as np
import torch

import pydensecrf.densecrf as DCRF
from pydensecrf.utils import unary_from_softmax

from utils.metric import scores
from tqdm import tqdm
import torch.nn.functional as F

class DENSE_CRF(object):
    def __init__(self, bi_w, bi_xy_std, bi_rgb_std, pos_w=3, pos_xy_std=3, max_iter=10):
        self.bi_w, self.bi_xy_std, self.bi_rgb_std = bi_w, bi_xy_std, bi_rgb_std
        self.pos_w, self.pos_xy_std = pos_w, pos_xy_std
        self.max_iter = max_iter

    def inference(self, image, prob):
        '''
        image : array (HxWx3)
        unary : array (CxHxW)
        '''
        C, H, W = prob.shape
        img = np.ascontiguousarray(image)
        unary = unary_from_softmax(prob)
        unary = np.ascontiguousarray(unary)
        
        d = DCRF.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=img, compat=self.bi_w)
        
        Q = d.inference(self.max_iter)
        out = np.array(Q).reshape((C, H, W))
        return out
    
def dense_crf(cfg, data_loader, model):
    """
    Dense CRF evaluation function

    Apply the Dense CRF post processing to the model output and calculate the metrics.

    Inputs:
    - cfg: config file
    - data_loader: dataloader
    - model: model

    Outputs:
    - Mean Accuracy and IoU
    """
    with torch.no_grad():
        model.eval()
        label_trues = []
        label_preds = []
        # Initialising Dense CRF object
        bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = cfg.MODEL.DCRF
        dCRF = DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
        for batch in tqdm(data_loader):
            img, masks = batch
            ygt = masks[0]
            # Forward pass
            img = img.cuda()
            img_size = img.size()
            logit, feature_map = model(img, (img_size[2], img_size[3]))
            prob = F.softmax(logit, dim=1)[0].cpu().detach().numpy()
            img = img[0].cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
            ygt = ygt[0].cpu().detach().numpy()
            # Apply DenseCRF
            prob = dCRF.inference(img, prob)
            label = np.argmax(prob, axis=0)
            # Append labels for evaluation
            label_preds.append(label)
            label_trues.append(ygt)
        # Calculate final results
        results = scores(label_trues, label_preds, cfg.DATA.NUM_CLASSES)
        accuracy = results["Mean Accuracy"]
        iou = results["Mean IoU"]
        return accuracy, iou
