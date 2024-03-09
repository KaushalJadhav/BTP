# [RE] Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation

This repository is the PyTorch and PyTorch Lightning implementation of the paper ["Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation"](https://arxiv.org/pdf/2104.00905.pdf). It is well documented version of the original repository with the code flow available [here](assets/codeflow.pdf). The paper address the problem of weakly-supervised semantic segmentation (WSSS) using bounding box annotations by proposing two novel methods:
- **Background-aware pooling (BAP)**, to extract high-quality pseudo segmentation labels
- **Noise-aware Loss (NAL)**, to make the networks less susceptible to incorrect labels

<p align="center">
<a><img src="https://i.ibb.co/rcn1F2D/error.png" alt="error" border="0"><br>Visual comparison of pseudo ground-truth labels</a>
</p>

For more information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/BANA/)].

## Requirements

The code is developed and tested using `Python >= 3.6`. To install the requirements:

```bash
pip install -r requirements.txt
```

### To setup the dataset:
VOC:
```bash
bash data/setup_voc.bash /path-to-data-directory
```
### To generate background masks:
VOC:
```bash
python3 utils/voc_bbox_setup.py --data-root /path-to-data-directory
```
<br>COCO:
```bash
python3 utils/coco_bbox_setup.py --data-root /path-to-data-directory
```
Once finished, the folder `data` should be like this:
VOC:
```
    data   
    └── VOCdevkit
        └── VOC2012
            ├── JPEGImages
            ├── SegmentationClassAug
            ├── Annotations
            ├── ImageSets
            ├── BgMaskfromBoxes
            └── Generation
                ├── Y_crf
                └── Y_ret
```

## Training

The training procedure is divided into 3 stages and example commands for each have been given below. Hyperparameters can be adjusted accordingly in the corresponding configuration files.

### **Stage 1:** Training the classification network

Change the `MODEL.GAP` parameter in config file to train the model: 

1. `True`: With the Global Average Pooling method 
2. `False`: With the proposed Background Aware Pooling method

By default, all the models are trained using Augmented PASCAL VOC containing 10,582 images and can be trained using Non-Augmented dataset by changing the `DATA.AUG` parameter to `False`. 

```bash
python3 stage1.py --config-file configs/stage1.yml --gpu-id 0
```

### **Stage 2:** Generating pseudo labels

```bash
python3 stage2.py --config-file configs/stage2.yml --gpu-id 0
```

### **Stage 3:** Training a CNN using the pseudo labels

DeepLab Large FOV (VGG Backbone):

```bash
python3 stage3.py --config-file configs/stage3_vgg.yml --gpu-id 0
```

DeepLab ASPP (Resnet Backbone): 

```bash
python3 stage3.py --config-file configs/stage3_res.yml --gpu-id 0
```

Change the `MODEL.LOSS` parameter in the corresponding config file to train the model: 

1. `NAL` : With the proposed Noise Aware Loss using Ycrf and Yret
2. `ER` : With the Entropy Regularization Loss using Ycrf and Yret
3. `BS` : With the Bootstraping Loss using Ycrf and Yret
4. `BASELINE` : With the CE Loss using only the relaible region obtained by Ycrf and Yret
5. `CE_CRF`: With the Cross Entropy Loss using Ycrf
6. `CE_RET`: With the Cross Entropy Loss using Yret

### VOC to COCO psuedo label Generation
Change the `DATA.ROOT` and `MODEL.NUM_CLASSES` in the Stage2 config file for psuedolabel generation on MS COCO 2017 train dataset 

```bash
python3 stage2_voc_to_coco.py --config-file configs/stage2.yml --gpu-id 0
```

## Evaluation

To evaluate the model on the validation set of Pascal VOC 2012 dataset before and after Dense CRF processing change the `DATA.MODE` parameter to `val` in the corresponding config file:

```bash
python3 stage3.py --config-file configs/stage3_vgg.yml --gpu-id 0
```

DeepLab ASPP (Resnet Backbone): 

```bash
python3 stage3.py --config-file configs/stage3_res.yml --gpu-id 0
```

Evaluation would be performed on raw validation set images to obtain the Mean Accuracy and IOU metrics pre and post-Dense CRF processing.
## Pre-trained Models and Pseudo Labels

- Pretrained models: [Link](https://drive.google.com/drive/folders/14F1vU7Gp-nIZVhPe2XzrbnyUYTmnt2Sz?usp=sharing)

- Pseudo Labels: [Link](https://drive.google.com/drive/folders/1wC9qr1lE_JN0Htrf0SfPhKz4AdqQ0zbt?usp=sharing)

## Quantitative Results

We achieve the following results:

- Comparison of pseudo labels on the PASCAL VOC 2012 validation set in terms of mIoU

| **Method**          | **Original Author's Results** | **Our Results** |
|:-------------------:|:-----------------------------:|:---------------:|
| **GAP**             | 76.1                          | 75.5            |
| **BAP Ycrf w/o u0** | 77.8                          | 77              |
| **BAP Ycrf**        | 79.2                          | 78.8            |
| **BAP Yret**        | 69.9                          | 69.9            |
| **BAP Ycrf & Yret** | 68.2                          | 72.7            |

- Comparison of mIoU scores using different losses on the PASCAL VOC 2012 training set. We provide both mIoU scores before/after applying DenseCRF

| **Method**                    | **Original Author's Results** | **Our Results** |
|:-----------------------------:|:-----------------------------:|:---------------:|
| **Baseline**                  | 61.8 / 67.5                   | 60.9 / 64.5     |
| **w/ Entropy Regularization** | 61.4 / 67.3                   | 60.8 / 64.1     |
| **w/ Bootstrapping**          | 61.9 / 67.6                   | 60.9 / 64.6      |
| **w/ Lwce**                   | 62.4 / 68.1                   | 61.4 / 64.8     |

- Quantitative comparison using DeepLab-V1 (VGG-16) on the PASCAL VOC 2012 dataset in terms of mIoU
    - Weakly supervised learning

    | **Method**          | **Original Author's Results** | **Our Results** |
    |:-------------------:|:-----------------------------:|:---------------:|
    | **w/ Ycrf**         | 67.8                          | 64.7            |
    | **w/ Yret**         | 66.1                          | 62.8            |
    | **w/ NAL**          | 68.1                          | 64.8            |
    | **w/ NAL (test)**   | 69.4                          | 65.6            |

- Quantitative comparison using DeepLab-V2 (ResNet-101) on the PASCAL VOC 2012 dataset in terms of mIoU
    - Weakly supervised learning

    | **Method**          | **Original Author's Results** | **Our Results** |
    |:-------------------:|:-----------------------------:|:---------------:|
    | **w/ Ycrf**         | 74.0                          | 67.0            |
    | **w/ Yret**         | 72.4                          | 70.2            |
    | **w/ NAL**          | 74.6                          | 70.8            |
    | **w/ NAL**          | 76.1                          | 71.7            |
    
- Quantitative comparison of psuedo lables on the MS-COCO train set for model trained on Pascal VOC
    
    | **Results on COCO train**    | **AP**   | **AP<sub>50</sub>** | **AP<sub>75</sub>** | **AP<sub>S</sub>** | **AP<sub>M</sub>**  | **AP<sub>L</sub>**  |
    |:----------------------------:|:---------|:--------:|:--------:|:-------:|:--------:|:--------:|
    | **BAP: Ycrf** (Authors)      | 11.7     | 28.7     | 8.0      | 3.0     | 15.0     | 27.1     |
    | **BAP: Ycrf** (Ours)         | 8.6      | 20.1     | 6.5      | 1.9     |  8.8     | 15.9     |
    | **BAP: Yret** (Authors)      | 9.0      | 30.1     | 2.8      | 4.4     | 10.2     | 16.2     |
    | **BAP: Yret** (Ours)         | 6.6      | 20.2     | 2.5      | 3.3     | 5.7      | 10.6     |

- Comparison of pseudo labels on the PASCAL VOC 2012 validation set in terms of mIoU for different values of Grid size

    | Grid Size |   1   |   2   |   3   |
    |:---------:|:-----:|:-----:|:-----:|
    |     1     | 75.82 | 75.77 | 75.65 |
    |     2     | 76.11 | 76.10 | 75.15 |
    |     3     | 75.87 | 75.78 | 75.81 |
    |     4     | 78.83 | 78.72 | 78.82 |
    |     5     | 74.16 | 74.07 | 74.02 |
    
- Quantitative comparison using DeepLab-V1 (VGG-16) on the PASCAL VOC 2012 dataset in terms of mIoU with different loss functions

    |         **Method**             | **Authors’ Results** | **Our Results** |
    |:------------------------------:|:--------------------:|:---------------:|
    |          **Baseline**          |    61.8 / 67.5       | 60.9 / 64.5     |
    | **w / Entropy Regularization** |    61.4 / 67.3       | 60.8 / 64.1     |
    |      **w / Bootstrapping**     |    61.9 / 67.6       | 60.9 / 64.6     |
    |          **w / Lwce**          |    62.4 / 68.1       | 61.4 / 64.8     |

## Qualitative Results

### Pseudo Labels

| Input Image                  | Ground Truth                | Y CRF                        | Y RET                        |
|------------------------------|-----------------------------|------------------------------|------------------------------|
| ![](assets/stage2/img/1.jpg) | ![](assets/stage2/gt/1.png) | ![](assets/stage2/crf/1.png) | ![](assets/stage2/ret/1.png) |
| ![](assets/stage2/img/2.jpg) | ![](assets/stage2/gt/2.png) | ![](assets/stage2/crf/2.png) | ![](assets/stage2/ret/2.png) |
| ![](assets/stage2/img/3.jpg) | ![](assets/stage2/gt/3.png) | ![](assets/stage2/crf/3.png) | ![](assets/stage2/ret/3.png) |
| ![](assets/stage2/img/4.jpg) | ![](assets/stage2/gt/4.png) | ![](assets/stage2/crf/4.png) | ![](assets/stage2/ret/4.png) |
| ![](assets/stage2/img/5.jpg) | ![](assets/stage2/gt/5.png) | ![](assets/stage2/crf/5.png) | ![](assets/stage2/ret/5.png) |

## Contributors

[Aryan Mehta](https://github.com/victorvini08), [Karan Uppal](https://github.com/karan-uppal3), [Kaushal Jadhav](https://github.com/KaushalJadhav), [Monish Natarajan](https://github.com/Monish-Natarajan) and [Mradul Agrawal](https://github.com/mradul2)

This repository is maintained by [AGV.AI (IIT Kharagpur)](http://www.agv.iitkgp.ac.in/)

## Bibtex

- [RE] Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation
    ```
    @article{Mehta:2022,
      title = {{[Re] Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation}},
      author = {Mehta, Aryan and Uppal, Karan and Jadhav, Kaushal and Natarajan, Monish and Agrawal, Mradul and Chakravarty, Debashish},
      journal = {ReScience C},
      year = {2022},
      doi = {10.5281/zenodo.6574677},
    }
    ```

- Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation
    ```
    @inproceedings{oh2021background,
      title     = {Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation},
      author    = {Oh, Youngmin and Kim, Beomjun and Ham, Bumsub},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2021},
    }
    ```

## Acknowledgments

- PASCAL VOC 2012 Setup adopted from [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md)
