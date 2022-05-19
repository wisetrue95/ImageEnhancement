# DACON - Camera Image Quality Enhancement AI Contest
https://dacon.io/competitions/official/235746/overview/description

This project is based on megvii-model's [HINet implementation](https://github.com/megvii-model/HINet).

## Installation
```
git clone https://github.com/NextLevel-AI/CameraImageEnhancement
cd CameraImageEnhancement
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

## environment
- Ubuntu 16.04
- GPU : NVIDIA Tesla V100 x 2
- Cuda 10.1
- python 3.7
- torch 1.7.1
- cudnn 7.6
- addict 2.4.0
- future 0.18.2
- lmdb 1.2.1
- numpy 1.19.5
- opencv-python 4.5.3.56
- Pillow 8.3.1
- pyyaml 5.4.1
- requests 2.26.0
- scikit-image 0.17.2
- scipy 1.5.4
- tb-nightly 2.6.0a20210728
- tqdm 4.61.2
- yapf 0.31.0

## Pre-processing
```
mkdir datasets
python scripts/split.py -train [original train_input_img path] -label [original train_label_img path]
python scripts/preprocessing.py
```
## Data augmentation
(Data augmentation code is included during training process. This code is for saving some sample augmented images)

`python scripts/augmentation.py`

## Training
`python basicsr/train.py -opt options/train/LG_HINet_Crop512_Step384.yml`

## Test
`python basicsr/inference.py  -opt options/inference/HINet-Inference.yml`

### Weight path
**`weights/HINet-136k.pth`**





