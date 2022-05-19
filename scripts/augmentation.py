# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder                                    
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

#import cv2
import matplotlib.image as mpimg

# augmentation for training

gt_size = 512

gt_folder = '/home/ubuntu/whyeo/DACON/LG_Camera/submission_code/datasets/train/gt_crops'
lq_folder = '/home/ubuntu/whyeo/DACON/LG_Camera/submission_code/datasets/train/input_crops'
paths = paired_paths_from_folder([lq_folder, gt_folder], ['lq', 'gt'], '{}')

for index in range(len(paths)):
  scale = 1
  file_client = FileClient('disk')
  gt_path = paths[index]['gt_path']
  # print('gt path,', gt_path)
  img_bytes = file_client.get(gt_path, 'gt')
  try:
      img_gt = imfrombytes(img_bytes, float32=True)
  except:
      raise Exception("gt path {} not working".format(gt_path))

  lq_path = paths[index]['lq_path']
  # print(', lq path', lq_path)
  img_bytes = file_client.get(lq_path, 'lq')
  try:
      img_lq = imfrombytes(img_bytes, float32=True)
  except:
      raise Exception("lq path {} not working".format(lq_path))

  img_gt, img_lq

  # padding
  img_gt, img_lq = padding(img_gt, img_lq, gt_size)
  
  # random crop
  img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                      gt_path)
  # flip, rotation
  img_gt, img_lq = augment([img_gt, img_lq], True, True)

  gt_fname = gt_path.split('/')[-1]
  lq_fname = lq_path.split('/')[-1]
  mpimg.imsave(f'[~~~root_path~~~]/datasets/augmentation_result/gt/{gt_fname}', img_gt)
  mpimg.imsave(f'[~~~root_path~~~]/datasets/augmentation_result/input/{lq_fname}', img_lq)
