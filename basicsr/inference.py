# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str

import zipfile

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    output_path = opt['img_path'].get('output_img')
    os.makedirs(output_path, exist_ok=True)  # mkdir if not exist

    # create model
    model = create_model(opt)
    sub_imgs = []
    for test_loader in test_loaders:
        for idx, val_data in enumerate(test_loader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0].replace('_input', '') + '.png'
            print(img_name)
            output_fname = os.path.join(output_path, img_name)
            model.single_image_inference(val_data, output_fname)

            sub_imgs.append(output_fname)
    
    # save as submission.zip file  
    submission = zipfile.ZipFile(output_path + "submission.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()

if __name__ == '__main__':
    main()
