import os
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copyfile

def split_dataset(train_data_path, train_input_label_path, train_size=0.85, random_state=1004):
    data_list = os.listdir(train_data_path)
    data_len = len(data_list)

    train_len = int(data_len * train_size)
    validation_len = data_len - train_len

    np.random.seed(random_state)
    shuffled = np.random.permutation(data_len)

    train_txt_path = "./datasets/split_train.txt"
    validation_txt_path = "./datasets/split_validation.txt"

    os.makedirs('./datasets/train_input_img')
    os.makedirs('./datasets/train_label_img')
    os.makedirs('./datasets/val_input_img')
    os.makedirs('./datasets/val_label_img')

    f = open(train_txt_path, 'w')
    print("\n >>> Currently spliting training set.")
    for idx in tqdm(shuffled[:train_len]):
        img_id = data_list[idx].split("_")[2]
        img_input = data_list[idx]
        label_img = data_list[idx][:6] + "label" + data_list[idx][11:]

        copyfile(os.path.join(train_data_path,img_input), './datasets/train_input_img/'+img_input)
        copyfile(os.path.join(train_input_label_path , label_img), './datasets/train_label_img/' + img_input)

        train_line = "%s, %s, %s \n" % (img_id, img_input, label_img)
        f.write(train_line)
    f.close()

    f = open(validation_txt_path, 'w')
    print("\n >>> Currently spliting validation set.")
    for idx in tqdm(shuffled[train_len:]):
        img_id = data_list[idx].split("_")[2]
        img_input = data_list[idx]
        label_img = data_list[idx][:6] + "label" + data_list[idx][11:]

        copyfile(os.path.join(train_data_path , img_input), './datasets/val_input_img/' + img_input)
        copyfile(os.path.join(train_input_label_path , label_img), './datasets/val_label_img/' + img_input)
        train_line = "%s, %s, %s \n" % (img_id, img_input, label_img)
        f.write(train_line)
    f.close()

    print("\n Completely split all dataset.")



parser = argparse.ArgumentParser()
parser.add_argument('-train', type=str, help='original train_input_img path')
parser.add_argument('-label', type=str, help='original train_label_img path')
args = parser.parse_args()


split_dataset(args.train, args.label)



