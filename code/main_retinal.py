from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
from models_retinal import CHyVAE
import os
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class DC_dataset(Dataset):
    def __init__(self, df, root_dir, type, target=None, transform=None):
        self.df = df
        self.target = target
        self.root_dir = root_dir
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pt_dir = self.df.loc[idx, 'ptid']
        ins_dir = self.df.loc[idx, 'instance']
        if self.target:
            label = self.df.loc[idx, self.target]
            label = torch.tensor(label)
        else:
            label = torch.tensor(np.nan)

        if self.type == 'IR':
            img_name = 'ir.png'
        elif self.type == 'OCT':
            middle = self.df.loc[idx, 'middle']
            img_name = 'oct-0' + str(middle) + '.png'
        img_path = os.path.join(self.root_dir, pt_dir, 'macOCT', ins_dir, img_name)
        img_pre = cv2.imread(img_path, flags=0)
        img = np.expand_dims(img_pre, axis=2)

        # default transform
        img = img / np.float32(255)
        img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)

        return img, label


def dataloader_create(train, test, root_dir, type='OCT', batch_size=100, shuffle=False, label = None, augmentation=None):
    """
    create train and test dataloaders directly from train and test csv

    :param train: csv file for training set
    :param test: csv file for testing set
    :param root_dir: root directory for images
    :param type: type of images
    :param batch_size: batch_size
    :param shuffle: shuffle indicator for dataloader
    :param label: column used for label
    :param augmentation: additional augmentation
    :return:
    """
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)

    train_set = DC_dataset(train_df, root_dir, type, label, augmentation)
    test_set = DC_dataset(test_df, root_dir, type, label, augmentation)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return train_loader, test_loader


def set_args(parser):
    parser.add_argument('--batch_size', type=int,
                        default=50, help='input batch size')
    parser.add_argument('--z_dim', type=int, default=20,
                        help='latent vector dim')
    parser.add_argument('--nu', type=int, default=33,
                        help='degrees of freedom (> z_dim + 1)')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='numbers of training steps')
    parser.add_argument('--run', type=int, default=None,
                        help='run number')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = set_args(parser)
    args = parser.parse_args()

    train_path = '/data/project/jiyu/dataset_prep/imgs_label_0213_train.csv'
    test_path = '/data/project/jiyu/dataset_prep/imgs_label_0213_test.csv'
    root = '/data/macoct/'
    train, test = dataloader_create(train_path, test_path, root, label='label')

    model = CHyVAE(train, test, n_clusters=20, z_dim=args.z_dim, im_h=496, im_w=768, channels=1, batch_size=args.batch_size, n_epochs = args.n_epoch, nu=args.nu, prior_cov=np.eye(args.z_dim), run_no=args.run)

    model.train()
