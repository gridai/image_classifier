import os
import numpy as np
from tqdm.notebook import tqdm
from argparse import ArgumentParser

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import albumentations # augmentations library
import timm # image model library
import torch #import torch


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import flash
from flash.vision import ImageClassificationData
from flash.vision import ImageClassifier
from flash import Trainer
from torch.nn import functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

# import our libraries
from flash import download_data


def load_data(path, batch_size=32, num_workers=4):
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    test_path = os.path.join(path, 'test')

    # 2. Load the data
    datamodule = ImageClassificationData.from_folders(
        train_folder=train_path,
        valid_folder=val_path,
        test_folder=test_path,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    return datamodule


def train_cli(args):
    pl.seed_everything(args.seed)

    data = load_data(args.data_dir, args.batch_size, args.num_workers)

    print('train samples:', len(data.train_dataloader().dataset))
    print('valid samples:', len(data.val_dataloader().dataset))

    if args.backbone == "resnet200d":
        model_name = 'resnet200d'#'resnet200d'
        model = timm.create_model(model_name, pretrained=True)
        num_features = model.num_features
        model.global_pool = torch.nn.Identity()
        model.fc = torch.nn.Identity()
        pooling = torch.nn.AdaptiveAvgPool2d(1)
        backbone = (model, model.num_features)

    elif args.backbone =="ViT":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        backbone = (model, model.num_features)

    else:
        backbone = args.backbone

    ## Create Flash Classifier
    model = ImageClassifier(backbone=backbone,
                            num_classes=data.num_classes,
                            optimizer = torch.optim.Adam,
                            learning_rate=args.learning_rate
                            )

    ## Fine Tune
    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs
    )
    trainer.finetune(model, data, strategy='no_freeze')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--valid_split', type=float, default=.1)
    parser.add_argument('--backbone', type=str, default="resnet18")
    parser.add_argument('--data_dir', type=str, default=os.getcwd())
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--gpus', type=int, default=None)
    args = parser.parse_args()
    train_cli(args)
