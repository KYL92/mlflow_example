# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import cv2
import timm
import numpy as np
import mlflow
import pandas as pd

import torch
import torchvision.transforms as T
import albumentations
import albumentations.pytorch
from torch.nn import functional as F
from torchmetrics import Accuracy

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import LightningCLI


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_df, transforms):        
        self.image_folder = image_folder
        self.label_df = label_df
        self.transforms = transforms

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):        
        image_fn = self.image_folder +\
            str(self.label_df.iloc[index,0]).zfill(5) + '.png'
        image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)        
        image = cv2.resize(image, dsize=(224,224))
        image = cv2.merge((image, image, image))

        label = self.label_df.iloc[index,1:].values.astype('float')

        if self.transforms:            
            image = self.transforms(image=image)['image'] / 255.0

        return image, label
    
class ImageClassifier(LightningModule):
    def __init__(self, model, lr=1.0, gamma=0.7, batch_size=32):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model or timm.create_model('gluon_resnext101_64x4d', num_classes=512)
        self.val_acc = Accuracy()
        self.loss_func = torch.nn.MultiLabelSoftMarginLoss()
        # self.conv2d = torch.nn.Conv2d(1, 3, 3, stride=1)
        self.FC = torch.nn.Linear(512, 26)

    def forward(self, x):

        x = F.relu(self.model(x))
        x = self.FC(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        self.log("train_loss", loss)
        self.log("train_lr", self.optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        self.val_acc(logits, y.int())
        self.log("valid_acc", self.val_acc)
        self.log("valid_loss", loss)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=self.hparams.gamma)

        return (
            {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "valid_acc",
                },
            }
        )


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        
        self.image_path = "./data/dirty_mnist_2nd/"
        self.label_path = "./data/dirty_mnist_2nd_answer.csv"
        self.data_set = pd.read_csv(self.label_path)
        self.valid_idx_nb = int(len(self.data_set) * (1 / 5))
        self.valid_idx = np.arange(0, self.valid_idx_nb)

        print('[info msg] validation fold idx !!\n')        
        print(self.valid_idx)
        print('=' * 50)

        self.train_data = self.data_set.drop(self.valid_idx)
        self.valid_data = self.data_set.iloc[self.valid_idx]

        self.mnist_transforms = {
            'train' : albumentations.Compose([
                    albumentations.RandomRotate90(),
                    albumentations.OneOf([
                        albumentations.GridDistortion(distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, p=1),
                        albumentations.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),        
                        albumentations.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=1),
                    ], p=1),    
                    albumentations.Cutout(num_holes=16, max_h_size=15, max_w_size=15, fill_value=0),
                    albumentations.pytorch.ToTensorV2(),
                ]),
            'valid' : albumentations.Compose([        
                albumentations.pytorch.ToTensorV2(),
                ]),
            'test' : albumentations.Compose([        
                albumentations.pytorch.ToTensorV2(),
                ]),
        }

    @property
    def transform(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def train_dataloader(self):
        train_dataset = DatasetMNIST(
                image_folder=self.image_path ,
                label_df=self.train_data,
                transforms=self.mnist_transforms['train']
            )
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=8)

    def val_dataloader(self):
        val_dataset = DatasetMNIST(
                image_folder=self.image_path ,
                label_df=self.train_data,
                transforms=self.mnist_transforms['valid']
            )
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=8)


def cli_main():
    # The LightningCLI removes all the boilerplate associated with arguments parsing. This is purely optional.
    cli = LightningCLI(
        ImageClassifier, MNISTDataModule, seed_everything_default=42, save_config_overwrite=True, run=False
    )
    with mlflow.start_run():
        mlflow.log_artifact(os.path.abspath(__file__), 'source code')
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    mlflow.set_tracking_uri('http://prlab02.iptime.org:5000')  # set up connection
    mlflow.set_experiment('gluon_resnext101_64x4d')  # set the experiment
    mlflow.pytorch.autolog()
    cli_main()