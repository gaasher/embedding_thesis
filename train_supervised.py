import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
from models.patchTST import PatchTST
from datasets import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Pred


class patchTSTDataloader(pl.LightningDataModule):
    def __init__(self, path, batch_size, shuffle=True, num_workers=0, seq_len=336, target_len=96):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.target_len = target_len


        self.train_dataset = Dataset_Custom(root_path='./datasets/electricity/', flag="train", size=(seq_len, 0, target_len), 
                                            features="M", data_path='electricity.csv', freq='h')
        self.val_dataset = Dataset_Custom(root_path='./datasets/electricity/', flag="val", size=(seq_len, 0, target_len), 
                                            features="M", data_path='electricity.csv', freq='h')
        self.test_dataset = Dataset_Custom(root_path='./datasets/electricity/', flag="test", size=(seq_len, 0, target_len), 
                                            features="M", data_path='electricity.csv', freq='h')

        self.train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def train_dataloader(self):
        return self.train_loader


    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        loader1 = self.test_loader
                    
        return loader1
    
class patchTST(pl.LightningModule):
    def __init__(self, seq_len, num_channels, embed_dim, heads, depth, target_seq_size, patch_len=8, dropout=0.0, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.heads = heads
        self.depth = depth
        self.target_seq_size = target_seq_size
        self.patch_len = patch_len
        self.dropout = dropout
        self.lr = lr

        self.model = PatchTST(seq_len, num_channels, embed_dim, heads, depth, target_seq_size, patch_len, dropout)

        # loss functions
        self.criterion = nn.MSELoss()
        self.mae = nn.L1Loss()   

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = self.mae(y_hat, y)

        self.log("train_mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = self.mae(y_hat, y)

        self.log("val_mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = self.mae(y_hat, y)

        self.log("test_mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_mae": mae}


    def configure_optimizers(self) :
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            div_factor=25,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }



if __name__ == '__main__':
    pl.seed_everything(40)

    config = {
        "seq_len": 336,
        "num_channels": 321,
        "embed_dim": 64,
        "heads":2,
        "depth": 2,
        "target_seq_size": 96,
        "patch_len": 8,
        "dropout": 0.0,
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 4,
        "num_workers": 0,
        "checkpoint_path": None,
    }


    # Initialize wandb
    wandb_logger = WandbLogger(
        project="timeseries_embeds",
        config=config,
        log_model=False,
        mode="offline",
    )
    config = wandb_logger.experiment.config
    model_name = wandb_logger.experiment.name
    wandb_logger.experiment.log_code(".")

    # Load dataset
    dataset = patchTSTDataloader(
        path="./datasets/electricity/",
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        seq_len=config["seq_len"],
        target_len=config["target_seq_size"],
    )

    # Initialize model
    model = patchTST(
        seq_len=config["seq_len"],
        num_channels=config["num_channels"],
        embed_dim=config["embed_dim"],
        heads=config["heads"],
        depth=config["depth"],
        target_seq_size=config["target_seq_size"],
        patch_len=config["patch_len"],
        dropout=config["dropout"],
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
         dirpath='trained_models', filename=f"{model_name}_best", monitor="val_loss", mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)



    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="cpu",
        #devices=[0],
        #strategy="ddp_find_unused_parameters_true",
        precision='32',
        sync_batchnorm=True,
        # use_distributed_sampler=True,
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=1.,
        logger=wandb_logger,
        # accumulate_grad_batches=2,
    )

    if config['checkpoint_path'] is not None:
        print("Loading pre-trained checkpoint")
        trainer.fit(model, dataset, ckpt_path=config['checkpoint_path'])
    else:
        trainer.fit(model, dataset)
    
    # Evaluate on test dataset
    test_results = trainer.test(model, datamodule=dataset)
    print(f"Test Results: {test_results}")

    # Finish wandb run
    wandb_logger.experiment.finish()