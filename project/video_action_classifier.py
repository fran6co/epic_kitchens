import pathlib
from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl
import timesformer.utils.checkpoint as cu
import torch
import torch.utils.data
from timesformer.models.vit import TimeSformer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from project.datasets import Epickitchens


class EpickitchensDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        num_workers=8,
        batch_size=32,
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        num_frames=8,
        num_clips=1,
        jitter_scales=(256, 320),
        crop_size=224,
        num_spatial_crops=3,
    ):
        super().__init__()
        self.data_path = data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.jitter_scales = jitter_scales
        self.crop_size = crop_size
        self.num_spatial_crops = num_spatial_crops

    def setup(self, stage: Optional[str] = None):
        # split dataset
        if stage in (None, "fit"):
            self.dataset_train = Epickitchens(
                self.data_path,
                mode="train",
                mean=self.mean,
                std=self.std,
                num_frames=self.num_frames,
                num_clips=self.num_clips,
                jitter_scales=self.jitter_scales,
                crop_size=self.crop_size,
                num_spatial_crops=self.num_spatial_crops,
            )
            self.dataset_val = Epickitchens(
                self.data_path,
                mode="val",
                mean=self.mean,
                std=self.std,
                num_frames=self.num_frames,
                num_clips=self.num_clips,
                jitter_scales=self.jitter_scales,
                crop_size=self.crop_size,
                num_spatial_crops=self.num_spatial_crops,
            )
        if stage == "test":
            self.dataset_test = Epickitchens(
                self.data_path,
                mode="test",
                mean=self.mean,
                std=self.std,
                num_frames=self.num_frames,
                num_clips=self.num_clips,
                jitter_scales=self.jitter_scales,
                crop_size=self.crop_size,
                num_spatial_crops=self.num_spatial_crops,
            )
        if stage == "predict":
            self.dataset_predict = Epickitchens(
                self.data_path,
                mode="test",
                mean=self.mean,
                std=self.std,
                num_frames=self.num_frames,
                num_clips=self.num_clips,
                jitter_scales=self.jitter_scales,
                crop_size=self.crop_size,
                num_spatial_crops=self.num_spatial_crops,
            )

    def train_dataloader(self):
        mnist_train = DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )
        return mnist_train

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_predict,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )


class LitTimesformer(pl.LightningModule):
    def __init__(
        self, num_classes, num_frames=8, freeze_layers=False, learning_rate=0.005
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TimeSformer(num_classes=num_classes, num_frames=num_frames)

        # Freeze all layers, this helps to avoid overfitting
        if freeze_layers:
            for module in self.model.modules():
                for param in module.parameters():
                    param.requires_grad = False
            # We want to train the head only
            for param in self.model.model.head.parameters():
                param.requires_grad = True

        self.train_accuracy = Accuracy(num_classes=num_classes)
        self.valid_accuracy = Accuracy(num_classes=num_classes)
        self.test_accuracy = Accuracy(num_classes=num_classes)

    def forward(self, x):
        # B T H W C -> B C T H W.
        x = x.permute(0, 4, 1, 2, 3)

        # RGB to BGR
        x = x[:, [2, 1, 0], :, :, :]
        x = self.model(x)
        x = F.softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        inputs, labels, _, meta = batch
        preds = self.model(inputs)
        loss = F.cross_entropy(preds, labels)
        self.train_accuracy.update(preds, labels)

        self.log("train_loss", loss, on_epoch=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _, meta = batch
        preds = self.model(inputs)
        loss = F.cross_entropy(preds, labels)
        self.valid_accuracy.update(preds, labels)

        self.log("valid_loss", loss)
        self.log("valid_acc", self.valid_accuracy)

    def test_step(self, batch, batch_idx):
        inputs, labels, _, meta = batch
        preds = self.model(inputs)
        loss = F.cross_entropy(preds, labels)
        self.test_accuracy.update(preds, labels)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_accuracy)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def train(
    data_path, batch_size, num_classes, checkpoint, learning_rate, freeze_layers, args
):
    pl.seed_everything(1234)

    # ------------
    # data
    # ------------
    dataset = EpickitchensDataModule(data_path, batch_size=batch_size)

    # ------------
    # model
    # ------------
    # Freezing layers doesn't help with over fitting, let's disable it
    model = LitTimesformer(
        freeze_layers=freeze_layers,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    # Use a starting point for fine tuning
    if checkpoint:
        cu.load_checkpoint(checkpoint, model.model)

    # ------------
    # training
    # ------------
    # Get the model with the best accuracy in the validation set
    on_best_accuracy = pl.callbacks.ModelCheckpoint(
        monitor="valid_acc",
        mode="max",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[on_best_accuracy],
        gpus=min(1, torch.cuda.device_count()),
    )

    trainer.fit(model, datamodule=dataset)

    # ------------
    # testing
    # ------------
    result = trainer.test(ckpt_path="best", datamodule=dataset)
    print(result)

    # ----------------------------------
    # onnx
    # ----------------------------------
    input_sample = torch.randn((1, 8, 224, 224, 3))
    onnx_path = pathlib.Path(trainer.checkpoint_callback.best_model_path).with_suffix(
        ".onnx"
    )
    model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    dynamic_axes = {"images": {0: "batch"}, "scores": {0: "batch"}}
    model.to_onnx(
        str(onnx_path),
        input_sample,
        export_params=True,
        opset_version=13,
        input_names=["images"],
        output_names=["scores"],
        dynamic_axes=dynamic_axes,
    )


if __name__ == "__main__":
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--freeze_layers", default=False, type=bool)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train(
        args.data_path,
        args.batch_size,
        args.num_classes,
        args.checkpoint,
        args.learning_rate,
        args.freeze_layers,
        args,
    )
