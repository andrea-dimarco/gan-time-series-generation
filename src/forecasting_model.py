

# Libraries
from typing import Sequence, Dict, Tuple, Union, Mapping

from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch import optim

import wandb
import pytorch_lightning as pl

# My modules
import dataset_handling as dh
import utilities as ut
from hyperparameters import Config

'''
Simple Sensor Forecasting
'''
class SSF(pl.LightningModule):
    def __init__(self,
        hparams: Union[Dict, Config],
        train_file_path: Path,
        val_file_path: Path
    ) -> None:
        '''
        The Single Sensor Forecasting model.

        Arguments:
            - `hparams`: dictionary that contains all the hyperparameters
            - `train_file_path`: Path to the folder that contains the training stream
            - `val_file_path`: Path to the file that contains the testing stream
        '''
        super().__init__()
        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Mapping) else hparams)

        # Dataset paths
        self.train_file_path = train_file_path
        self.val_file_path   = val_file_path

        # loss criteria
        self.rec_loss = torch.nn.L1Loss()

        # Expected shapes 
        self.data_dim = hparams.data_dim
        self.seq_len = hparams.seq_len

        # Initialize Modules
        # input = ( batch_size, seq_len, data_dim )
        self.lstm = nn.LSTM(input_size=hparams.data_dim,
                            hidden_size=hparams.forecaster_hidden,
                            num_layers=hparams.forecaster_epochs,
                            batch_first=True
                            )
        self.fc = nn.Linear(in_features=hparams.forecaster_hidden,
                            out_features=hparams.data_dim
                            )

        # init weights
        self.fc.apply(self.init_weights)
        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.lstm.__getattr__(p))
        
        # Forward pass cache to avoid re-doing some computation
        #self.fake = None

        # It avoids wandb logging when lighting does a sanity check on the validation
        self.is_sanity = True

        # For the end of the validation step
        self.validation_step_output = []


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, seq_len, data_dim]

        Returns:
            - the predicted sequences [batch, seq_len, data_dim]
        '''
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


    def train_dataloader(self) -> DataLoader:
        '''
        Create the train set DataLoader

        Returns:
            - `train_loader`: the train set DataLoader
        '''
        train_loader = DataLoader(
            dh.RealDataset(
                file_path=self.train_file_path,
                seq_len=self.seq_len
            ),
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            pin_memory=True
        )
        return train_loader
        

    def val_dataloader(self) -> DataLoader:
        '''
        Create the validation set DataLoader.

        It is deterministic.
        It does not shuffle and does not use random transformation on each image.
        
        Returns:
            - `val_loader`: the validation set DataLoader
        '''
        val_loader = DataLoader(
            dh.RealDataset(
                file_path=self.val_file_path,
                seq_len=self.seq_len
            ),
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            pin_memory=True
        )
        return val_loader


    def configure_optimizers(self
    ) -> Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer, optim.Optimizer, optim.Optimizer]:
        '''
        Instantiate the optimizers and schedulers.

        We have five optimizers (and relative schedulers):

        - `optim`: optimzer for the Embedder
        
        - `lr_scheduler`: learning rate scheduler for the Embedder

        Each scheduler implements a linear decay to 0 after `self.hparams.decay_epoch`

        Returns:
            - the optimizers
            - the schedulers for the optimizers
        '''

        # Optimizers
        optim = torch.optim.Adam(self.parameters(recurse=True),
                                 lr=self.hparams["lr"],
                                 betas=(self.hparams["b1"], self.hparams["b2"])
                                 )

        return optim
   

    def training_step(self,
                      batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        '''
        Implements a single training step

        The parameter `optimizer_idx` identifies with optimizer "called" this training step,
        this way we can change the behaviour of the training depending on which optimizer
        is currently performing the optimization

        Arguments:
            - `batch`: current training batch
            - `batch_idx`: the index of the batch being processed

        Returns:
            - the total loss for the current training step, together with other information for the
                  logging and possibly the progress bar
        '''
        # Process the batch
        x, y = batch
        pred = self(x)
        loss = self.rec_loss(pred, y)

        # Log results
        loss_dict = { "loss": loss }
        self.log_dict(loss_dict)

        return loss


    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]:
        '''
        Implements a single validation step

        In each validation step some translation examples are produced and a
        validation loss that uses the cycle consistency is computed

        Arguments:
            - `batch`: the current validation batch

        Returns:
            - the loss and example images
        '''
        # Process the batch
        x, y = batch
        pred = self(x)
        loss = self.rec_loss(pred, y)

        # visualize result
        image = self.get_image_examples(y[0], self(x)[0], fake_label="Predicted Samples")

        # Validation loss
        val_out = { "val_loss": loss, "image": image }
        self.log("val_loss", loss)
        self.validation_step_output.append(val_out)
        return val_out


    def get_image_examples(self,
                           real: torch.Tensor, fake: torch.Tensor,
                           fake_label:str="Synthetic"):
        '''
        Given real and "fake" translated images, produce a nice coupled images to log

        Arguments:
            - `real`: the real sequence with shape [seq_len, data_dim]
            - `fake`: the fake sequence with shape [seq_len, data_dim]

        Returns:
            - A sequence of wandb.Image to log and visualize the performance
        '''
        example_images = []
        couple = ut.compare_sequences(real=real, fake=fake, save_img=False, show_graph=False, fake_label=fake_label)

        example_images.append(
            wandb.Image(couple, mode="RGB")
        )
        return example_images


    def on_validation_epoch_end(self
    ) -> Dict[str, torch.Tensor]:
        '''
        Implements the behaviouir at the end of a validation epoch

        Currently it gathers all the produced examples and log them to wandb,
        limiting the logged examples to `hparams["log_images"]`.

        Then computes the mean of the losses and returns it.
        Updates the progress bar label with this loss.

        Arguments:
            - outputs: a sequence that aggregates all the outputs of the validation steps

        Returns:
            - The aggregated validation loss and information to update the progress bar
        '''
        images = []

        for x in self.validation_step_output:
            images.extend(x["image"])

        images = images[: self.hparams["log_images"]]

        if not self.is_sanity:  # ignore if it not a real validation epoch. The first one is not.
            self.logger.experiment.log(
                {f"images": images },
                step=self.global_step,
            )
        self.is_sanity = False

        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_output]).mean()
        self.log_dict({"val_loss": avg_loss})
        self.validation_step_output = []
        return {"val_loss": avg_loss}
    

    def init_weights(self, m):
        '''
        Initialized the weights of the nn.Sequential block
        '''
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)