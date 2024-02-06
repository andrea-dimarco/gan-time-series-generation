
from timegan_model import TimeGAN
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import cuda
from pytorch_lightning import Trainer
import wandb
from hyperparamets import Config

def train():
    # Parameters
    hparams = Config()
    accelerator = "cuda" if cuda.is_available() else "cpu"

    # Instantiate the model
    timegan = TimeGAN(hparams=hparams,
                        train_file_path=hparams.train_file_path,
                        test_file_path=hparams.test_file_path
                        )

    # Define the logger -> https://www.wandb.com/articles/pytorch-lightning-with-weights-biases.
    wandb_logger = WandbLogger(project="TimeGAN PyTorch (2024)", log_model=True)

    wandb_logger.experiment.watch(timegan, log='all', log_freq=100)

    # Define the trainer
    trainer = Trainer(logger=wandb_logger,
                    max_epochs=hparams.n_epochs,
                    accelerator=accelerator,
                    val_check_interval=0.25
                    )

    # Start the training
    trainer.fit(timegan)

    # Log the trained model
    trainer.save_checkpoint('timegan.pth')
    wandb.save('timegan.pth')

    return timegan

### Testing Area
'''
import numpy as np
np.random.seed(0)
train()
'''