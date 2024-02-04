
from timegan_model import TimeGAN
from hyperparamets import Config
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import cuda
from pytorch_lightning import Trainer
import wandb

# Parameters
hparams = Config()
accelerator = "cuda" if cuda.is_available() else "cpu"

# Instantiate the model
timegan = TimeGAN(hparams=hparams,
                    train_file_path=hparams.train_file_path,
                    test_file_path=hparams.test_file_path
                    )

# Define the logger
# https://www.wandb.com/articles/pytorch-lightning-with-weights-biases.
wandb_logger = WandbLogger(project="CycleGAN Tutorial 2021", log_model=True)

## Currently it does not log the model weights, there is a bug in wandb and/or lightning.
wandb_logger.experiment.watch(timegan, log='all', log_freq=100)

# Define the trainer
trainer = Trainer(logger=wandb_logger,
                  max_epochs=hparams.n_epochs,
                  accelerator=accelerator,
                  limit_val_batches=.2,
                  val_check_interval=0.25
                  )

# Start the training
trainer.fit(timegan)

# Log the trained model
trainer.save_checkpoint('timegan.pth')
wandb.save('timegan.pth')