
from timegan_model import TimeGAN
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch import cuda
from pytorch_lightning import Trainer
import wandb
from hyperparamets import Config

import torch

from data_generation import iid_sequence_generator, sine_process, wiener_process
from dataset_handling import train_test_split
from numpy import loadtxt, float32

def generate_data(folder="./datasets"):
    hparams = Config()

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        # Generate and store the dataset as requested
        dataset_path = f"{folder}/{hparams.dataset_name}_generated_stream.csv"
        if hparams.dataset_name == 'sine':
            sine_process.save_sine_process(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
        elif hparams.dataset_name == 'wien':
            wiener_process.save_wiener_process(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
            print("\n")
        elif hparams.dataset_name == 'iid':
            iid_sequence_generator.save_iid_sequence(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
        elif hparams.dataset_name == 'cov':
            iid_sequence_generator.save_cov_sequence(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
        else:
            raise ValueError
        print(f"The {hparams.dataset_name} dataset has been succesfully created and stored into:\n\t- {dataset_path}")
    elif hparams.dataset_name == 'real':
        pass
    else:
        raise ValueError("Dataset not supported.")
    

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        train_dataset_path = f"{folder}/{hparams.dataset_name}_training.csv"
        test_dataset_path  = f"{folder}/{hparams.dataset_name}_testing.csv"

        train_test_split(X=loadtxt(dataset_path, delimiter=",", dtype=float32),
                        split=hparams.train_test_split,
                        train_file_name=train_dataset_path,
                        test_file_name=test_dataset_path    
                        )
        print(f"The {hparams.dataset_name} dataset has been split successfully into:\n\t- {train_dataset_path}\n\t- {test_dataset_path}")
    elif hparams.dataset_name == 'real':
        train_dataset_path = hparams.train_file_path
        test_dataset_path  = hparams.test_file_path
    else:
        raise ValueError("Dataset not supported.")


def train(folder="./datasets"):

    torch.multiprocessing.set_sharing_strategy('file_system')

    # Parameters
    hparams = Config()
    accelerator = "cuda" if cuda.is_available() else "cpu"
    folder = "./datasets"

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        train_dataset_path = f"{folder}/{hparams.dataset_name}_training.csv"
        test_dataset_path  = f"{folder}/{hparams.dataset_name}_testing.csv"

    elif hparams.dataset_name == 'real':
        train_dataset_path = hparams.train_file_path
        test_dataset_path  = hparams.test_file_path
    else:
        raise ValueError("Dataset not supported.")

    # Instantiate the model
    timegan = TimeGAN(hparams=hparams,
                    train_file_path=train_dataset_path,
                    test_file_path=test_dataset_path
                    )

    # Define the logger -> https://www.wandb.com/articles/pytorch-lightning-with-weights-biases.
    wandb_logger = WandbLogger(project="TimeGAN PyTorch (2024)", log_model=True)

    wandb_logger.experiment.watch(timegan, log='all', log_freq=100)

    # Define the trainer
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.early_stop_patience,
        strict=False,
        verbose=False
    )
    trainer = Trainer(logger=wandb_logger,
                    max_epochs=hparams.n_epochs,
                    accelerator=accelerator,
                    val_check_interval=0.25
                    #callbacks=[early_stop]
                    )

    # Start the training
    trainer.fit(timegan)

    # Log the trained model
    trainer.save_checkpoint('timegan.pth')
    wandb.save('timegan.pth')

    return timegan


### Testing Area
folder = "./datasets"
generate_data(folder)
train(folder)

