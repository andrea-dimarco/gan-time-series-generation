
import torch
import wandb
from numpy import loadtxt, float32
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

import utilities as ut
import dataset_handling as dh
from timegan_model import TimeGAN
from hyperparameters import Config

import warnings
warnings.filterwarnings("ignore")


def train(datasets_folder="./datasets/"):
    '''
    Train the TimeGAN model
    '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    # Parameters
    hparams = Config()

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
        val_dataset_path  = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

    elif hparams.dataset_name == 'real':
        train_dataset_path = hparams.train_file_path
        val_dataset_path  = hparams.test_file_path
    else:
        raise ValueError("Dataset not supported.")

    # Instantiate the model
    timegan = TimeGAN(hparams=hparams,
                    train_file_path=train_dataset_path,
                    val_file_path=val_dataset_path
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
                    val_check_interval=1.0,
                    callbacks=[early_stop]
                    )

    # Start the training
    trainer.fit(timegan)

    # Log the trained model
    #wandb.save('timegan-wandb.pth') # use this for wandb online
    torch.save(timegan.state_dict(), f"timegan-{hparams.dataset_name}.pth") # use this when logging progress offline
    return timegan


def validate_model(model:TimeGAN, datasets_folder:str="./datasets/", limit:int=1) -> None:
    '''
    Plot the synthetic sequences and compare the original sequences with ther reconstructions

    Arguments:
        - model: the TimeGAN model, preferrably already trained.
        - datasets_folder: folder containing the datasets
        - limit: amout of samples to run validation on, if 0 then it will be the whole dataset
    '''
    hparams = Config()
    test_dataset_path = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

    # Test Dataset
    test_dataset = dh.RealDataset(
                    file_path=test_dataset_path,
                    seq_len=hparams.seq_len
                )

    horizon = limit if limit>0 else len(test_dataset)

    # run tests
    for idx, (X, Z) in enumerate(test_dataset):
        if idx < horizon:
            # Generate the synthetic sequence
            Z_seq = Z.reshape(1, hparams.seq_len, hparams.noise_dim).to(model.dev)
            X_synth = model(Z_seq).cpu().detach().reshape(hparams.seq_len, hparams.data_dim)
            # save result
            ut.plot_process(samples=X_synth, save_picture=True, img_idx=idx, show_plot=False)

            # Reconstruct the real sequence
            X_seq = X.reshape(1, hparams.seq_len, hparams.data_dim).to(model.dev)
            X_tilde = model.cycle(X_seq).cpu().detach().reshape(hparams.seq_len, hparams.data_dim)
            # save result
            ut.compare_sequences(real=X, fake=X_tilde, save_img=True, img_idx=idx, show_graph=False)
        else:
            break


# # # # # # # # #
# Training Area #
# # # # # # # # #
datasets_folder = "./datasets/"
ut.generate_data(datasets_folder)
ut.set_seed(seed=1337)
train(datasets_folder=datasets_folder)

