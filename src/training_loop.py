
from timegan_model import TimeGAN
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
import wandb
from hyperparamets import Config

import torch

from data_generation import iid_sequence_generator, sine_process, wiener_process
from dataset_handling import train_test_split
from numpy import loadtxt, float32

import dataset_handling as dh
import utilities as ut

import warnings
warnings.filterwarnings("ignore")


def generate_data(datasets_folder="./datasets/"):
    hparams = Config()

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        # Generate and store the dataset as requested
        dataset_path = f"{datasets_folder}{hparams.dataset_name}_generated_stream.csv"
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
        train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
        val_dataset_path   = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

        train_test_split(X=loadtxt(dataset_path, delimiter=",", dtype=float32),
                        split=hparams.train_test_split,
                        train_file_name=train_dataset_path,
                        test_file_name=val_dataset_path    
                        )
        print(f"The {hparams.dataset_name} dataset has been split successfully into:\n\t- {train_dataset_path}\n\t- {val_dataset_path}")
    elif hparams.dataset_name == 'real':
        train_dataset_path = datasets_folder + hparams.train_file_name
        val_dataset_path   = datasets_folder + hparams.test_file_name
    else:
        raise ValueError("Dataset not supported.")


def train(datasets_folder="./datasets/"):

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
                    val_file_path=val_dataset_path,
                    device=device,
                    plot_losses=False
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
                    val_check_interval=1.0
                    )

    # Start the training
    trainer.fit(timegan)

    #timegan.plot()

    torch.save(timegan.state_dict(), "timegan-model.pth")

    # Log the trained model
    trainer.save_checkpoint('timegan.pth')
    wandb.save('timegan.pth')

    return timegan


def validate_model(model:TimeGAN, datasets_folder:str="./datasets/", limit:int=1) -> None:
    '''
    Run some tests after the traning has ended.

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

    


### Testing Area
datasets_folder = "./datasets/"
generate_data(datasets_folder)
train(datasets_folder)

