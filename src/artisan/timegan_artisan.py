# Absolutely necessary
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple

# Utilities
import dataset_handling as dh
import utilities as ut
from hyperparameters import Config

# Data Generation
from data_generation.wiener_process import save_wiener_process
from data_generation.sine_process import save_sine_process
from data_generation.iid_sequence_generator import save_iid_sequence, save_cov_sequence

# Modules
import artisan.losses as losses
from modules.regressor_cell import RegCell
from modules.classifier_cell import ClassCell

# Functions
def generate_data(datasets_folder="./datasets/"
                  ) -> Tuple[str, str]:
    hparams = Config()

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        # Generate and store the dataset as requested
        dataset_path = f"{datasets_folder}{hparams.dataset_name}_generated_stream.csv"
        if hparams.dataset_name == 'sine':
            save_sine_process(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
        elif hparams.dataset_name == 'wien':
            save_wiener_process(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
            print("\n")
        elif hparams.dataset_name == 'iid':
            save_iid_sequence(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
        elif hparams.dataset_name == 'cov':
            save_cov_sequence(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
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

        dh.train_test_split(X=np.loadtxt(dataset_path, delimiter=",", dtype=np.float32),
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
    
    return train_dataset_path, val_dataset_path


# Setup
do_training   = True
do_validation = True
load_modules  = False
do_testing    = False

hparams = Config()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# modules
Gen = RegCell(input_size=hparams.noise_dim,
              hidden_size=hparams.gen_hidden_dim,
              output_size=hparams.latent_space_dim,
              num_layers=hparams.gen_num_layers,
              module_type=hparams.gen_module_type
              )
Emb = RegCell(input_size=hparams.data_dim,
              hidden_size=hparams.emb_hidden_dim,
              output_size=hparams.latent_space_dim,
              num_layers=hparams.emb_num_layers,
              module_type=hparams.emb_module_type
              )
Rec = RegCell(input_size=hparams.latent_space_dim,
              hidden_size=hparams.rec_hidden_dim,
              output_size=hparams.data_dim,
              num_layers=hparams.rec_num_layers,
              module_type=hparams.rec_module_type
              )
Sup = RegCell(input_size=hparams.latent_space_dim,
              hidden_size=hparams.sup_hidden_dim,
              output_size=hparams.latent_space_dim,
              num_layers=hparams.sup_num_layers,
              module_type=hparams.sup_module_type
              )
Dis = ClassCell(input_size=hparams.latent_space_dim,
                hidden_size=hparams.dis_hidden_dim,
                num_classes=1,
                num_layers=hparams.dis_num_layers,
                module_type=hparams.dis_module_type
                )

# optimizers
E_optim = torch.optim.Adam(
    Emb.parameters(recurse=True), lr=hparams.lr, betas=(hparams.b1, hparams.b2)
    )
D_optim = torch.optim.Adam(
    Dis.parameters(recurse=True), lr=hparams.lr, betas=(hparams.b1, hparams.b2)
    )
G_optim = torch.optim.Adam(
    Gen.parameters(recurse=True), lr=hparams.lr, betas=(hparams.b1, hparams.b2)
    )
S_optim = torch.optim.Adam(
    Sup.parameters(recurse=True), lr=hparams.lr, betas=(hparams.b1, hparams.b2)
    )
R_optim = torch.optim.Adam(
    Rec.parameters(recurse=True), lr=hparams.lr, betas=(hparams.b1, hparams.b2)
    )


# losses
rec_loss = torch.nn.MSELoss()
dis_loss = torch.nn.BCELoss()


# generate data
ds_folder_path = "./datasets/"
train_dataset_path, val_dataset_path = generate_data(datasets_folder=ds_folder_path)


# data loaders
train_loader = DataLoader(
    dh.RealDataset(
        file_path=train_dataset_path,
        seq_len=hparams.seq_len
    ),
    batch_size=hparams.batch_size,
    shuffle=True,
    pin_memory=True
)
 

if do_training:
    n_epochs = hparams.n_epochs
    for epoch in range(n_epochs):
        # Modules Mode
        Gen.train()
        Emb.train()
        Rec.train()
        Dis.train()
        Sup.train()

        for idx, batch in enumerate(train_loader):
            # Process the batch
            X_batch, Z_batch = batch
        
            '''
            # Discriminator Loss
            D_loss = losses.D_loss(X=X_batch, Z=Z_batch,
                                   Emb=Emb,   Gen=Gen,
                                   Sup=Sup,   Dis=Dis,
                                   discrimination_loss=dis_loss)
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()
            

            # Generator Loss
            G_loss = losses.G_loss(X=X_batch, Z=Z_batch,
                                   Gen=Gen,   Sup=Sup,
                                   Rec=Rec,   Dis=Dis,
                                   discrimination_loss=dis_loss,
                                   reconstruction_loss=rec_loss)
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()
            '''
            
            # Supervisor Loss
            S_loss = losses.S_loss(X=X_batch, Z=Z_batch,
                                   Emb=Emb,   Gen=Gen,
                                   Sup=Sup,
                                   reconstruction_loss=rec_loss)
            S_optim.zero_grad()
            S_loss.backward()
            S_optim.step()


            # Embedder Loss
            E_loss = losses.E_loss(X=X_batch,
                                   Emb=Emb,   Sup=Sup,
                                   Rec=Rec,
                                   reconstruction_loss=rec_loss)
            E_optim.zero_grad()
            E_loss.backward()
            E_optim.step()


            # Recovery Loss
            R_loss = losses.R_loss(X=X_batch,
                                   Emb=Emb,   Rec=Rec,
                                   reconstruction_loss=rec_loss)
            R_optim.zero_grad()
            R_loss.backward()
            R_optim.step()


            # Save an image of the progress
            if True: 
                with torch.no_grad():
                    '''
                    ut.compare_sequences(real=X_batch[0],
                                        fake=Rec(Gen(Z_batch))[0],
                                        save_img=True,
                                        show_graph=False,
                                        fake_label="Generated",
                                        img_idx=(idx + (epoch*len(train_loader))),
                                        img_name="synth",
                                        folder_path="./training_pics/"
                                        )
                    '''
                    ut.compare_sequences(real=X_batch[0],
                                        fake=Rec(Emb(X_batch))[0],
                                        save_img=True,
                                        show_graph=False,
                                        fake_label="Reconstructed",
                                        img_idx=(idx + (epoch*len(train_loader))),
                                        img_name="rec",
                                        folder_path="./training_pics/"
                                        )
    
        # Validation
        if epoch % 100 == 0 and do_validation:
            pass
            # model.eval()
            # with torch.no_grad():
            #     y_pred = model(X_train)
            #     train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            #     y_pred = model(X_test)
            #     test_rmse = np.sqrt(loss_fn(y_pred, y_test))
            # print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
        else:
            print(f"Epoch: {epoch}/{n_epochs} done.")


