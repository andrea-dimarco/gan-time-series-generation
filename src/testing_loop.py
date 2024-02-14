from anomaly_detection import anomaly_detector_api as AD_API
import dataset_handling as dh

from typing import Tuple

import pandas as pd
import numpy as np
import torch
import os

# My stuff
from timegan_model import TimeGAN
import utilities as ut
from hyperparameters import Config
from data_generation import sine_process, wiener_process, iid_sequence_generator
from numpy import loadtxt, float32

import warnings
warnings.filterwarnings("ignore")


def generate_data(datasets_folder="./datasets/"
                  ) -> Tuple[str, str]:
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

        dh.train_test_split(X=loadtxt(dataset_path, delimiter=",", dtype=float32),
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


def AD_tests(model:TimeGAN, test_dataset:dh.RealDataset
             ) -> Tuple[float, float]:
    hparams = Config()
    if hparams.operating_system != 'windows':
        # metrics
        FAR_tot = 0.0 # False Alarm Rate (on nominal data)
        TAR_tot = 0.0 # True Alarm Rate (on synthetic data)

        # train anomaly detectors
        AD_folder = "./src/anomaly_detection/"
        AD_offline_path = f"{datasets_folder}{hparams.dataset_name}_testing_AD_offline.csv"
        AD_online_path  = f"{datasets_folder}{hparams.dataset_name}_testing_AD_online.csv"

        # since the model is trained on normalized data
        # we must train the AD on normalized data as well
        df = pd.DataFrame( np.transpose(test_dataset.get_whole_stream().numpy()) )
        df.to_csv(AD_offline_path, index=False, header=False)

        # train AD
        AD_API.pca_offline(AD_offline_path, folder=AD_folder)

        # Get AD accuracy on the real data
        anomalies_found = AD_API.pca_online(file_path=AD_offline_path, folder=AD_folder, h=hparams.h, alpha=hparams.alpha)
        FAR_tot += anomalies_found

        # free memory
        os.system(f"rm {AD_offline_path}")

        # run tests
        for idx, (X, Z) in enumerate(test_dataset):
            with torch.no_grad():
                # Get the synthetic sequence
                Z_seq = Z.reshape(1, hparams.seq_len, hparams.noise_dim)
                X_seq = model(Z_seq).reshape(hparams.seq_len, hparams.data_dim)

                # save synthetic sequence to a file
                X_seq = np.transpose(X_seq.numpy())
                df = pd.DataFrame(X_seq)
                df.to_csv(AD_online_path, index=False, header=False)

                # run simulation
                anomalies_found = AD_API.pca_online(file_path=AD_online_path, folder=AD_folder, h=hparams.h, alpha=hparams.alpha)
                TAR_tot += anomalies_found
        TAR_tot /= len(test_dataset)

        # free memory
        os.system(f"rm {AD_online_path}")
        AD_API.cleanup_files()

        # print results
        print(f"Anomalies found on real data: {round(FAR_tot*100, 2)}%")
        print(f"Anomalies found on fake data: {round(TAR_tot*100, 2)}%")

        return FAR_tot, TAR_tot
    else:
        print("The PCA-based Anomaly Detector related tests are not currently supported for this operating system.")


def recovery_seq_test(model:TimeGAN, test_dataset:dh.RealDataset,
                  limit:int=0, frequency:int=10,
                  save_pictures:bool=True, folder_path:str="./test_results/recovery_tests/"
                  ) -> float:
    '''
    Ask the model to embedd and recover real sequences from test_dataset.

    Arguments:
        - `model`: TimwGAN model to test
        - `test_dataset`: test dataset
        - `limit`: test sample number horizon
        - `frequency`: how often to save the picture
        - `save_pictures`: if to save the pictures or not
        - `folder_path`: where to save the pictures

    Returns:
        - average loss found
    '''
    horizon = min(limit, len(test_dataset)) if limit>0 else len(test_dataset)
    small_seq = min(limit, test_dataset.seq_len) if limit>0 else test_dataset.seq_len
    pic_id = 0
    loss = 0
    test_name = "recovery"
    print(f"Test type:{test_name} has started.")
    for idx, (X, Z) in enumerate(test_dataset):
        if idx >= horizon:
            break
        # embedd & reconstruct the sequence
        with torch.no_grad():
            X_seq = X.reshape(1, hparams.seq_len, hparams.data_dim)[:,:small_seq]
            X_rec = model.cycle(X_seq).reshape(small_seq, hparams.data_dim)
            loss += model.R_loss(X_seq).item()

        # save a picture every frequency steps
        if (idx % frequency) == 0 and save_pictures:
            ut.compare_sequences(real=X[:small_seq], fake=X_rec,
                                save_img=True, show_graph=False,
                                img_idx=pic_id, img_name=test_name,
                                real_label="Real", fake_label="Reconstructed",
                                folder_path=folder_path)
            pic_id += 1
            print(f" Saved picture {pic_id}/{int(horizon/frequency)}.")
    loss /= len(test_dataset)
    print(f"Avg {test_name} loss: {loss}")
    return loss


def generate_seq_test(model:TimeGAN, test_dataset:dh.RealDataset,
                    limit:int=0, frequency:int=10,
                    save_pictures:bool=True, folder_path:str="./test_results/generation_tests/"
                    ) -> float:
    '''
    Ask the model to embedd and recover real sequences from test_dataset.

    Arguments:
        - `model`: TimeGAN model to test
        - `test_dataset`: test dataset
        - `limit`: test sample number horizon
        - `frequency`: how often to save the picture
        - `save_pictures`: if to save the pictures or not
        - `folder_path`: where to save the pictures

    Returns:
        - average loss found
    '''
    horizon = min(limit, len(test_dataset)) if limit>0 else len(test_dataset)
    small_seq = min(limit, test_dataset.seq_len) if limit>0 else test_dataset.seq_len
    pic_id = 0
    loss = 0
    test_name = "generation"
    print(f"Test type: {test_name} has started.")
    for idx, (X, Z) in enumerate(test_dataset):
        if idx >= horizon:
            break
        with torch.no_grad():
            # Get the synthetic sequence
            Z_seq = Z.reshape(1, hparams.seq_len, hparams.noise_dim)[:,:small_seq]
            X_seq = X.reshape(1, hparams.seq_len, hparams.data_dim)[:,:small_seq]
            X_hat = model(Z_seq).reshape(small_seq, hparams.data_dim)

        loss += model.G_loss(X_seq, Z_seq).item()

        # save a picture every frequency steps
        if (idx % frequency) == 0 and save_pictures:
            ut.compare_sequences(real=X[:small_seq], fake=X_hat,
                                save_img=True, show_graph=False,
                                img_idx=pic_id, img_name=test_name,
                                real_label="Real", fake_label="Synthetic",
                                folder_path=folder_path)
            pic_id += 1
            print(f" Saved picture {pic_id}/{int(horizon/frequency)}.")
    loss /= horizon
    print(f"Avg {test_name} loss: {loss}")
    return loss


def discriminative_seq_test(model:TimeGAN, test_dataset:dh.RealDataset,
                        limit:int=0
                        ) -> float:
    '''
    Ask if the sequences are real or not.

    Arguments:
        - `model`: TimeGAN model to test
        - `test_dataset`: test dataset
        - `limit`: test sample number horizon

    Returns:
        - average accuracy
    '''
    horizon = limit if limit>0 else len(test_dataset)
    good_preds = 0
    loss = 0
    test_name = "discrimination"
    for idx, (X, Z) in enumerate(test_dataset):
        if idx >= horizon:
            break
        # Get the sequences
        Z_seq = Z.reshape(1, hparams.seq_len, hparams.noise_dim)
        X_seq = X.reshape(1, hparams.seq_len, hparams.data_dim)

        # embedd
        H_real = model.Sup(model.Emb(X_seq))
        H_fake = model.Sup(model.Gen(Z_seq))
        
        # disrcriminate
        Y_real = 1 if model.Dis(H_real).squeeze().item()>0.5 else 0 # 1 if RIGHT 0 if WRONG
        Y_fake = 1 if model.Dis(H_fake).squeeze().item()<0.5 else 0 # 1 if RIGHT 0 if WRONG

        good_preds += Y_real + Y_fake

    loss = good_preds/(horizon*2)
    print(f"Avg discriminator accuracy: {round(loss*100, 2)}%")
    return loss


def predictive_test(model:TimeGAN, test_dataset:dh.RealDataset,
                    limit:int=0, frequency:int=10,
                    save_pictures:bool=True, folder_path:str="./test_results/generation_tests/"
                    ) -> float:
    '''
    Train a simple LSTM to predict the next steps of the generated data
     then evaluate it on the real data

    Arguments:
        - `model`: TimeGAN model to test
        - `test_dataset`: test dataset
        - `limit`: test sample number horizon
        - `frequency`: how often to save the picture
        - `save_pictures`: if to save the pictures or not
        - `folder_path`: where to save the pictures

    Returns:
        - average loss found
    '''
    pass


def generate_stream_test(model:TimeGAN, test_dataset:dh.RealDataset,
                                    limit:int=0, save_pic:bool=False,
                                    show_plot:bool=True, folder_path:str="./",
                                    compare:bool=True
) -> None:
    '''
    Generates a synthetic sequence and plots it against the real one.
    '''
    with torch.no_grad():
        horizon = limit if limit>0 else test_dataset.n_samples
        timegan.eval()
        synth = model(test_dataset.get_whole_noise_stream()[:horizon]
                        ).reshape(horizon, test_dataset.p)
        print("Synthetic stream has been generated.")
        if compare:
            ut.compare_sequences(real=test_dataset.get_whole_stream()[:horizon],
                                fake=synth,
                                real_label="Original data",
                                fake_label="Synthetic Data",
                                img_name="real-vs-synth",
                                save_img=save_pic,
                                show_graph=show_plot,
                                folder_path=folder_path
                                )
        else:
            ut.plot_process(samples=synth,
                            save_picture=save_pic,
                            show_plot=show_plot,
                            folder_path=folder_path,
                            img_name="synth"
                            )
        print("Plot done.")


def distribution_visualization(model:TimeGAN, test_dataset:dh.RealDataset,
                                    limit:int=0, save_pic:bool=False,
                                    show_plot:bool=True, folder_path:str="./test_results"
) -> None:
    '''
    Generates a synthetic sequence and plots it against the real one.
    '''
    with torch.no_grad():
        horizon = limit if limit>0 else test_dataset.n_samples
        timegan.eval()
        synth = model(test_dataset.get_whole_noise_stream()[:horizon]
                        ).reshape(horizon, test_dataset.p)
        original = test_dataset.get_whole_stream()[:horizon]
        print("Synthetic stream has been generated.")
        
        ut.PCA_visualization(ori_data=original,
                             generated_data=synth,
                             show_plot=show_plot,
                             folder_path=folder_path,
                             save_plot=save_pic
                             )
        print("Distribution visualization done.")



# # # # # # # # 
# Testing Area #
 # # # # # # # #
## SETUP
hparams = Config()
datasets_folder = "./datasets/"
train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
test_dataset_path   = f"{datasets_folder}{hparams.dataset_name}_testing.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

# Instantiate the model
timegan = TimeGAN(hparams=hparams,
                    train_file_path=train_dataset_path,
                    val_file_path=test_dataset_path
                    )
timegan.load_state_dict(torch.load("./timegan-model.pth"))
timegan.eval()
print(f"TimeGAN model loaded and ready for testing.")

# Load the dataset
test_dataset = dh.RealDataset(
                file_path=test_dataset_path,
                seq_len=hparams.seq_len
                )


## TESTING LOOP
limit = hparams.limit
frequency = hparams.pic_frequency
if True:
    
    avg_rec_loss = recovery_seq_test(model=timegan,
                                 test_dataset=test_dataset,
                                 limit=limit,
                                 frequency=frequency
                                 )

    avg_gen_loss = generate_seq_test(model=timegan,
                                   test_dataset=test_dataset,
                                   limit=limit,
                                   frequency=frequency
                                   )
    
    generate_stream_test(model=timegan,
                        test_dataset=test_dataset,
                        limit=limit,
                        folder_path="./test_results/",
                        save_pic=True,
                        show_plot=True
                        )

    distribution_visualization(model=timegan,
                               test_dataset=test_dataset,
                               limit=limit,
                               folder_path="./test_results/",
                               save_pic=True,
                               show_plot=True
                               )
    
    avg_pred_loss = predictive_test(model=timegan,
                                    test_dataset=test_dataset,
                                    limit=limit,
                                    frequency=frequency
                                    )

