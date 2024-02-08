from anomaly_detection import anomaly_detector_api as AD_API
import dataset_handling as dh

import pandas as pd
import numpy as np
import os


from timegan_model import TimeGAN
from hyperparamets import Config
from data_generation import sine_process, wiener_process, iid_sequence_generator
from numpy import loadtxt, float32

## SETUP
# Parameters
hparams = Config()
datasets_folder = "./datasets/"


# create datasets
if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
    # Generate and store the dataset as requested
    dataset_path = f"{datasets_folder}{hparams.dataset_name}_generated_stream.csv"
    if hparams.dataset_name == 'sine':
        sine_process.save_sine_process(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
    elif hparams.dataset_name == 'wien':
        wiener_process.save_wiener_process(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
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
    test_dataset_path  = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

    dh.train_test_split(X=loadtxt(dataset_path, delimiter=",", dtype=float32),
                    split=hparams.train_test_split,
                    train_file_name=train_dataset_path,
                    test_file_name=test_dataset_path    
                    )
    print(f"The {hparams.dataset_name} dataset has been split successfully into:\n\t- {train_dataset_path}\n\t- {test_dataset_path}")
elif hparams.dataset_name == 'real':
    train_dataset_path = datasets_folder + hparams.train_file_name
    val_dataset_path   = datasets_folder + hparams.test_file_name
else:
    raise ValueError("Dataset not supported.")


# get dataset name
if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
    test_dataset_path  = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

elif hparams.dataset_name == 'real':
    test_dataset_path  = hparams.test_file_path
else:
    raise ValueError("Dataset not supported.")


# Instantiate the model
timegan = TimeGAN(hparams=hparams,
                    train_file_path=train_dataset_path,
                    val_file_path=test_dataset_path
                    )





## REAL TESTING LOOP
if hparams.operating_system != 'windows':
    test_dataset = dh.RealDataset(
                    file_path=test_dataset_path,
                    seq_len=hparams.seq_len
                )

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
        # Get the synthetic sequence
        Z_seq = Z.reshape(1, hparams.seq_len, hparams.noise_dim)
        X_seq = timegan(Z_seq).detach().reshape(hparams.seq_len, hparams.data_dim)

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

else:
    print("The PCA-based Anomaly Detector related tests are not currently supported for this operating system.")