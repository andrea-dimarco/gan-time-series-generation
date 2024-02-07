import os
import numpy as np
import pandas as pd


def transpose_dataset(file_path:str, new_file_path:str) -> None:
    '''
    Loads a csv file and saves it with flipped dimensions.

    Arguments:
        - file_path: path to the csv file to flip
        - new_file_path: path to the new csv file to be created, can be the same as above
    '''
    X = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
    X = X.transpose()

    df = pd.DataFrame(X)
    df.to_csv(new_file_path, index=False, header=False)


def pca_offline(nominal_dataset:str, folder:str="./anomaly_detection/"
) -> None:
    '''
    This function performs the offline phase of the PCA-based anomaly detection model.

    Arguments:
        - `nominal_dataset`: path to the csv file with the real samples
        - `folder`: path to the folder with the anomaly detector executable
    '''
    command = f"{folder}anomaly_detector n 1.0 1.0 y y {nominal_dataset}"
    os.system(command)


def pca_online(file_path:str, folder:str="./anomaly_detection/",
               h:float=5, alpha:float=0.4
) -> float:
    '''
    This function will call the PCA-based anomaly detection model
     and returns the percentage of anomalies found.

    Arguments:
        - `file_path`: path to the csv file with the samples to analyze
        - `folder`: path to the folder with the anomaly detector executable
        - `h`: model's tolerance to anomalies
        - `alpha`: model's sensibility to anomalies

    Returns:
        - `output`: percentage of anomalies found, as a float in [0,1]
    '''
    command = f"{folder}anomaly_detector n {h} {alpha} n y {file_path}"

    # Run simulation
    output = float(os.popen(command).readlines()[0])

    return output


def gem_offline(file_path:str, folder:str="./anomaly_detection/"):
    '''
    This function will call the GEM-based anomaly detection model
     and returns the percentage of anomalies found.

    Arguments:
        - `file_path`: path to the csv file with the samples to analyze
        - `folder`: path to the folder with the anomaly detector executable
        - `h`: model's tolerance to anomalies
        - `alpha`: model's sensibility to anomalies

    Returns:
        - `output`: percentage of anomalies found, as a float in [0,1]
    '''
    command = f"{folder}anomaly_detector y 1.0 1.0 y y {file_path}"
    os.system(command)


def gem_online(file_path:str, folder:str="./anomaly_detection/",
               h:float=5, alpha:float=0.4
) -> float:
    '''
    This function will call the GEM-based anomaly detection model
     and returns the percentage of anomalies found.

    Arguments:
        - `file_path`: path to the csv file with the samples to analyze
        - `folder`: path to the folder with the anomaly detector executable
        - `h`: model's tolerance to anomalies
        - `alpha`: model's sensibility to anomalies

    Returns:
        - `output`: percentage of anomalies found, as a float in [0,1]
    '''
    command = f"{folder}anomaly_detector y {h} {alpha} n y {file_path}"

    # Run simulation
    output = float(os.popen(command).readlines()[0])

    return output


def cleanup_files(folder="./") -> None:
    '''
    Deletes the files created by the AD models
    
    Arguments:
        - folder: the folder where the files are located.
    '''
    os.system(f"rm {folder}gem_baseline_distances.csv")
    os.system(f"rm {folder}gem_parameters.csv")
    os.system(f"rm {folder}gem_S1.csv")
    os.system(f"rm {folder}pca_baseline_distances.csv")
    os.system(f"rm {folder}pca_mean_vector.csv")
    os.system(f"rm {folder}pca_parameters.csv")
    os.system(f"rm {folder}pca_res_proj.csv")