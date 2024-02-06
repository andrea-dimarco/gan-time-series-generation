import os


def pca_offline(nominal_dataset:str, folder:str="./anomaly_detection"
) -> None:
    '''
    This function performs the offline phase of the PCA-based anomaly detection model.

    Arguments:
        - `nominal_dataset`: path to the csv file with the real samples
        - `folder`: path to the folder with the anomaly detector executable
    '''
    command = f"{folder}/anomaly_detector n 1.0 1.0 y y {nominal_dataset}"
    os.system(command)


def pca_online(dataset:str, folder:str="./anomaly_detection",
               h:float=5, alpha:float=0.4
) -> float:
    '''
    This function will call the PCA-based anomaly detection model
     and returns the percentage of anomalies found.

    Arguments:
        - `dataset`: path to the csv file with the samples to analyze
        - `folder`: path to the folder with the anomaly detector executable
        - `h`: model's tolerance to anomalies
        - `alpha`: model's sensibility to anomalies

    Returns:
        - `output`: percentage of anomalies found, as a float in [0,1]
    '''
    command = f"{folder}/anomaly_detector n {h} {alpha} n y {dataset}"

    # Run simulation
    output = float(os.popen(command).readlines()[0])

    return output


def gem_offline(nominal_dataset:str, folder:str="./anomaly_detection"):
    '''
    This function will call the GEM-based anomaly detection model
     and returns the percentage of anomalies found.

    Arguments:
        - `dataset`: path to the csv file with the samples to analyze
        - `folder`: path to the folder with the anomaly detector executable
        - `h`: model's tolerance to anomalies
        - `alpha`: model's sensibility to anomalies

    Returns:
        - `output`: percentage of anomalies found, as a float in [0,1]
    '''
    command = f"{folder}/anomaly_detector y 1.0 1.0 y y {nominal_dataset}"
    os.system(command)


def gem_online(anomalous_dataset:str, folder:str="./anomaly_detection",
               h:float=5, alpha:float=0.4
) -> float:
    '''
    This function will call the PCA model and returns the loss value
    '''
    command = f"{folder}/anomaly_detector y {h} {alpha} n y {anomalous_dataset}"

    # Run simulation
    output = float(os.popen(command).readlines()[0])

    return output