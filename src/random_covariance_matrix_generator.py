'''
This script generates a dataset of i.i.d. realizations <-- this is very important
 sampled from a process with a randomly perturbed covariance matrix. 
'''

import numpy as np
import pandas as pd


def rnd_cov_gen(folder_path="./", dataset_name="random_covariance", p=100, N=10000, mean=0.0, variance=1.0, noise = 0.2):
    '''
    This function generates a dataset of i.i.d. realizations
    sampled from a process with a randomly perturbed covariance matrix. 

    Args:
        - folder_path: Folder where to save the dataset
        - dataset_name: Name of the dataset file
        - p: Data dimension
        - N: Number of samples to generate
        - mean: Mean of the data to be sampled
        - variance: Variance of the data to be sampled
        - noise: Intensity of the perturbation on the covariance matrix
    '''
    # Generate the covariance matrix
    mu = np.ones(p) * mean
    noise_matrix = np.random.uniform(-noise, noise, size=(p,p))
    noise_matrix = (noise_matrix + noise_matrix.T) / 2 # must be semidefinite positive
    np.fill_diagonal(noise_matrix,0) 
    noisy_cov = np.eye(p)*variance + noise_matrix

    # Generate & Save the dataset
    dataset_path = folder_path + dataset_name + ".csv"
    df = pd.DataFrame(np.random.multivariate_normal(mean=mu, cov=noisy_cov, size=N))
    df.to_csv(dataset_path, index=False, header=False)
