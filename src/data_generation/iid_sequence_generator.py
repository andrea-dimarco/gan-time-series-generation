'''
This script generates a dataset of i.i.d. realizations <-- this is very important
 sampled from a process with a randomly perturbed covariance matrix. 
'''

import numpy as np
import pandas as pd


def save_cov_sequence(file_path="./random_covariance.csv", p=100, N=10000, mean=0.0, variance=1.0, noise = 0.2):
    '''
    This function generates a dataset of i.i.d. realizations
    sampled from a process with a randomly perturbed covariance matrix. 

    Args:
        - `file_path`: Name of the dataset file
        - `p`: Data dimension
        - `N`: Number of samples to generate
        - `mean`: Mean of the data to be sampled
        - `variance`: Variance of the data to be sampled
        - `noise`: Radius of the perturbation on the covariance matrix
    '''
    # Generate the covariance matrix
    mu = np.ones(p) * mean
    noise_matrix = np.random.uniform(-noise, noise, size=(p,p))
    noise_matrix = (noise_matrix + noise_matrix.T) / 2 # must be semidefinite positive
    np.fill_diagonal(noise_matrix,0) 
    noisy_cov = np.eye(p)*variance + noise_matrix

    # Generate & Save the dataset
    df = pd.DataFrame(np.random.multivariate_normal(mean=mu, cov=noisy_cov, size=N))
    df.to_csv(file_path, index=False, header=False)


def get_cov_sequence(p=100, N=1000, mean=0.0, variance=1.0, noise=0.5):
    '''
    Returns a sequence of randomly sampled numbers with mean `mean`
     and randomly perturbed covariance matrix.

    Arguments:
        - `p`: dimension of one sample
        - `N`: number of samples to take
        - `mean`: mean of the data, the same for every channel
        - `variance`: variance of the data, the same for every channel
        - `noise`: radius of the perturbation to the covariance matrix

    Returns:
        - The data stream as a numpy matrix with dimension (N, p)
    '''
    # Generate the covariance matrix
    mu = np.ones(p)*mean
    noise_matrix = np.random.uniform(-noise, noise, size=(p,p))
    noise_matrix = (noise_matrix + noise_matrix.T) / 2 # must be semidefinite positive
    np.fill_diagonal(noise_matrix,0) 
    noisy_cov = np.eye(p)*variance + noise_matrix

    return np.random.multivariate_normal(mean=mu, cov=noisy_cov, size=N)


def get_iid_sequence(p=100, N=1000, mean=0.0, variance=1.0):
    '''
    Generate and a sequence of iid variables.

    Arguments:
        - `p`: sample dimension
        - `N`: number of samples
        - `mean`: mean of the samples
        - `variance`: variance of the samples
    '''
    # Generate the covariance matrix
    mu = np.ones(p)*mean
    variance_matrix = np.eye(p)*variance
    return np.random.multivariate_normal(mean=mu, cov=variance_matrix, size=N)


def save_iid_sequence(p=100, N=1000, mean=0.0, variance=1.0, file_path="./iid.csv"):
    '''
    Generate and save in a csv file the iid sequence.

    Arguments:
        - `p`: sample dimension
        - `N`: number of samples
        - `mean`: mean of the samples
        - `variance`: variance of the samples
        - `file_path`: path of the file where to save the sequence
    '''
    # Generate the covariance matrix
    mu = np.ones(p)*mean
    variance_matrix = np.eye(p)*variance

    # Generate & Save the dataset
    df = pd.DataFrame(np.random.multivariate_normal(mean=mu, cov=variance_matrix, size=N))
    df.to_csv(file_path, index=False, header=False)