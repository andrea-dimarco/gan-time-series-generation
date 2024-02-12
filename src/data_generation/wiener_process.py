'''
This script simulates a wiener process of chosen dimensions
 where each dimention is related to the ther by a randomly perturbed covariance matrix
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def weiner_process(N:int=1000, lower_bound:float=0.0, upper_bound:float=1.0, alpha:float=0.01):
    '''
    This function samples a Weiner process 
     and makes sure it stays withing bounds

    Args:
        - `N`: Number of samples to generate

    Returns
        - `Zt`: numpy array with the process' realizations
    '''
    assert(upper_bound >= lower_bound), "upper_bound must be greater than lower_bound"
    Zt = np.zeros(N)
    for i in range(1, N):
        Zt[i] = max( lower_bound, min( upper_bound, Zt[i-1]+alpha*np.random.normal() ))
    return Zt


def multi_dim_wiener_process(p:int=100, N:int=1000, corr=None):
    '''
    This function generates a dataset from a multi dimensional wiener process
     sampled from a randomly generated correlation matrix between the dimensions. 

    Args:
        - `p`: Data dimension
        - `N`: Number of samples to generate

    Returns:
        - `M`: matrix with the p-dimensional samples obtained by linear combination between the p weiner processes and the correlation matrix
    '''
    brownian_motions = np.zeros((N,p))
    for i in range(p):
        brownian_motions[:,i] += weiner_process(N)

    if corr is not None:
        assert(corr.shape[0] == corr.shape[1]), "The provided correlation matrix has mismatching shapes."
        assert(corr.shape[0] == p), "The provided correlation matrix has the wrong shape."
    else:
        corr = get_rnd_corr_matrix(p)

    # every dimention is a linear combination between the other dimensions
    M = np.einsum('kj,ik -> ij', corr, brownian_motions/2)

    return M


def get_rnd_corr_matrix(p:int):
    '''
    Generate randomly a (p,p) symmetric matrix representing the processes correlations with eachother
     corr[i][j] will represent how much process [i] is related with process [j]
     the matric does NOT have all 1s on the diagonal to facilitate the GAN's job, since this way
     the processes will be more "influenced" by the others.

    Args:
        - `p`: Data dimension

    Returns:
        - `corr`: matrix with the processes correlations
    '''
    corr = np.zeros((p,p))


    for i in range(p):
        last_upper_bound = 1.0
        for j in range(p-1):
            a = np.random.uniform(0.0, last_upper_bound)
            if corr[i][j] == 0:
                corr[i][j] = a
                corr[j][i] = 1-a
            last_upper_bound -= corr[i][j]
        corr[i][p-1] = last_upper_bound
    
    return corr


def save_wiener_process(p:int=100, N:int=1000, file_path:str="./generated_stream.csv", show_plot:bool=False):
    '''
    Save the generated samples as a csv file.

    Arguments:
        - `p`: data dimension
        - `N`: number of samples to generate
        - `file_path`: csv file path were to save the stream
    '''
    # Generate stream
    samples = multi_dim_wiener_process(p=p, N=N)

    # Save it
    df = pd.DataFrame(samples)
    df.to_csv(file_path, index=False, header=False)

    if show_plot:
        plot_process(samples)

