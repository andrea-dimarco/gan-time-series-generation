'''
This script simulates a wiener process of chosen dimensions
 where each dimention is related to the ther by a randomly perturbed covariance matrix
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from os import system

def weiner_process(N=1000):
    '''
    This function samples a Weiner process

    Args:
        - N: Number of samples to generate
    Returns
        - Zt: numpy array with the process' realizations
    '''
    Zt = np.zeros(N)
    for i in range(1, N):
        Zt[i] = Zt[i-1] + np.random.normal()
    return Zt


def multi_dim_wiener_process(p=100, N=1000):
    '''
    This function generates a dataset from a multi dimensional wiener process
     sampled from a randomly generated correlation matrix between the dimensions. 

    Args:
        - p: Data dimension
        - N: Number of samples to generate
    Returns:
        - M: matrix with the p-dimensional samples obtained by linear combination between the p weiner processes and the correlation matrix
    '''
    brownian_motions = np.zeros((N,p))
    for i in range(p):
        brownian_motions[:,i] += weiner_process(N)#.reshape(N,1)

    corr = get_rnd_corr_matrix(p)

    # every dimention is a linear combination between the other dimensions
    M = np.einsum('kj,ik -> ij', corr, brownian_motions)

    return M


def get_rnd_corr_matrix(p):
    '''
    Generate randomly a (p,p) symmetric matrix representing the processes correlations with eachother
     corr[i][j] will represent how much process [i] is related with process [j]
     the matric does NOT have all 1s on the diagonal to facilitate the GAN's job, since this way
     the processes will be more "influenced" by the others.

    Args:
        - p: Data dimension
    Returns:
        - corr: matrix with the processes correlations
    '''
    corr = np.zeros((p,p))


    for i in range(p):
        last_upper_bound = 1.0
        for j in range(p-1):
            a = np.random.uniform(0.0, last_upper_bound)
            if corr[i][j] == 0:
                corr[i][j] = a
                corr[j][i] = a
            last_upper_bound -= corr[i][j]
        corr[i][p-1] = last_upper_bound
    return corr

def plot_processes(samples, save_picture=False, show_plot=True):
    '''
    For testing purposes ONLY!!
     plots all the dimensions of the generated dataset.
    '''
    if save_picture or show_plot:
        for i in range(samples.shape[1]):
            plt.plot(samples[:,i])

        # naming the x axis 
        plt.xlabel('time step') 
        # naming the y axis 
        plt.ylabel('Zt')
        # giving a title to my graph 
        plt.title("Wiener Process plot")
        
        # function to show the plot 
        if save_picture:
            plt.savefig("{title}-plot.png".format(title="wiener"))
        if show_plot:
            plt.show()

def save_weiner_process(p=100, N=1000, folder_path="./", dataset_name="generated_stream", show_plot=False):
    '''
    Save the generated samples as a csv file.
    '''
    # Generate stream
    samples = multi_dim_wiener_process(p=p, N=N)

    # Save it
    dataset_path = folder_path + dataset_name + ".csv"
    df = pd.DataFrame(samples)
    df.to_csv(dataset_path, index=False, header=False)

    if show_plot:
        plot_processes(samples)

def get_weiner_process(p=100,N=1000):
    '''
    Get the generated samples as a numpy matrix.
    '''
    return multi_dim_wiener_process(p=p, N=N)


save_weiner_process(p=2, N=1000)
