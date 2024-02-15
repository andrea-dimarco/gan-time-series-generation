import torch
import random
import numpy as np
import numpy as np
import pandas as pd
from typing import List
import matplotlib as mpl
from torch.nn import Module
from torch import Tensor, cat
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from random import uniform, randint
from sklearn.decomposition import PCA

# Just a function to count the number of parameters
def count_parameters(model: Module) -> int:
  """ Counts the number of trainable parameters of a module

  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ReplayBuffer:
    def __init__(self, max_size: int = 50) -> None:
        '''
        Image buffer to increase the robustness of the generator.

        Once it is full, i.e. it contains max_size images, each image in a given batch
        is swapped with probability p=0.5 with another one contained in the buffer.
        '''
        assert (
            max_size > 0
        ), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: Tensor) -> Tensor:
        '''
        Fill the buffer with each element in data.
        If the buffer is full, with p=0.5 swap each element in data with
        another in the buffer.

        Arguments:
            - data: tensor with shape [batch, ...]

        Returns:
            - tensor with shape [batch, ...]
        '''
        to_return = []

        for i in range(data.shape[0]):
            element = data[[i], ...]

            if len(self.data) < self.max_size:
                self.data.append(element)

            elif uniform(0, 1) > 0.5:
                i = randint(0, self.max_size - 1)
                self.data[i], element = element, self.data[i]

            to_return.append(element)

        return cat(to_return)
    

def plot_process(samples, labels:List[str]|None=None,
                 save_picture=False, show_plot=True,
                 img_idx=0, img_name:str="plot",
                 folder_path:str="./") -> None:
    '''
    Plots all the dimensions of the generated dataset.
    '''
    if save_picture or show_plot:
        for i in range(samples.shape[1]):
            if labels is not None:
                plt.plot(samples[:,i], label=labels[i])
            else:
                plt.plot(samples[:,i])

        # giving a title to my graph 
        if labels is not None:
            plt.legend()
        
        # function to show the plot 
        if save_picture:
            plt.savefig(f"{folder_path}{img_name}-{img_idx}.png")
        if show_plot:
            plt.show()
        plt.clf()


def compare_sequences(real: Tensor, fake: Tensor,
                      real_label:str="Real sequence", fake_label:str="Fake Sequence",
                      show_graph:bool=False, save_img:bool=False,
                      img_idx:int=0, img_name:str="plot", folder_path:str="./"):
    '''
    Plots two graphs with the two sequences.

    Arguments:
        - `real`: the first sequence with dimension [seq_len, data_dim]
        - `fake`: the second sequence with dimension [seq_len, data_dim]
        - `show_graph`: whether to display the graph or not
        - `save_img`: whether to save the image of the graph or not
        - `img_idx`: the id of the graph that will be used to name the file
        - `img_name`: the file name of the graph that will be used to name the file
        - `folder_path`: path to the folder where to save the image

    Returns:
        - numpy matrix with the pixel values for the image
    '''
    mpl.use('Agg')
    fig, (ax0, ax1) = plt.subplots(2, 1, layout='constrained')
    ax0.set_xlabel('Time-Steps')

    for i in range(real.shape[1]):
        ax0.plot(real.cpu()[:,i])
    ax0.set_ylabel(real_label)

    for i in range(fake.shape[1]):
        ax1.plot(fake.cpu()[:,i])
    ax1.set_ylabel(fake_label)

    if show_graph:
        plt.show()
    if save_img:
        plt.savefig(f"{folder_path}{img_name}-{img_idx}.png")


    # return picture as array
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    plt.clf()
    plt.close('all')

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    return image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    

class LambdaLR():
    def __init__(self, n_epochs: int, decay_start_epoch: int) -> None:
        '''
        Linearly decay the leraning rate to 0, starting from `decay_start_epoch`
        to the final epoch.

        Arguments:
            - `n_epochs`: total number of epochs
            - `decay_start_epoch`: epoch in which the learning rate starts to decay
        '''
        assert(n_epochs > decay_start_epoch), "Decay must start BEFORE the training session ends!"
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int) -> float:
        '''
        One step of lr decay:
        - if `epoch < self.decay_start_epoch` it doesn't change the learning rate.
        - Otherwise, it linearly decay the lr to reach zero

        Arguments:
            - `epoch`: current epoch
        
        Returns:
            - Learning rate multiplicative factor
        '''
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


def PCA_visualization(ori_data:torch.Tensor, generated_data:torch.Tensor, 
                               show_plot:bool=False, save_plot:bool=True,
                               folder_path:str="./"
                               ) -> None:
    """
    Using PCA for generated and original data visualization
     on both the original and synthetic datasets (flattening the temporal dimension).
     This visualizes how closely the distribution of generated samples
     resembles that of the original in 2-dimensional space

    Args:
    - `ori_data`: original data (num_sequences, seq_len, data_dim)
    - `generated_data`: generated synthetic data (num_sequences, seq_len, data_dim)
    - `show_plot`: display the plot
    - `save_plot`: save the .png of the plot
    - `folder_path`: where to save the file
    """  
    if show_plot or save_plot:
        # Data preprocessing
        N, data_dim = ori_data.size()  
        p = data_dim

        prep_data = ori_data.reshape(N,p).numpy()
        prep_data_hat = generated_data.reshape(N,p).numpy()
        
        # Visualization parameter        
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        red = ["red" for i in range(N)]
        blue = ["blue" for i in range(N)]
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c=red, alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c=blue, alpha = 0.2, label = "Synthetic")

        ax.legend()  
        plt.title('Distribution comparison')
        if save_plot:
            plt.savefig(f"{folder_path}pca-visual.png")
        if show_plot:
            plt.show()
        plt.clf()


def set_seed(seed=0) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    _ = pl.seed_everything(seed)


def save_timeseries(samples, folder_path:str="./", file_name="timeseries.csv") -> None:
    '''
    Save the samples as a csv file.
    '''
    # Save it
    df = pd.DataFrame(samples)
    df.to_csv(f"{folder_path}{file_name}", index=False, header=False)