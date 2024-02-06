from torch.nn import Module
from random import uniform, randint
from torch import Tensor, cat
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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
    

def plot_processes(samples, labels=None, save_picture=False, show_plot=True):
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
            plt.savefig("plot.png")
        if show_plot:
            plt.show()


def compare_sequences(real: Tensor, fake: Tensor,
                      real_label="Real sequence", fake_label="Fake Sequence",
                      show_graph=False, save_img=False, img_idx=0):
    '''
    Plots two graphs with the two sequences.

    Arguments:
        - `real`: the first sequence with dimension [seq_len, data_dim]
        - `fake`: the second sequence with dimension [seq_len, data_dim]
        - `show_graph`: whether to display the graph or not
        - `save_img`: whether to save the image of the graph or not
        - `img_idx`: the id of the graph that will be used to name the file

    Returns:
        - numpy matrix with the pixel values for the image
    '''
    fig, (ax0, ax1) = plt.subplots(2, 1, layout='constrained')
    ax0.set_xlabel('Time-Steps')

    for i in range(real.shape[1]):
        ax0.plot(real[:,i])
    ax0.set_ylabel(real_label)

    for i in range(fake.shape[1]):
        ax1.plot(fake[:,i])
    ax1.set_ylabel(fake_label)

    if show_graph:
        plt.show()
    if save_img:
        plt.savefig(f"double-plot{img_idx}.png")


    # return picture as array
    canvas = fig.canvas
    plt.close()
    canvas.draw()  # Draw the canvas, cache the renderer

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
    
