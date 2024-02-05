'''
Fancy class to hold data
'''

from dataclasses import dataclass

@dataclass
class Config:

    train_file_path = "./datasets/training.csv"
    test_file_path = "./datasets/testing.csv"

    n_epochs: int = 5 # number of epochs of training
    decay_epoch: int = 0 # epoch from which to start lr decay

    batch_size: int = 32 # . . . . Amount of samples in each batch
    lr: float = 0.0002 # adam: learning rate
    b1: float = 0.5  # . adam: decay of first order momentum of gradient
    b2: float = 0.999 #. adam: decay of first order momentum of gradient

    lambda_cyc: float = 10.0  # cycle loss weight
    lambda_id: float = 5.0  # identity loss weight

    n_cpu: int = 12  # number of cpu threads to use for the dataloaders

    log_images: int = 25 # . . . . . . . number of images to logg

    alpha: float = 1.0 # . . . . . . . . Regularization coefficient 

    ## Data loading
    dataset_name: str = 'wein' #. . . . . which dataset to use
                               #. . . . . . wein: runs a number or weiner processes 
                               #. . . . . . . . . . with random mutual correlations
                               #. . . . . . sine: runs independent sine processes wih random phases
                               #. . . . . . iid: samples iids from a multivariate
                               #. . . . . . . . . . with random covariance matrix
                               #. . . . . . real: gets the samples from a csv file
    num_samples: int = 10**4 #. . . . . . Number of samples to generate (if any)
    data_dim: int = 6 # . . . . . . . . . Dimension of one generated sample (if any)
    seq_len: int = 20 # . . . . . . . . . Length of the input sequences

    # Network dimensions
    latent_space_dim: int = 2 # . . . . . Dimension of the latent space sample
    noise_dim: int = 1 #. . . . . . . . . Dimension of the noise to feed the generator

    emb_hidden_dim: int = 4 # . . . . . . Dimension of the hidden layers for the embedder
    gen_hidden_dim: int = 2 # . . . . . . Dimension of the hidden layers for the generator
    rec_hidden_dim: int = 4 # . . . . . . Dimension of the hidden layers for the recovery
    dis_hidden_dim: int = 3 # . . . . . . Dimension of the hidden layers for the discriminator

    dis_alpha: float = 0.2 #. . . . . . . Parameter for the discriminator's LeakyReLU (currently unused)

    gen_num_layers: int = 2 # . . . . . . Number of layers for the generator
    dis_num_layers: int = 2 # . . . . . . Number of layers for the discriminator
    emb_num_layers: int = 2 # . . . . . . Number of layers for the embedder
    rec_num_layers: int = 2 # . . . . . . Number of layers for the recovery
    sup_num_layers: int = 2 # . . . . . . Number of layers for the supervisor

    # Can be 'gru', 'lstm' or 'lstmLN
    gen_module_type: str = 'gru' #. . . . Module type for the generator
    dis_module_type: str = 'gru' #. . . . Module type for the discriminator
    emb_module_type: str = 'gru' #. . . . Module type for the embedder
    rec_module_type: str = 'gru' #. . . . Module type for the recovery
    sup_module_type: str = 'gru' #. . . . Module type for the supervisor


    metric_iteration: int = 5 # . . . . . Number of iteration for each metric