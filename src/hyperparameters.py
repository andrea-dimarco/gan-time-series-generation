'''
Fancy class to hold data
'''

from dataclasses import dataclass

@dataclass
class Config:

    ## System parameters
    operating_system:str = 'windows' # . Will affect the testing results
    dataset_folder:str = "../datasets/"# Path to the datasets folder 

    ## Training parameters
    n_epochs: int = 20**1 #. . . . . . . Number of epochs of training
    early_stop_patience: int = 100 # . . Amount of epochs to wait for improvement
    decay_epoch: int = 0 # . . . . . . . Epoch from which to start lr decay

    batch_size: int = 32 # . . . . . . . Amount of samples in each batch
    lr: float = 0.02 # . . . . . . . . . adam: learning rate
    b1: float = 0.75 # . . . . . . . . . adam: decay of first order momentum of gradient
    b2: float = 0.95 # . . . . . . . . . adam: decay of second order momentum of gradient
    decay_start: float = 1.0 # . . . . . Starting decay factor for the schedulers
    decay_end: float   = 0.1 # . . . . . Ending decay factor for the scheduler

    log_images: int =  1 # . . . . . . . Number of images to log at every validation epoch


    ## Data loading
    #. . . . . . . . . . . . . . . . . . Datasets file names
    train_file_name = "training.csv"
    test_file_name  = "testing.csv"
    val_file_name   = "validating.csv"
    dataset_name: str = 'wien' # . . . . Which dataset to use
                               # . . . . . . real: gets the samples from csv files
                               # . . . . . . sine: runs independent sine processes wih random phases
                               # . . . . . . iid: samples iids from a multivariate
                               # . . . . . . cov: samples iids from a multivariate
                               # . . . . . . . . . with random covariance matrix
                               # . . . . . . wien: runs a number or wiener processes 
                               # . . . . . . . . . with random mutual correlations
    train_test_split: float = 0.8 #. . . Split between training and testing samples
    train_val_split: float  = 0.8 #. . . Split between training and validating samples
    num_samples: int = 10**5 # . . . . . Number of samples to generate (if any)
    data_dim: int =  2 # . . . . . . . . Dimension of one generated sample (if any)
    seq_len: int  = 50 # . . . . . . . . Length of the input sequences


    ## Network parameters
    latent_space_dim: int = 1 #. . . . . Dimension of the latent space sample
    noise_dim: int = 1 # . . . . . . . . Dimension of the noise to feed the generator

    emb_hidden_dim: int = 10 # . . . . . Dimension of the hidden layers for the embedder
    gen_hidden_dim: int = 10 # . . . . . Dimension of the hidden layers for the generator
    rec_hidden_dim: int = 10 # . . . . . Dimension of the hidden layers for the recovery
    dis_hidden_dim: int = 10 # . . . . . Dimension of the hidden layers for the discriminator
    sup_hidden_dim: int = 10 # . . . . . Dimension of the hidden layers for the supervisor

    gen_num_layers: int = 1 #. . . . . . Number of layers for the generator
    dis_num_layers: int = 1 #. . . . . . Number of layers for the discriminator
    emb_num_layers: int = 1 #. . . . . . Number of layers for the embedder
    rec_num_layers: int = 1 #. . . . . . Number of layers for the recovery
    sup_num_layers: int = 1 #. . . . . . Number of layers for the supervisor

    gen_module_type: str = 'gru' #. . . Module type for the generator
    dis_module_type: str = 'gru' #. . . Module type for the discriminator
    emb_module_type: str = 'gru' #. . . Module type for the embedder
    rec_module_type: str = 'gru' #. . . Module type for the recovery
    sup_module_type: str = 'gru' #. . . Module type for the supervisor
    #. . . . . . . . . . . . . . . . . . . . Can be rnn, gru lstm 


    ## Testing phase
    alpha: float = 0.1 # . . . . . . . . Parameter for the Anomaly Detector
    h: float = 10 #. . . . . . . . . . . Parameter for the Anomaly Detector
    limit:int = 0 #. . . . . . . . . . . Amount of elements to consider when running tests