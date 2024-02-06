'''
This is the data module for the model
'''
from typing import Optional, Any, Callable, Tuple#, Dict
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd

import numpy as np

from data_generation import iid_sequence_generator, sine_process, wiener_process
from hyperparamets import Config


class SequenceDataset(Dataset):
    def __init__(self,
                 seq_type: str,
                 p: int,
                 N: int,
                 seq_len: int,
                 transform: Optional[Callable] = None,
                 ) -> None:
        '''
        Generate a dataset of sequences sampled from the selected process.

        Arguments:
            - `seq_type`:
            - `p`: dimension of one sample
            - `N`: number of SAMPLES (not sequences) to generate
            - `seq_len`: length of the sequence to extract from the data stream
            - `transform`: optional transformation to be done on the data 
        '''
        super().__init__()

        assert(seq_type in ['wein','sine','iid'])

        # generate sequence
        if seq_type == 'wein':
            xy = wiener_process.get_weiner_process(p=p, N=N)
        elif seq_type == 'sine':
            xy = sine_process.get_sine_process(p=p, N=N)
        else:
            xy = iid_sequence_generator.get_iid_sequence(p=p, N=N)

        # initialize parameters
        self.N: int = N
        self.p: int = p
        self.seq_len: int = seq_len
        self.n_seq: int = int(self.N / seq_len)
        self.transform: Optional[Callable] = transform

        # transform data
        scaler = MinMaxScaler(feature_range=(0,1)) # preserves the data distribution
        scaler.fit(xy)
        self.x = torch.from_numpy(
            scaler.transform(xy)
            .reshape(-1, self.seq_len, self.p)
            ).type(torch.float32)
        
        # generate noise
        self.noise_dim = Config().noise_dim
        self.z = torch.from_numpy(
            iid_sequence_generator.get_iid_sequence(p=self.noise_dim, N=self.N)
            .reshape(-1, self.seq_len, self.noise_dim)
            ).type(torch.float32)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.z[index]

    def __len__(self) -> int:
        return self.n_seq
    
    def get_all_sequences(self):
        return self.x
    
    def get_whole_stream(self):
        return self.x.reshape(self.N, self.p)
    
    def get_all_noise_sequences(self):
        return self.z
    
    def get_whole_noise_stream(self):
        return self.z.reshape(self.N, self.noise_dim)
    

class RealDataset(Dataset):
    def __init__(self,
                 file_path: Path,
                 seq_len: int,
                 transform: Optional[Callable] = None,
                 verbose: bool = True
                 ) -> None:
        '''
        Load the dataset from a given file containing the data stream.

        Arguments:
            - `file_path`: the path of the file containing the data stream
            - `seq_len`: length of the sequence to extract from the data stream
            - `transform`: optional transformation to be done on the data
        '''
        super().__init__()

        xy = np.loadtxt(file_path, delimiter=",", dtype=np.float32)

        # initialize parameters
        self.n_samples: int = xy.shape[0]
        self.p: int = xy.shape[1]
        self.seq_len: int = seq_len
        self.n_seq: int = int(self.n_samples / seq_len)
        self.transform: Optional[Callable] = transform

        # transform data
        scaler = MinMaxScaler(feature_range=(0,1)) # preserves the data distribution
        scaler.fit(xy)
        self.x = torch.from_numpy(
            scaler.transform(xy)
            .reshape(-1, self.seq_len, self.p)
            ).type(torch.float32)
        
        # generate noise
        self.noise_dim = Config().noise_dim
        self.z = torch.from_numpy(
            iid_sequence_generator.get_iid_sequence(p=self.noise_dim, N=self.n_samples)
            .reshape(-1, self.seq_len, self.noise_dim)
            ).type(torch.float32)

        if verbose:
            print(f"Loaded dataset with {self.n_samples} of dimension {self.p}, resulted in {self.n_seq} sequences.")

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.z[index]

    def __len__(self) -> int:
        return self.n_seq
    
    def get_all_sequences(self):
        return self.x
    
    def get_whole_stream(self):
        return self.x.reshape(self.n_samples, self.p)
    
    def get_all_noise_sequences(self):
        return self.z
    
    def get_whole_noise_stream(self):
        return self.z.reshape(self.n_samples, self.noise_dim)


def train_test_split(X: torch.Tensor, split: float=0.7, folder_path: str="./datasets/", train_file_name: str="training", test_file_name: str="testing"):
    '''
    This function takes a tensor and saves it as two different csv files according to the given split parameter.

    Arguments:
    - `X`: the tensor containing the data, dimensions must be ( num_samples, sample_dim )
    - `split`: the perchentage of samples to keep for training
    - `folder_path`: relative path to the folder where the .csv files will be stored
    - `train_file_name`: name of the .csv file that will contain the training set
    - `test_file_name`: name of the .csv file that will contain the testing set
    '''
    assert(split > 0 and split < 1)
    delimiter = int( X.size()[0] * split )

    # Train
    dataset_path = folder_path + train_file_name + ".csv"
    df = pd.DataFrame(X[:delimiter,:])
    df.to_csv(dataset_path, index=False, header=False)

    # Test
    dataset_path = folder_path + test_file_name + ".csv"
    df = pd.DataFrame(X[delimiter:,:])
    df.to_csv(dataset_path, index=False, header=False)

