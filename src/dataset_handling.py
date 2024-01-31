'''
This is the data module for the model
'''

from typing import Optional, Any, Callable#, Dict
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch

import numpy as np

from data_generation import iid_sequence_generator, sine_process, wiener_process


class SequenceDataset(Dataset):
    def __init__(self,
                 seq_type: str,
                 p: int,
                 N: int,
                 seq_len: int,
                 transform: Optional[Callable] = None,
                 ) -> None:
        super().__init__()

        assert(seq_type in ['wein','sine','iids'])

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
        scaler = MinMaxScaler(feature_range=(-1,1)) # preserves the data distribution
        scaler.fit(xy)
        self.x = torch.from_numpy(
            scaler.transform(xy)
            .reshape(-1, self.seq_len, self.p)
            ).type(torch.float32)

    def __getitem__(self, index) -> Any:
        sample = self.x[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_seq
    
    def get_all_sequences(self):
        return self.x
    
    def get_whole_stream(self):
        return self.x.reshape(self.N, self.p)
    

class RealDataset(Dataset):
    def __init__(self,
                 pathA: Path,
                 seq_len: int,
                 transform: Optional[Callable] = None,
                 ) -> None:
        super().__init__()

        xy = np.loadtxt(pathA, delimiter=",", dtype=np.float32)

        self.n_samples: int = xy.shape[0]
        self.p: int = xy.shape[1]
        self.seq_len: int = seq_len
        self.n_seq: int = int(self.n_samples / seq_len)
        self.transform: Optional[Callable] = transform

        self.x = torch.from_numpy(xy)

    def __getitem__(self, index) -> Any:
        sample = self.x[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_seq
    

#dataset = SequenceDataset(p=3, N=100000, seq_type='wein', seq_len=25)
#print(dataset[0])
#wiener_process.plot_processes(dataset.get_whole_stream())