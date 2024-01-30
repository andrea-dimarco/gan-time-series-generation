'''
This is the data module for the model
'''

from typing import Optional, Any, Callable#, Dict
from torch.utils.data import Dataset
from pathlib import Path
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

        if seq_type == 'wein':
            xy = wiener_process.get_weiner_process(p=p, N=N)
        elif seq_type == 'sine':
            xy = sine_process.get_sine_process(p=p, N=N)
        else:
            xy = iid_sequence_generator.get_iid_sequence(p=p, N=N)

        self.n_samples: int = xy.shape[0]
        self.p: int = p
        self.seq_len: int = seq_len
        self.n_seq: int = int(self.n_samples / seq_len)
        self.transform: Optional[Callable] = transform

        self.x = torch.from_numpy(xy.reshape(-1, self.seq_len, self.p))

    def __getitem__(self, index) -> Any:
        sample = self.x[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_seq
    

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