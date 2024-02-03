

# Libraries
from typing import Sequence, List, Dict, Tuple, Optional, Any, Union, Mapping
import itertools

from dataclasses import asdict
from pathlib import Path

from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import torch
import torchvision.transforms as transforms # this is not really needed
from torch import nn, optim

import wandb
import pytorch_lightning as pl

from torchvision import transforms

# My modules
import losses
import dataset_handling as dh
from utilities import ReplayBuffer
from utilities import compare_sequences
from lr_scheduler import LambdaLR
from hyperparamets import Config
from modules.discriminator import Discriminator
from modules.embedder import Embedder
from modules.generator import Generator
from modules.recovery import Recovery
from modules.supervisor import Supervisor

'''
This is the main model. It encapsulates all the logic into a clear and well defined framework, as defined by Lightning.

The main methods of every Lightning model are:

- `train_dataloader` and `val_dataloader`: defines the dataloader for the train and test set

- `configure_optimizers`: configure optimizers and schedulers. For each couple (optimizer, scheduler) there will be one call to `training_step` with the appropriate `optimizer_idx` to identify the optimizer.

- `training_step`: defines what happens in a single training step

- `validation_step`: defines what happens in a single validation step

- `validation_epoch_end`: receive in input an aggregation of all the output of the `validation_step`. It is useful to compute metrics and log examples.
'''
class TimeGAN(pl.LightningModule):
    def __init__(
        self,
        hparams: Union[Dict, Config],
        train_file_path: Path,
        test_file_path: Path,
    ) -> None:
        '''
        The TimeGAN model.

        Arguments:
            - `hparams`: dictionary that contains all the hyperparameters
            - `train_file_path`: Path to the folder that contains the training stream
            - `test_file_path`: Path to the file that contains the testing stream
        '''
        super().__init__()
        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Mapping) else hparams)

        # Dataset paths
        self.train_file_path = train_file_path
        self.test_file_path  = test_file_path

        # Expected shapes 
        self.data_dim = self.hparams["data_dim"]
        self.latent_space_dim = self.hparams["latent_space_dim"]
        self.noise_dim = self.hparams["noise_dim"]
        self.seq_len = self.hparams["seq_len"]

        # Initialize Modules
        self.Gen = Generator(input_size=self.noise_dim,
                            hidden_size=self.hparams["gen_hidden_dim"],
                            output_size=self.latent_space_dim,
                            num_layers=self.hparams["gen_num_layers"],
                            module_type=self.hparams["gen_module_type"]
                            )
        self.Emb = Embedder(input_size=self.data_dim,
                            hidden_size=self.hparams["emb_hidden_dim"],
                            output_size=self.latent_space_dim,
                            num_layers=self.hparams["emb_num_layers"],
                            module_type=self.hparams["emb_module_type"]
                            )
        self.Rec = Recovery(input_size=self.latent_space_dim,
                            hidden_size=self.hparams["rec_hidden_dim"],
                            output_size=self.data_dim,
                            num_layers=self.hparams["rec_num_layers"],
                            module_type=self.hparams["rec_module_type"]
                            )
        self.Sup = Supervisor(input_size=self.latent_space_dim,
                              num_layers=self.hparams["sup_num_layers"],
                              module_type=self.hparams["sup_module_type"]
                              )
        self.Dis = Discriminator(input_size=self.latent_space_dim,
                                 hidden_size=self.hparams["dis_hidden_dim"],
                                 alpha=self.hparams["dis_alpha"],
                                 num_layers=self.hparams["dis_num_layers"],
                                 module_type=self.hparams["dis_module_type"]
                                 )

        # Image buffers
        self.fake_buffer = ReplayBuffer()

        # Forward pass cache to avoid re-doing some computation
        self.fake = None

        # It avoids wandb logging when lighting does a sanity check on the validation
        self.is_sanity = True


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for this model.

        This is not used while training!

        Arguments:
            - `z`: input of the forward pass with shape [batch, seq_len, noise_dim]

        Returns:
            - the translated image with shape [batch, seq_len, data_dim]
        '''

        assert(z.size()[2] == self.noise_dim), "Invalid dimention for noise"

        # Generate a synthetic sequence in the latent space
        E_hat = self.Gen(z) # generator

        # Tune the sequence (superflous step)
        H_hat = self.Sup(E_hat) # supervisor

        # Recontrust fake data from the synthetic sequence
        X_hat = self.Rec(H_hat) # recovery

        return X_hat


    def train_dataloader(self) -> DataLoader:
        '''
        Create the train set DataLoader

        Returns:
            - `train_loader`: the train set DataLoader
        '''
        train_loader = DataLoader(
            dh.RealDataset(
                file_path=self.train_file_path,
                seq_len=self.seq_len
            ),
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        return train_loader


    def val_dataloader(self) -> DataLoader:
        '''
        Create the validation set DataLoader.

        It is deterministic.
        It does not shuffle and does not use random transformation on each image.
        
        Returns:
            - `test_loader`: the validation set DataLoader
        '''
        test_loader = DataLoader(
            dh.RealDataset(
                file_path=self.test_file_path,
                seq_len=self.seq_len
            ),
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        return test_loader


    def configure_optimizers(
        self,
    ) -> Tuple[Sequence[optim.Optimizer], Sequence[Dict[str, Any]]]:
        '''
        Instantiate the optimizers and schedulers.

        We have five optimizers (and relative schedulers):

        - `E_solver`: optimzer for the Embedder
        - `D_solver`: optimizer for the Discriminator
        - `G_solver`: optimizer for the Generator      
        - `S_solver`: optimizer for the Supervisor
        - `R_solver`: optimizer for the Recovery
        
        - `lr_scheduler_E`: learning rate scheduler for the Embedder
        - `lr_scheduler_D`: learning rate scheduler for the Discriminator
        - `lr_scheduler_G`: learning rate scheduler for the Generator
        - `lr_scheduler_S`: learning rate scheduler for the Supervisor
        - `lr_scheduler_R`: learning rate scheduler for the Recovery

        Each scheduler implements a linear decay to 0 after `self.hparams["decay_epoch"]`

        Returns:
            - the optimizers
            - the schedulers for the optimizers
        '''

        # Optimizers
        E_solver = torch.optim.Adam(
            self.Emb.parameters(), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )
        D_solver = torch.optim.Adam(
            self.Dis.parameters(), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )
        G_solver = torch.optim.Adam(
            self.Gen.parameters(), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )
        S_solver = torch.optim.Adam(
            self.Sup.parameters(), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )
        R_solver = torch.optim.Adam(
            self.Rec.parameters(), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )


        # Schedulers 
        lr_scheduler_E = torch.optim.lr_scheduler.LambdaLR(
            E_solver,
            lr_lambda=LambdaLR(self.hparams["n_epochs"], self.hparams["decay_epoch"]).step,
        )
        lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
            D_solver,
            lr_lambda=LambdaLR(self.hparams["n_epochs"], self.hparams["decay_epoch"]).step,
        )
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            G_solver,
            lr_lambda=LambdaLR(self.hparams["n_epochs"], self.hparams["decay_epoch"]).step,
        )
        lr_scheduler_S = torch.optim.lr_scheduler.LambdaLR(
            S_solver,
            lr_lambda=LambdaLR(self.hparams["n_epochs"], self.hparams["decay_epoch"]).step,
        )
        lr_scheduler_R = torch.optim.lr_scheduler.LambdaLR(
            R_solver,
            lr_lambda=LambdaLR(self.hparams["n_epochs"], self.hparams["decay_epoch"]).step,
        )

        return (
            [E_solver, D_solver, G_solver, S_solver, R_solver],
            [
                {"scheduler": lr_scheduler_E, "interval": "epoch", "frequency": 1},
                {"scheduler": lr_scheduler_D, "interval": "epoch", "frequency": 1},
                {"scheduler": lr_scheduler_G, "interval": "epoch", "frequency": 1},
                {"scheduler": lr_scheduler_S, "interval": "epoch", "frequency": 1},
                {"scheduler": lr_scheduler_R, "interval": "epoch", "frequency": 1}
            ],
        )


    def get_opt_idx(module_name: str) -> int:
        '''
        Given the module name returns the index of the optimizer.
        The names are:

            - Emb: optimzer for the Embedder
            - Dis: optimizer for the Discriminator
            - Gen: optimizer for the Generator      
            - Sup: optimizer for the Supervisor
            - Rec: optimizer for the Recovery

        Arguments:
            - `module_name`: the name of the module
        '''
        module_list = ['Emb','Dis','Gen','Sup','Rec']
        assert(module_name in module_list), "typo!!"
        return module_list.index(module_name)


    def D_loss(Y_real: torch.Tensor, Y_fake: torch.Tensor,
               Y_fake_e: torch.Tensor) -> float:
        '''
        This function computes the loss for the DISCRIMINATOR module.

        Arguments:
            - `Y_real`: Discriminator's results on the sequences of the EMBEDDINGS from the REAL data (H)
            - `Y_fake`: Discriminator's results on the sequences resulted from the SUPERVISOR with the GENERATED sequence (H_hat)
            - `Y_fake_e`: Discriminator's results on the GENERATED sequences (E_hat)

        Returns:
            - `D_loss`: float  the loss of the Discriminator module
        '''
        return losses.discrimination_loss(Y_real, Y_fake, Y_fake_e)


    def GS_loss(Y_fake: torch.Tensor, Y_fake_e: torch.Tensor,
               X: torch.Tensor, H: torch.Tensor,
               H_hat_supervise: torch.Tensor, X_hat: torch.Tensor) -> Tuple[float, float]:
        '''
        This function computes the loss for the DISCRIMINATOR module.

        Arguments:
            - `Y_fake`: Discriminator's results on the sequences resulted from the SUPERVISOR with the GENERATED sequence (H_hat)
            - `Y_fake_e`: Discriminator's results on the GENERATED sequences (E_hat)
            - `X`: The real data
            - `H`: The sequence EMBEDDED from the real data (X)
            - `H_hat_supervise`: The sequence rturned by the SUPERVISOR on the legitimate EMBEDDING (H) of the real data (X)
            - `X_hat`: the sequence obtained by asking the RECOVERY module to reconstruct a feature space sequence from a SUPERVISED (H_hat), GENERATED (E_hat) sequence from noise (Z)

        Returns:
            - `G_loss`: float with the overall loss of the GENERATOR module
            - `S_loss`: the supervised loss
        '''
        return losses.generation_loss(Y_fake, Y_fake_e, X, H, H_hat_supervise, X_hat)


    def ER_loss(X: torch.Tensor, X_tilde: torch.Tensor, S_loss: torch.float32) -> Tuple[float, float]:
        '''
        This function computes the loss for the DISCRIMINATOR module.

        Arguments:
            - `X`: The original data
            - `X_tilde`: the data reconstructed by the RECOVERY module from the EMBEDDING (H) of the original data (X)
            - `S_loss`: the supervised loss returned as the second result of generator_loss(...)

        Returns:
            - `E_loss`: float with the overall loss of the Embedder module
            - `R_loss`: float with the overall loss for the Recovery module
        '''
        return losses.reconstruction_loss(X, X_tilde, S_loss)


    def training_step(
        self, X_batch: torch.Tensor,
        Z_batch: torch.Tensor, optimizer_idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        '''
        Implements a single training step

        The parameter `optimizer_idx` identifies with optimizer "called" this training step,
        this way we can change the behaviour of the training depending on which optimizer
        is currently performing the optimization

        Arguments:
            - `X_batch`: current training batch of real sequences
            - `Z_batch`: current training batch of noise
            - `optimizer_idx`: the index of the optimizer in use, see the function `configure_optimizers`

        Returns:
            - the total loss for the current training step, together with other information for the
                  logging and possibly the progress bar
        '''
        # Process the batch
        # Embedder
        H = self.Emb(X_batch)

        # Generator
        E_hat = self.Gen(Z_batch) 

        # Supervisor
        H_hat = self.Sup(E_hat)
        H_hat_supervise = self.Sup(H)

        # Recovery
        X_tilde = self.Rec(H)
        X_hat = self.Rec(H_hat)

        # Discriminator
        Y_fake   = self.Dis(H_hat)
        Y_real   = self.Dis(H) 
        Y_fake_e = self.Dis(E_hat)

        # Losses
        D_loss = self.D_loss(Y_real, Y_fake,
                             Y_fake_e)
        
        G_loss, S_loss = self.GS_loss(Y_fake, Y_fake_e,
                                      X_batch, H,
                                      H_hat_supervise, X_hat)
        
        E_loss, R_loss = self.ER_loss(X_batch, X_tilde,
                                      S_loss)

        self.log_dict({
                "E_loss": E_loss,
                "D_loss": D_loss,
                "G_loss": G_loss,
                "S_loss": S_loss,
                "R_loss": R_loss
            }
        )
        return { "E_loss": E_loss, "D_loss": D_loss, "G_loss": G_loss, "S_loss": S_loss, "R_loss": R_loss }


    def get_image_examples(self,
                           real: torch.Tensor, fake: torch.Tensor):
        '''
        Given real and "fake" translated images, produce a nice coupled images to log

        Arguments:
            - `real`: the real sequence with shape [batch, seq_len, data_dim]
            - `fake`: the fake sequence with shape [batch, seq_len, data_dim]

        Returns:
            - A sequence of wandb.Image to log and visualize the performance
        '''
        example_images = []
        for i in range(real.shape[0]):
            couple = compare_sequences(real=real, fake=fake, save_img=False, show_graph=False)
            example_images.append(
                wandb.Image(couple.permute(1, 2, 0).detach().cpu().numpy(), mode="RGB")
            )
        return example_images


    def save_image_examples(self, real: torch.Tensor, fake: torch.Tensor, idx: int=0) -> None:
        '''
        Save the image of the plot
        '''
        compare_sequences(real=real, fake=fake, save_img=True, show_graph=False, img_idx=idx)


    def validation_step(
        self, X_batch: torch.Tensor, Z_batch: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]:
        '''
        Implements a single validation step

        In each validation step some translation examples are produced and a
        validation loss that uses the cycle consistency is computed

        Arguments:
            - `batch`: the current validation batch

        Returns:
            - the loss and example images
        '''
        # Embedder
        H = self.Emb(X_batch)

        # Generator
        E_hat = self.Gen(Z_batch) 

        # SUpervisor
        H_hat = self.Sup(E_hat)
        H_hat_supervise = self.Sup(H)

        # Recovery
        X_tilde = self.Rec(H)
        X_hat = self.Rec(H_hat)

        # Discriminator
        Y_fake   = self.Dis(H_hat)
        Y_real   = self.Dis(H) 
        Y_fake_e = self.Dis(E_hat)

        # Losses
        D_loss = self.D_loss(Y_real, Y_fake, Y_fake_e)
        G_loss, S_loss = self.GS_loss(Y_fake, Y_fake_e, X_batch, H, H_hat_supervise, X_hat)
        E_loss, R_loss = self.ER_loss(X_batch, X_tilde, S_loss)

        image = self.get_image_examples(X_batch[0], X_hat[0])

        aggregated_loss = 1/5*(E_loss + D_loss + G_loss + S_loss + R_loss)

        return { "val_loss": aggregated_loss, "image": image }


    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]]]:
        '''
        Implements the behaviouir at the end of a validation epoch

        Currently it gathers all the produced examples and log them to wandb,
        limiting the logged examples to `hparams["log_images"]`.

        Then computes the mean of the losses and returns it.
        Updates the progress bar label with this loss.

        Arguments:
            - outputs: a sequence that aggregates all the outputs of the validation steps

        Returns:
            - The aggregated validation loss and information to update the progress bar
        '''
        images = []

        for x in outputs:
            images.extend(x["image"])

        images = images[: self.hparams["log_images"]]

        if not self.is_sanity:  # ignore if it not a real validation epoch. The first one is not.
            print(f"Logged {len(images)} images.")

            self.logger.experiment.log(
                {f"images": images },
                step=self.global_step,
            )
        self.is_sanity = False

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log_dict({"val_loss": avg_loss})
        return {"val_loss": avg_loss}
    


### TESTING AREA
'''
from data_generation.wiener_process import get_weiner_process
from pprint import pprint
from dataclasses import asdict
hparams = Config()
p = 6
N = 10**3
#dh.train_test_split(torch.from_numpy(get_weiner_process(p=p, N=N)))
train_file_path = "./datasets/training.csv"
test_file_path = "./datasets/testing.csv"
model = TimeGAN(hparams, train_file_path, test_file_path)
'''