

# Libraries
from typing import Sequence, Dict, Tuple, Union, Mapping

from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

import torch
from torch import optim

import wandb
import pytorch_lightning as pl

# My modules
import dataset_handling as dh
import utilities as ut
from hyperparamets import Config
from modules.discriminator import Discriminator
from modules.embedder import Embedder
from modules.generator import Generator
from modules.recovery import Recovery
from modules.supervisor import Supervisor

'''
This is the main model.
 Following the Lightning Module API
'''
class TimeGAN(pl.LightningModule):
    def __init__(self,
        hparams: Union[Dict, Config],
        train_file_path: Path,
        val_file_path: Path,
        device: torch.device,
        plot_losses: bool=False
    ) -> None:
        '''
        The TimeGAN model.

        Arguments:
            - `hparams`: dictionary that contains all the hyperparameters
            - `train_file_path`: Path to the folder that contains the training stream
            - `val_file_path`: Path to the file that contains the testing stream
            - `plot_losses`: Saves the losses in the `loss_history`
        '''
        super().__init__()
        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Mapping) else hparams)
        self.dev = device

        # Dataset paths
        self.train_file_path = train_file_path
        self.val_file_path  = val_file_path

        # loss criteria
        self.discrimination_loss = torch.nn.BCELoss()
        self.reconstruction_loss = torch.nn.L1Loss()#MSELoss()

        # Expected shapes 
        self.data_dim = self.hparams["data_dim"]
        self.latent_space_dim = self.hparams["latent_space_dim"]
        self.noise_dim = self.hparams["noise_dim"]
        self.seq_len = self.hparams["seq_len"]

        if plot_losses:
            self.plot_losses = True
            self.loss_history = []
        else:
            self.plot_losses = False
            self.loss_history = None

        # Initialize Modules
        self.Gen = Generator(input_size=self.noise_dim,
                            hidden_size=self.hparams["gen_hidden_dim"],
                            output_size=self.latent_space_dim,
                            num_layers=self.hparams["gen_num_layers"],
                            module_type=self.hparams["gen_module_type"],
                            device=self.dev
                            )
        self.Emb = Embedder(input_size=self.data_dim,
                            hidden_size=self.hparams["emb_hidden_dim"],
                            output_size=self.latent_space_dim,
                            num_layers=self.hparams["emb_num_layers"],
                            module_type=self.hparams["emb_module_type"],
                            device=self.dev
                            )
        self.Rec = Recovery(input_size=self.latent_space_dim,
                            hidden_size=self.hparams["rec_hidden_dim"],
                            output_size=self.data_dim,
                            num_layers=self.hparams["rec_num_layers"],
                            module_type=self.hparams["rec_module_type"],
                            device=self.dev
                            )
        self.Sup = Supervisor(input_size=self.latent_space_dim,
                              num_layers=self.hparams["sup_num_layers"],
                              module_type=self.hparams["sup_module_type"],
                              device=self.dev
                              )
        self.Dis = Discriminator(input_size=self.latent_space_dim,
                                 hidden_size=self.hparams["dis_hidden_dim"],
                                 num_layers=self.hparams["dis_num_layers"],
                                 module_type=self.hparams["dis_module_type"],
                                 device=self.dev
                                 )

        # Image buffers
        #TODO: actually use this
        self.fake_buffer = ut.ReplayBuffer()

        # Forward pass cache to avoid re-doing some computation
        self.fake = None

        # It avoids wandb logging when lighting does a sanity check on the validation
        self.is_sanity = True

        # multiple optimizers are no longer supported by LightningModule
        self.automatic_optimization = False

        # For the end of the validation step
        self.validation_step_output = []


    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `Z`: input of the forward pass with shape [batch, seq_len, noise_dim]

        Returns:
            - the translated image with shape [batch, seq_len, data_dim]
        '''

        assert(Z.size()[2] == self.noise_dim), "Invalid dimention for noise sample"

        # Generate a synthetic sequence in the latent space
        E_hat = self.Gen(Z) # generator

        # Recontrust fake data from the synthetic sequence
        X_hat = self.Rec(E_hat) # recovery

        return X_hat


    def cycle(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Embeds the sequence and recovers it back.

        Arguments:
            - `X`: input of the forward pass with shape [batch, seq_len, data_dim]

        Returns:
            - the translated image with shape [batch, seq_len, data_dim]
        '''

        assert(X.size()[2] == self.data_dim), "Invalid dimention for data sample"

        # 1. Embedder
        H = self.Emb(X)
        # 2. Recovery
        X_tilde = self.Rec(H)

        return X_tilde


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
            pin_memory=True
        )
        return train_loader
        

    def val_dataloader(self) -> DataLoader:
        '''
        Create the validation set DataLoader.

        It is deterministic.
        It does not shuffle and does not use random transformation on each image.
        
        Returns:
            - `val_loader`: the validation set DataLoader
        '''
        val_loader = DataLoader(
            dh.RealDataset(
                file_path=self.val_file_path,
                seq_len=self.seq_len
            ),
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            pin_memory=True
        )
        return val_loader


    def configure_optimizers(self
    ) -> Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer, optim.Optimizer, optim.Optimizer]:
    #) -> Tuple[Sequence[optim.Optimizer], Sequence[Dict[str, Any]]]:
        '''
        Instantiate the optimizers and schedulers.

        We have five optimizers (and relative schedulers):

        - `E_optim`: optimzer for the Embedder
        - `D_optim`: optimizer for the Discriminator
        - `G_optim`: optimizer for the Generator      
        - `S_optim`: optimizer for the Supervisor
        - `R_optim`: optimizer for the Recovery
        
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
        E_optim = torch.optim.Adam(
            self.Emb.parameters(recurse=True), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )
        D_optim = torch.optim.Adam(
            self.Dis.parameters(recurse=True), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )
        G_optim = torch.optim.Adam(
            self.Gen.parameters(recurse=True), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )
        S_optim = torch.optim.Adam(
            self.Sup.parameters(recurse=True), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )
        R_optim = torch.optim.Adam(
            self.Rec.parameters(recurse=True), lr=self.hparams["lr"], betas=(self.hparams["b1"], self.hparams["b2"])
        )

        # linear decay scheduler
        #assert(self.hparams["n_epochs"] > self.hparams["decay_epoch"]), "Decay must start BEFORE the training ends!"
        #linear_decay = lambda epoch: float(1.0 - max(0, epoch-self.hparams["decay_epoch"]) / (self.hparams["n_epochs"]-self.hparams["decay_epoch"]))
        

        # Schedulers 
        # lr_scheduler_E = torch.optim.lr_scheduler.LinearLR(
        #     E_optim,
        #     start_factor=1.0,
        #     end_factor=0.1
        # )
        # lr_scheduler_D = torch.optim.lr_scheduler.LinearLR(
        #     D_optim,
        #     start_factor=1.0,
        #     end_factor=0.1
        # )
        # lr_scheduler_G = torch.optim.lr_scheduler.LinearLR(
        #     G_optim,
        #     start_factor=1.0,
        #     end_factor=0.1
        # )
        # lr_scheduler_S = torch.optim.lr_scheduler.LinearLR(
        #     S_optim,
        #     start_factor=1.0,
        #     end_factor=0.1
        # )
        # lr_scheduler_R = torch.optim.lr_scheduler.LinearLR(
        #     R_optim,
        #     start_factor=1.0,
        #     end_factor=0.1
        # )

        # return (
        #     [E_optim, D_optim, G_optim, S_optim, R_optim],
        #     [
        #         {"scheduler": lr_scheduler_E, "interval": "epoch", "frequency": 1},
        #         {"scheduler": lr_scheduler_D, "interval": "epoch", "frequency": 1},
        #         {"scheduler": lr_scheduler_G, "interval": "epoch", "frequency": 1},
        #         {"scheduler": lr_scheduler_S, "interval": "epoch", "frequency": 1},
        #         {"scheduler": lr_scheduler_R, "interval": "epoch", "frequency": 1}
        #     ]
        # )
        return E_optim, D_optim, G_optim, S_optim, R_optim


    def get_opt_idx(self, module_name: str
    ) -> int:
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


    def D_loss(self, X: torch.Tensor, Z: torch.Tensor,
               w1:float=0.40, w2:float=0.40, w3:float=0.20
    ) -> torch.Tensor:
        '''
        This function computes the loss for the DISCRIMINATOR module.

        Arguments:
            - `X`: batch of real sequences
            - `Z`: batch of random noise sequences

        Returns:
            - `D_loss`: tensor with one element containing the Discriminator module's loss
        '''
        # Compute model outputs
            # 1. Embedder
        H = self.Emb(X)
            # 2. Generator
        E_hat = self.Gen(Z) 
            # 3. Supervisor
        H_hat = self.Sup(E_hat)
            # 4. Discriminator
        Y_fake   = self.Dis(H_hat)
        Y_real   = self.Dis(H)
        Y_fake_e = self.Dis(E_hat)


        # Adversarial truths
        valid = torch.ones_like(Y_real)
        fake  = torch.zeros_like(Y_fake)


        # Loss Components
        loss_real   = self.discrimination_loss(Y_real,   valid)
        loss_fake   = self.discrimination_loss(Y_fake,   fake)
        loss_fake_e = self.discrimination_loss(Y_fake_e, fake)

        return w1*loss_real + w2*loss_fake + w3*loss_fake_e


    def G_loss(self, X: torch.Tensor, Z: torch.Tensor,
                w1:float=0.10, w2:float=0.35, w3:float=0.10, w4:float=0.45
    ) -> torch.Tensor:
        '''
        This function computes the loss for the GENERATOR module.

        Arguments:
            - `X`: batch of real sequences
            - `Z`: batch of random noise sequences

        Returns:
            - `G_loss`: tensor with one element containing the Generator module's loss
        '''
        # Compute model outputs
            # 1. Generator
        E_hat = self.Gen(Z) 
            # 2. Supervisor
        H_hat = self.Sup(E_hat)
            # 3. Recovery
        X_hat = self.Rec(E_hat)
            # 4. Discriminator
        Y_fake   = self.Dis(H_hat)
        Y_fake_e = self.Dis(E_hat)


        # Loss components
            # 1. Adversarial truth
        valid = torch.ones_like(Y_fake)
            # 2. Adversarial loss
        GA_loss   = self.discrimination_loss(Y_fake,   valid)
        GA_loss_e = self.discrimination_loss(Y_fake_e, valid)
            # 3. Supervised loss
        S_loss    = self.reconstruction_loss(H_hat[:,1:,:], E_hat[:,:-1,:])
            # 4. Deviation loss
        G_loss_mu = torch.mean(
            torch.abs((torch.mean(X_hat, dim=0)) - (torch.mean(X, dim=0))))
        G_loss_std = torch.mean(
            torch.abs(
                torch.sqrt(torch.var(X_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(X, dim=0) + 1e-6)))
        G_loss_V  = G_loss_mu + G_loss_std

        return w1*GA_loss + w2*GA_loss_e + w3*S_loss*0.0 + w4*G_loss_V 
    

    def S_loss(self, X: torch.Tensor, Z: torch.Tensor,
               w1:float=0.4, w2:float=0.6, scaling_factor=1000
    ) -> torch.Tensor:
        '''
        This function computes the loss for the SUPERVISOR module.

        Arguments:
            - `X`: batch of real sequences
            - `Z`: batch of random noise sequences

        Returns:
            - `S_loss`: tensor with one element containing the Supervisor module's loss
        '''
        # Compute model outputs
            # 1. Embedder
        H = self.Emb(X)
            # 2. Generator
        E_hat = self.Gen(Z) 
            # 3. Supervisor
        H_hat = self.Sup(E_hat)
        H_hat_supervise = self.Sup(H)

        # Loss components
            # 1. Reconstruction Loss
        Rec_loss = self.reconstruction_loss(H, H_hat_supervise)
        #Rec_loss = self.reconstruction_loss(H, H_hat_supervise)
            # 2. Deviation Loss
        Dev_loss_mu = torch.mean(
            torch.abs(
                torch.sqrt(torch.var(H_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(H_hat_supervise, dim=0) + 1e-6)))
        Dev_loss_std = torch.mean(
            torch.abs((torch.mean(H_hat, dim=0)) - (torch.mean(H_hat_supervise, dim=0))))
        Dev_loss = Dev_loss_mu + Dev_loss_std

        # Supervised loss
        return (w1*Rec_loss + w2*Dev_loss)*scaling_factor


    def E_loss(self, X: torch.Tensor,
               w1: float=0.5, w2:float=0.5
    ) -> torch.Tensor:
        '''
        This function computes the loss for the EMBEDDER module.

        Arguments:
            - `X`: batch of real sequences
            - `Z`: batch of random noise sequences

        Returns:
            - `E_loss`: tensor with one element containing the Embedder module's loss
        '''
        # Compute model outputs
            # 1. Embedder
        H = self.Emb(X)
            # 2. Supervisor
        H_hat_supervise = self.Sup(H)
            # 3. Recovery
        X_tilde = self.Rec(H)

        # Loss Components
        R_loss = self.reconstruction_loss(X, X_tilde)
        S_loss = self.reconstruction_loss(H, H_hat_supervise)

        return w1*R_loss + w2*S_loss
    

    def R_loss(self, X: torch.Tensor, scaling_factor=10
    ) -> torch.Tensor:
        '''
        This function computes the loss for the RECOVERY module.

        Arguments:
            - `X`: batch of real sequencess

        Returns:
            - `R_loss`: tensor with one element containing the Recovery module's loss
        '''
        # Compute model outputs
            # 1. Embedder
        H = self.Emb(X)
            # 2. Recovery
        X_tilde = self.Rec(H)

        return self.reconstruction_loss(X, X_tilde)*scaling_factor


    def training_step(self, 
                      batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int
    ) -> Dict[str, torch.Tensor]:
        '''
        Implements a single training step

        The parameter `optimizer_idx` identifies with optimizer "called" this training step,
        this way we can change the behaviour of the training depending on which optimizer
        is currently performing the optimization

        Arguments:
            - `batch`: current training batch
            - `batch_idx`: the index of the batch being processed

        Returns:
            - the total loss for the current training step, together with other information for the
                  logging and possibly the progress bar
        '''
        # Process the batch
        X_batch, Z_batch = batch
        E_optim, D_optim, G_optim, S_optim, R_optim = self.optimizers()
        
        # Zero grad
        E_optim.zero_grad()
        D_optim.zero_grad()
        G_optim.zero_grad()
        S_optim.zero_grad()
        R_optim.zero_grad()

        # Discriminator Loss
        D_loss = self.D_loss(X=X_batch, Z=Z_batch)
        D_loss.backward()
        D_optim.step()
        
        # Generator Loss
        G_loss = self.G_loss(X=X_batch, Z=Z_batch) 
        G_loss.backward()
        G_optim.step()
        
        # Supervisor Loss
        S_loss = self.S_loss(X=X_batch, Z=Z_batch)
        S_loss.backward()
        S_optim.step()
        
        # Embedder Loss
        E_loss = self.E_loss(X=X_batch)
        E_loss.backward()
        E_optim.step()

        # Recovery Loss
        R_loss = self.R_loss(X=X_batch)
        R_loss.backward()
        R_optim.step()

        # Log results
        if self.plot_losses:
            self.loss_history.append( [G_loss.item(), D_loss.item(), G_loss.item(), S_loss.item(), R_loss.item()] )
        loss_dict = { "E_loss": E_loss, "D_loss": D_loss, "G_loss": G_loss, "S_loss": S_loss, "R_loss": R_loss }
        self.log_dict(loss_dict)

        return loss_dict


    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
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
        # Process batch
        X_batch, Z_batch = batch

        # Losses
        D_loss = self.D_loss(X=X_batch, Z=Z_batch)
        G_loss = self.G_loss(X=X_batch, Z=Z_batch)
        S_loss = self.S_loss(X=X_batch, Z=Z_batch)
        E_loss = self.E_loss(X=X_batch)
        R_loss = self.R_loss(X=X_batch)

        # visualize result
        image = self.get_image_examples(X_batch[0], self.Rec(self.Sup(self.Gen(Z_batch)))[0])

        # Validation loss
        w_e = 0.20
        w_d = 0.20
        w_g = 0.20
        w_s = 0.20
        w_r = 0.20
        aggregated_loss = w_e*E_loss + w_d*D_loss + w_g*G_loss + w_s*S_loss + w_r*R_loss
        self.log("val_loss", aggregated_loss)

        val_out = { "val_loss": aggregated_loss, "image": image }
        self.validation_step_output.append(val_out)

        return val_out


    def get_image_examples(self,
                           real: torch.Tensor, fake: torch.Tensor):
        '''
        Given real and "fake" translated images, produce a nice coupled images to log

        Arguments:
            - `real`: the real sequence with shape [seq_len, data_dim]
            - `fake`: the fake sequence with shape [seq_len, data_dim]

        Returns:
            - A sequence of wandb.Image to log and visualize the performance
        '''
        example_images = []
        couple = ut.compare_sequences(real=real, fake=fake, save_img=False, show_graph=False)

        example_images.append(
            wandb.Image(couple, mode="RGB")
        )
        return example_images


    def save_image_examples(self,
                            real: torch.Tensor, fake: torch.Tensor, idx:int=0
    ) -> None:
        '''
        Save the image of the plot with the real and fake sequence
        '''
        ut.compare_sequences(real=real, fake=fake, save_img=True, show_graph=False, img_idx=idx)


    def on_validation_epoch_end(self
    ) -> Dict[str, torch.Tensor]:
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

        for x in self.validation_step_output:
            images.extend(x["image"])

        images = images[: self.hparams["log_images"]]

        if not self.is_sanity:  # ignore if it not a real validation epoch. The first one is not.
            #print(f"Logged {len(images)} images.")

            self.logger.experiment.log(
                {f"images": images },
                step=self.global_step,
            )
        self.is_sanity = False

        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_output]).mean()
        self.log_dict({"val_loss": avg_loss})
        self.validation_step_output = []
        return {"val_loss": avg_loss}
    

    def plot(self) -> None:
        '''
        Plot funky graaph
        '''
        import numpy as np
        if self.plot_losses and len(self.loss_history)>0:
            labels=["E_loss", "D_loss", "G_loss", "S_loss", "R_loss"],
            ut.plot_process(samples=np.asarray(self.loss_history),
                            labels=labels,
                            show_plot=True,
                            save_picture=True)
    
