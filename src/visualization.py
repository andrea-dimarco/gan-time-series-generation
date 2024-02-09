
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch

   
def visualization(ori_data:torch.Tensor, generated_data:torch.Tensor, 
                  show_plot:bool=False, save_plot:bool=True,
                  folder_path:str="./",
                  ) -> None:
    """
    Using PCA or tSNE for generated and original data visualization
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
        n_sequences, seq_len, data_dim = ori_data.size()  
        N = n_sequences * seq_len
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
        plt.title('PCA plot')
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(f"{folder_path}pca-plot.png")
        plt.clf()