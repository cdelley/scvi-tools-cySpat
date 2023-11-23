from typing import Iterable, Union

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch

from ._module import DiffusionEncoder

class make_test_data(DiffusionEncoder):
    """
    A class to generate dummy data for the diffusionVAE model.

    This class extends the DiffusionEncoder to simulate spatial count data,
    which can be used for testing and validation purposes.

    Parameters
    ----------
    diffusion_constant : float
        The constant defining the diffusion process.
    scale : float
        Scaling factor for the signal rates.
    diffusion_model_kwargs : dict, default={}
        Additional keyword arguments for the diffusion model.
    downsample : int, default=0
        Number of data points to downsample to. If 0, no downsampling is performed.
    lambda_noise : Union[float, Iterable[float]], default=0
        Noise level or an array of noise levels for the count data.
    **kwargs
        Additional keyword arguments.
    """
    
    def __init__(
        self,
        diffusion_constant: float,
        scale: float,
        diffusion_model_kwargs: dict = {},
        downsample: int = 0,
        lambda_noise: Union[float, Iterable[float]] = 0,
        **kwargs,
    ):
        
        self.lambda_noise = lambda_noise
        
        # generate the target shape
        self.generate_data()
        z = torch.stack([torch.tensor(self.x_img),torch.tensor(self.y_img)]).T
        if downsample:
            z = z[:downsample]
            self.x_img = self.x_img[:downsample]
            self.y_img = self.y_img[:downsample]
        # generate a dummy encoding
        self.make_array()
        
        # generate tag counts
        super().__init__(
            self.encodings,
            torch.tensor(self.x_enc),
            torch.tensor(self.y_enc),
        )
        self.lambda_signal = self.forward(
            z = z, 
            diffusion_constant = diffusion_constant, 
        ) * scale
        
        # sample from Poisson
        self.count_data = torch.distributions.Poisson(self.lambda_signal + lambda_noise).sample()
        
        adata = ad.AnnData(X = np.array(self.count_data))
        adata.layers['rate_signal'] = np.array(self.lambda_signal)
        adata.obs['batch'] = np.ones(len(adata))
        self.adata = adata
        
        # make some plots
        self.plot_test_data()
        
    def generate_data(
        self,
        num_points: int = 400,
        radius: int = 24,
        num_points_circle1: int = 200,
        num_points_circle2: int = 100,
        num_points_circle3: int = 50,
        radius_circle1: int = 41,
        radius_circle2: int = 22,
        radius_circle3: int = 13,
        seed: int = 0,
    ) -> None:
        """
        Generates a target shape for the test data.

        Parameters
        ----------
        num_points : int, default=400
            Total number of points in the generated shape.
        radius : int, default=24
            Radius for the crescents.
        num_points_circle1 : int, default=200
            Number of points in the first circle.
        num_points_circle2 : int, default=100
            Number of points in the second circle.
        num_points_circle3 : int, default=50
            Number of points in the third circle.
        radius_circle1 : int, default=41
            Radius of the first circle.
        radius_circle2 : int, default=22
            Radius of the second circle.
        radius_circle3 : int, default=13
            Radius of the third circle.
        seed : int, default=0
            Seed for the random number generator.
        """
        
        np.random.seed(seed)
        # generate some target shape
        # Generate interdigitating demi-circles
        angle_range = np.linspace(0, np.pi, num_points)
        crescent1_x = radius * np.cos(angle_range) + np.random.normal(16, 2, num_points)
        crescent1_y = radius * np.sin(angle_range) + np.random.normal(-40 , 2, num_points)
        crescent2_x = -radius * np.cos(angle_range) + np.random.normal(-10, 2, num_points)
        crescent2_y = -radius * np.sin(angle_range) + np.random.normal(10, 2, num_points)

        # Generate circles within each other
        angle_range_circle1 = np.linspace(0, 2 * np.pi, num_points_circle1)
        angle_range_circle2 = np.linspace(0, 2 * np.pi, num_points_circle2)
        angle_range_circle3 = np.linspace(0, 2 * np.pi, num_points_circle3)
        circle1_x = radius_circle1 * np.cos(angle_range_circle1) + 3 
        circle1_y = radius_circle1 * np.sin(angle_range_circle1) + 1  
        circle2_x = radius_circle2 * np.cos(angle_range_circle2) + 11  
        circle2_y = radius_circle2 * np.sin(angle_range_circle2) + 9  
        circle3_x = radius_circle3 * np.cos(angle_range_circle3) - 17  
        circle3_y = radius_circle3 * np.sin(angle_range_circle3) + 15 

        # stack the point clouds
        x_img = np.hstack([crescent1_x, crescent2_x, circle1_x, circle2_x, circle3_x]) 
        y_img = np.hstack([crescent1_y, crescent2_y, circle1_y, circle2_y, circle3_y])
        self.n_cells = len(x_img)

        # shuffle them
        _shuffle_idx = np.arange(self.n_cells)
        np.random.shuffle(_shuffle_idx)
        self.y_img = y_img[_shuffle_idx]
        self.x_img = x_img[_shuffle_idx]

    def make_array(
        self,
        n_labels: int = 96,
        max_n_spots: int = 48,
        hexagonal: bool = False,
        seed: int = 7
    ) -> None:
        """
        Creates a simulated array of spots for spatial encoding.

        Parameters
        ----------
        n_labels : int, default=96
            Number of different labels/tags.
        max_n_spots : int, default=48
            Maximum number of spots in the array.
        hexagonal : bool, default=False
            If True, spots are arranged in a hexagonal grid.
        seed : int, default=7
            Seed for the random number generator.
        """
        
        np.random.seed(seed)
        labels = list(np.arange(n_labels))
        encodings = list(itertools.combinations(labels, 2))
        np.random.shuffle(encodings)

        # spot positions of the emitters
        n_spots = np.min([max_n_spots, int(np.sqrt(len(encodings)))])

        self.x_enc = np.array(list(np.arange(0,n_spots*2,2))*n_spots) - n_spots
        self.y_enc = np.ravel(np.ones(n_spots)[None,:]*np.arange(0,n_spots*2,2)[:,None]) - n_spots 
        if hexagonal:
            self.y_enc[::2] += 1

        # to truncate encodings to the actually used ones
        self.encodings = encodings[:len(self.x_enc)]
        
    def plot_test_data(self) -> None:
        """
        Plots the generated test data for visualization.
        """
        
        lambda_signal = self.lambda_signal
        lambda_noise = self.lambda_noise
        signal_rate_tot = lambda_noise + lambda_signal
        
        fig = plt.figure(figsize=(24,6))
        ax = fig.add_subplot(1,3,1)
        ax.set_aspect('equal')
        
        ax.scatter(self.y_enc, self.x_enc, s=20, marker='+')
        cbar = ax.scatter(self.y_img, self.x_img, s=10, 
                    cmap='magma', c=torch.sum(self.count_data, axis=1), vmin=0)
        ax.set_aspect('equal')
        plt.colorbar(cbar)
        
        ax2 = fig.add_subplot(2,3,2)
        ax2.imshow(np.array(signal_rate_tot[:32,:]))
        ax2.set_title('tag rate')
        
        ax3 = fig.add_subplot(2,3,5)
        ax3.imshow(np.array(self.count_data[:32,:]))
        ax3.set_title('tag count')
        
        ax4 = fig.add_subplot(2,6,11)
        ax4.plot(
            np.sort(
                torch.sum(
                    signal_rate_tot, 
                    axis=1
                )
            )[::-1]
        )
        #ax4.set_xscale('log')
        #ax4.set_yscale('log')
        ax4.set_xlabel('cell rank')
        ax4.set_ylabel('total tag rate')
        plt.show()
        
        
