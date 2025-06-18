import torch, pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def load_cell_migration_data(file_path, initial_density_idx=0, plot=False):
    """
    Loads cell migration data formatted for BINNs training, using density index.

    Parameters:
    - file_path (str): Path to the .npy file.
    - initial_density_idx (int): Index for predefined density list.
    - plot (bool): Whether to show a 3D surface plot.

    Returns:
    - inputs (ndarray): Flattened [x, t] pairs, shape (N, 2)
    - outputs (ndarray): Corresponding densities, shape (N, 1)
    - shape (tuple): Original shape of the density grid (T, X)
    """

    # Predefined density levels
    densities = [0.3]
    density = densities[initial_density_idx]
    density_key = f'dens_{density}'

    # Load .npy file
    file = np.load(file_path, allow_pickle=True).item()
    
    # Extract meshgrid and density data
    X = file[density_key]['X']
    T = file[density_key]['T']
    U = file[density_key]['U']
    shape = U.shape  # (num_t, num_x)

    # Flatten for BINNs input
    inputs = np.concatenate([X.reshape(-1, 1), T.reshape(-1, 1)], axis=1)
    outputs = U.reshape(-1, 1)

    # Optional plotting
    if plot:

        #downsample
        time_step = 10
        space_step = 4 

        #downsample arrays
        X_plot= X[::time_step, ::space_step]
        T_plot= T[::time_step, ::space_step]
        U_plot= U[::time_step, ::space_step]
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_plot, T_plot, U_plot, cmap=cm.coolwarm, alpha=0.9)
        ax.set_title(f"Cell Migration Density (Initial Density: {density})")
        ax.set_xlabel("Position (space)")
        ax.set_ylabel("Time (days)")
        ax.set_zlabel("Density (cells/mm²)")
        plt.tight_layout()
        plt.show()

    return inputs, outputs, shape


'''
def load_cell_migration_data(file_path, initial_density, plot=False):
    
    densities = [0.4]
    density = densities[initial_density]
    
    # load data
    file = np.load(file_path, allow_pickle=True).item()

    # extract data
    density = densities[initial_density]
    # x = file[density]['x'].copy()[1:, :] 
    # t = file[density]['t'].copy()
    X = file[density]['X'].copy()[1:, :]
    T = file[density]['T'].copy()[1:, :]
    U = file[density]['U_mean'].copy()[1:, :]
    shape = U.shape

    # variable scales
    # x_scale = 1/1000 # micrometer -> millimeter
    # t_scale = 1/24 # hours -> days
    # u_scale = 1/(x_scale**2) # cells/um^2 -> cells/mm^2

    # scale variables
    # x *= x_scale
    # t *= t_scale
    # X *= x_scale
    # T *= t_scale
    # U *= u_scale

    # flatten for MLP
    inputs = np.concatenate([X.reshape(-1)[:, None],
                             T.reshape(-1)[:, None]], axis=1)
    outputs = U.reshape(-1)[:, None]

    if plot:
    
        # plot surface
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X, T, U, cmap=cm.coolwarm, alpha=1)
        ax.scatter(X.reshape(-1), T.reshape(-1), U.reshape(-1), s=5, c='k')
        plt.title('Initial density: '+density[5:])
        ax.set_xlabel('Position (millimeters)')
        ax.set_ylabel('Time (days)')
        ax.set_zlabel('Cell density (cells/mm^2)')
        ax.set_zlim(0, 2.2e3)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.show()
        
    return inputs, outputs, shape
'''