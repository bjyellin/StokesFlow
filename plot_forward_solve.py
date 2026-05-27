import numpy as np
import matplotlib.pyplot as plt
import scipy
import dolfin 
from dolfin import *
from pathlib import Path

def plot_forward(theta_initial, theta_curr, time_limit, iteration):

    #Create folder to save results
    results_folder = "Results"
    results_dir = Path(results_folder)
    results_dir.mkdir(exist_ok=True)

    # --- ADDED: Plotting Code ---
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if iteration % 10 == 0:

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- Plot Initial Condition ---
        plt.sca(axes[0]) # Set Current Axes to the first subplot
        p0 = plot(theta_initial, title="Initial Condition ($t=0$)", cmap="viridis")
        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(p0, cax=cax1)

        # --- Plot Final Condition ---
        plt.sca(axes[1]) # Set Current Axes to the second subplot
        pf = plot(theta_curr, title=f"Final Condition ($t={time_limit}$)", cmap="viridis")
        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(pf, cax=cax2)

        plt.tight_layout()
        
        plt.savefig(f'/Users/benjaminyellin/Desktop/BensStuff/UGA/Fluid Mixing/Results/forward_solve_results_{iteration}.png')
        plt.close(fig)