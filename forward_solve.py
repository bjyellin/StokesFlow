import numpy as np
import matplotlib.pyplot as plt
import scipy
import dolfin 
from dolfin import *
from fenics import as_vector
from PIL import Image
import io
from pathlib import Path


def solve_state(coefficients, b_exps, T, num_time_steps, V, iteration):

    #Create folder to save results
    results_folder = "Results"
    results_dir = Path(results_folder)
    results_dir.mkdir(exist_ok=True)

    # Set log level to hide INFO messages
    set_log_level(LogLevel.WARNING)

    mesh = V.mesh()
    dt = T / num_time_steps
    
    # 1. Initialize
    theta_0_expr = Expression("tanh((2*x[1]-1)/0.2)+1", degree=2)
    theta_n = interpolate(theta_0_expr, V)
    
    # Storage for trajectory
    theta_trajectory = [Function(V)]
    theta_trajectory[0].assign(theta_n)

    # 2. Construct u_eff
    u_x = sum(c * b[0] for c, b in zip(coefficients, b_exps))
    u_y = sum(c * b[1] for c, b in zip(coefficients, b_exps))
    u_eff = as_vector([u_x,u_y]) 

    # 3. Variational Formulation (SUPG)
    theta = TrialFunction(V)
    v = TestFunction(V)
    h = CellDiameter(mesh)
    # Stabilization term
    v_supg = v + (h / (2.0 * sqrt(dot(u_eff, u_eff)+1e-10))) * dot(u_eff, grad(v))

    F = (theta - theta_n) / dt * v_supg * dx + dot(u_eff, grad(theta)) * v_supg * dx
    a, L = lhs(F), rhs(F)

    frames = []

    # 4. Time-Stepping
    for n in range(num_time_steps):
        theta_curr = Function(V)
        solve(a == L, theta_curr)

        #Frame capture 
        plt.figure(figsize=(6,5))
        p = plot(theta_curr, title=f"Time Step {n+1}/{num_time_steps}", cmap="viridis")
        plt.colorbar(p)

        #Save the plot to a buffer in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()

        # Save to trajectory
        theta_trajectory.append(theta_curr)
        # Update previous
        theta_n.assign(theta_curr)

    #Save frames as a GIF #RENAME GIF SO IT CHANGES WITH EACH ITERATION
    if frames:
        frames[0].save(f'Results/theta_evolution_{iteration}.gif', save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        print(f"Animation saved as theta_evolution_{iteration}.gif")

    return theta_trajectory # Return the full list of Functions