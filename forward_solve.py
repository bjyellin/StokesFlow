import numpy as np
import matplotlib.pyplot as plt
import scipy
import dolfin 
from dolfin import *
from fenics import as_vector
from PIL import Image
import io
from pathlib import Path


# def solve_state(coefficients, b_exps, T, num_time_steps, V, iteration):

#     #Create folder to save results
#     results_folder = "Results"
#     results_dir = Path(results_folder)
#     results_dir.mkdir(exist_ok=True)

#     # Set log level to hide INFO messages
#     set_log_level(LogLevel.WARNING)

#     mesh = V.mesh()
#     dt = T / num_time_steps
    
#     # 1. Initialize
#     # theta_0_expr = Expression("tanh((2*x[1]-1)/0.2)+1", degree=2)
#     theta_0_expr = Expression("tanh((2*x[1]-1)/0.2)+1", degree=2)
#     theta_n = interpolate(theta_0_expr, V)
#     theta_curr = interpolate(theta_0_expr, V)
#     theta_curr.assign(theta_n)
    
#     # Storage for trajectory
#     theta_trajectory = [Function(V)]
#     theta_trajectory[0].assign(theta_n)

#     # 2. Construct u_eff
#     u_x = sum(c * b[0] for c, b in zip(coefficients, b_exps))
#     u_y = sum(c * b[1] for c, b in zip(coefficients, b_exps))
#     u_eff = as_vector([u_x,u_y]) 

#     # 3. Variational Formulation (SUPG)
#     theta = TrialFunction(V)
#     v = TestFunction(V)
#     h = CellDiameter(mesh)
#     # Stabilization term
#     v_supg = v + (h / (2.0 * sqrt(dot(u_eff, u_eff)+1e-10))) * dot(u_eff, grad(v))

#     F = (theta - theta_n) / dt * v_supg * dx + dot(u_eff, grad(theta)) * v_supg * dx
#     a, L = lhs(F), rhs(F)

#     frames = []

#     # 4. Time-Stepping
#     for n in range(num_time_steps):
#         # theta_curr = Function(V)
#         solve(a == L, theta_curr)

#         #Frame capture 
#         plt.figure(figsize=(6,5))
#         p = plot(theta_curr, title=f"Time Step {n+1}/{num_time_steps}", cmap="viridis")
#         plt.colorbar(p)

#         #Save the plot to a buffer in memory
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         frames.append(Image.open(buf))
#         plt.close()

#         # Save to trajectory
#         theta_trajectory.append(theta_curr)
#         # Update previous
#         theta_n.assign(theta_curr)

#     #Save frames as a GIF #RENAME GIF SO IT CHANGES WITH EACH ITERATION
#     if frames and iteration % 10 == 0:
#         print("are we about to save an animation?")
#         frames[0].save(f'/Users/benjaminyellin/Desktop/BensStuff/UGA/Fluid Mixing/Results/theta_evolution_{iteration}.gif', save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
#         print(f"Animation saved as theta_evolution_{iteration}.gif")

#     return theta_trajectory # Return the full list of Functions

# def solve_state(coefficients, b_exps, T, num_time_steps, V, iteration):

#     # Create folder to save results
#     results_folder = "Results"
#     results_dir = Path(results_folder)
#     results_dir.mkdir(exist_ok=True)

#     # Set log level to hide INFO messages
#     set_log_level(LogLevel.WARNING)

#     mesh = V.mesh()
#     dt = T / num_time_steps
    
#     # 1. Initialize Functions outside the loop
#     theta_0_expr = Expression("tanh((2*x[1]-1)/0.2)+1", degree=2)
#     theta_n = interpolate(theta_0_expr, V)
#     theta_curr = Function(V)
#     theta_curr.assign(theta_n)
    
#     # Storage for trajectory: Deep copy the initial condition
#     snapshot_0 = Function(V)
#     snapshot_0.assign(theta_n)
#     theta_trajectory = [snapshot_0]

#     # 2. Construct u_eff
#     u_x = sum(c * b[0] for c, b in zip(coefficients, b_exps))
#     u_y = sum(c * b[1] for c, b in zip(coefficients, b_exps))
#     u_eff = as_vector([u_x, u_y]) 

#     # 3. Corrected Variational Formulation (Isolated SUPG)
#     theta = TrialFunction(V)
#     v = TestFunction(V)
#     h = CellDiameter(mesh)

#     # OVERRIDE HERE: Create a local measure bound to your mesh with strict quadrature limits
#     dx_quad = Measure('dx', domain=mesh, metadata={'quadrature_degree': 4})
    
#     # Baseline Galerkin formulation
#     F_galerkin = ((theta - theta_n) / dt) * v * dx_quad + dot(u_eff, grad(theta)) * v * dx_quad
    
#     # Strong residual tracking for consistent stabilization
#     residual = (theta - theta_n) / dt + dot(u_eff, grad(theta))
    
#     # Stabilization parameter parameterization
#     u_norm = sqrt(dot(u_eff, u_eff) + 1e-10)
    
#     # Smooth harmonic combination of temporal and spatial limits
#     tau = 1.0 / sqrt((2.0 / dt)**2 + (2.0 * u_norm / h)**2)
    
#     # Total formulation
#     F = F_galerkin + tau * residual * dot(u_eff, grad(v)) * dx_quad
#     a, L = lhs(F), rhs(F)


#     #Check the CFL Condition 
#     # Define a DG0 function space to compute cell-wise maximums
#     DG0 = FunctionSpace(mesh, "DG", 0)
    
#     # Define the local CFL expression: ||u|| * dt / h
#     u_norm_expr = sqrt(dot(u_eff, u_eff) + 1e-10)
#     cfl_expr = (u_norm_expr * dt) / h
    
#     # Project the expression onto the DG0 space to get an array of cell values
#     cfl_function = project(cfl_expr, DG0)
#     max_cfl = cfl_function.vector().max()
    
#     # Print a warning or fail-safe error if it exceeds the stability limit
#     if max_cfl > 1.0:
#         print(f"\n[WARNING] CFL Condition Violated! Max CFL = {max_cfl:.3f}")
#         print(f"Velocity field is too fast for dt={dt:.5f} and local cell size h.")
#         print("Consider increasing num_time_steps in your main loop script.\n")
#         raise ValueError("CFL Condition Violated: Simulation may be unstable. Adjust time step or velocity field.")
#     else:
#         print(f"[INFO] CFL Condition Checked: Max CFL = {max_cfl:.3f} (Stable)")

#     frames = []

#     # 4. Time-Stepping Loop
#     for n in range(num_time_steps):
#         # Solves directly into the persistent theta_curr function memory space
#         solve(a == L, theta_curr)

#         # Frame capture 
#         plt.figure(figsize=(6,5))
#         p = plot(theta_curr, title=f"Time Step {n+1}/{num_time_steps}", cmap="viridis")
#         plt.colorbar(p)

#         # Save the plot to a buffer in memory
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         frames.append(Image.open(buf))
#         plt.close()

#         # FIX 1: Deep copy the current solution into an independent object for history tracking
#         snapshot = Function(V)
#         snapshot.assign(theta_curr)
#         theta_trajectory.append(snapshot)
        
#         # FIX 2: Explicitly progress time forward by writing current values over the history variable
#         theta_n.assign(theta_curr)

#     # Save frames as a GIF
#     if frames and iteration % 10 == 0:
#         print("Saving animation...")
#         gif_path = f'/Users/benjaminyellin/Desktop/BensStuff/UGA/Fluid Mixing/Results/theta_evolution_{iteration}.gif'
#         frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
#         print(f"Animation saved as theta_evolution_{iteration}.gif")

#     return theta_trajectory

def solve_state(coefficients, b_exps, T, num_time_steps, V, iteration):

    # Create folder to save results
    results_folder = "Results"
    results_dir = Path(results_folder)
    results_dir.mkdir(exist_ok=True)

    # Set log level to hide INFO messages
    set_log_level(LogLevel.WARNING)

    mesh = V.mesh()
    dt = T / num_time_steps
    
    # 1. Initialize Functions outside the loop
    theta_0_expr = Expression("tanh((2*x[1]-1)/0.2)+1", degree=2)
    theta_n = interpolate(theta_0_expr, V)
    theta_curr = Function(V)
    theta_curr.assign(theta_n)
    
    # Storage for trajectory: Deep copy the initial condition
    snapshot_0 = Function(V)
    snapshot_0.assign(theta_n)
    theta_trajectory = [snapshot_0]

    # 2. Construct u_eff
    u_x = sum(c * b[0] for c, b in zip(coefficients, b_exps))
    u_y = sum(c * b[1] for c, b in zip(coefficients, b_exps))
    u_eff = as_vector([u_x, u_y]) 

    # 3. Corrected Variational Formulation (Isolated SUPG)
    theta = TrialFunction(V)
    v = TestFunction(V)
    h = CellDiameter(mesh)

    # Create a local measure bound to your mesh with strict quadrature limits
    dx_quad = Measure('dx', domain=mesh, metadata={'quadrature_degree': 4})
    
    # Baseline Galerkin formulation
    F_galerkin = ((theta - theta_n) / dt) * v * dx_quad + dot(u_eff, grad(theta)) * v * dx_quad
    
    # # Strong residual tracking for consistent stabilization
    # residual = (theta - theta_n) / dt + dot(u_eff, grad(theta))
    
    # # Stabilization parameter parameterization
    # u_norm = sqrt(dot(u_eff, u_eff) + 1e-10)
    
    # # Smooth harmonic combination of temporal and spatial limits
    # tau = 1.0 / sqrt((2.0 / dt)**2 + (2.0 * u_norm / h)**2)
    
    # # Total formulation
    # F = F_galerkin + tau * residual * dot(u_eff, grad(v)) * dx_quad

    # 2. Strong residual tracking 
    residual = (theta - theta_n) / dt + dot(u_eff, grad(theta))
    
    # 3. Spatio-temporal streamline stabilization (SUPG)
    u_norm = sqrt(dot(u_eff, u_eff) + 1e-10)
    tau = 1.0 / sqrt((2.0 / dt)**2 + (2.0 * u_norm / h)**2)
    F_supg = tau * residual * dot(u_eff, grad(v)) * dx_quad
    
    # 4. ADDED: Isotropic Shock-Capturing Term (Cross-wind stabilization)
    grad_theta = grad(theta)
    grad_norm_n = sqrt(dot(grad(theta_n), grad(theta_n)) + 1e-10)
    residual_n = dot(u_eff, grad(theta_n))
    
    # Residual-based shock capturing parameter
    # It scales down dynamically where the solution is already smooth
    tau_sc = h / (2.0 * u_norm + 1e-10) 

    residual_n_abs = sqrt(residual_n**2 + 1e-10)
    
    # Isotropic diffusion proportional to the strong residual magnitude
    F_shock = tau_sc * residual_n_abs * (dot(grad(theta), grad(v)) / grad_norm_n) * dx_quad
    
    # Total formulation
    F = F_galerkin + F_supg + F_shock


    a, L = lhs(F), rhs(F)

    # Pre-assemble matrix LHS once since u_eff is static across this execution sweep
    A = assemble(a)

    # Check the CFL Condition 
    DG0 = FunctionSpace(mesh, "DG", 0)
    u_norm_expr = sqrt(dot(u_eff, u_eff) + 1e-10)
    cfl_expr = (u_norm_expr * dt) / h
    
    cfl_function = project(cfl_expr, DG0)
    max_cfl = cfl_function.vector().max()
    
    if max_cfl > 1.0:
        print(f"\n[WARNING] CFL Condition Violated! Max CFL = {max_cfl:.3f}")
        print(f"Velocity field is too fast for dt={dt:.5f} and local cell size h.")
        print("Consider increasing num_time_steps in your main loop script.\n")
        raise ValueError("CFL Condition Violated: Simulation may be unstable. Adjust time step or velocity field.")
    else:
        print(f"[INFO] CFL Condition Checked: Max CFL = {max_cfl:.3f} (Stable)")

    frames = []

    # 4. Time-Stepping Loop
    for n in range(num_time_steps):
        # Explicitly re-assemble RHS vector to accurately pull fresh data from theta_n
        b = assemble(L)
        solve(A, theta_curr.vector(), b)

        # Performance Optimization: Only plot if we are generating a GIF for this iteration
        if iteration % 10 == 0:
            plt.figure(figsize=(6,5))
            p = plot(theta_curr, title=f"Time Step {n+1}/{num_time_steps}", cmap="viridis")
            plt.colorbar(p)

            # Save the plot to a buffer in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close()

        # Deep copy the current solution into an independent object for history tracking
        snapshot = Function(V)
        snapshot.assign(theta_curr)
        theta_trajectory.append(snapshot)
        
        # Explicitly progress time forward by writing current values over the history variable
        theta_n.assign(theta_curr)

    # Save frames as a GIF
    if frames and iteration % 10 == 0:
        print("Saving animation...")
        gif_path = f'/Users/benjaminyellin/Desktop/BensStuff/UGA/Fluid Mixing/Results/theta_evolution_{iteration}.gif'
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        print(f"Animation saved as theta_evolution_{iteration}.gif")

    return theta_trajectory