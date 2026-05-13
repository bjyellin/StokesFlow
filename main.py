from forward_solve import solve_state
from plot_forward_solve import plot_forward
from adjoint_solve import solve_terminal_adjoint, solve_adjoint_backwards
from dolfin import *
import numpy as np

class AdjointOptimizer:
    def __init__(self, mesh_size, T, num_steps, b_exps, theta_bar=1.0):
        """Initializes the optimization environment."""
        self.T = T
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.theta_bar = theta_bar
        self.b_exps = b_exps
        
        # FEniCS Mesh and Function Space
        self.mesh = UnitSquareMesh(mesh_size, mesh_size)
        self.V = FunctionSpace(self.mesh, "Lagrange", 1)
        self.num_coeffs = len(b_exps)

    def get_velocity_vector(self, u_coeffs):
        """Constructs the velocity field from basis functions and coefficients."""
        u_x = sum(c * b[0] for c, b in zip(u_coeffs, self.b_exps))
        u_y = sum(c * b[1] for c, b in zip(u_coeffs, self.b_exps))
        return as_vector([u_x, u_y])

    def compute_gradient(self, theta_traj, rho_traj):
        """Calculates the sensitivity gradient for the coefficients."""
        gradient_output = np.zeros(self.num_coeffs)
        
        for i in range(self.num_coeffs):
            b_i = self.b_exps[i]
            total_integral = 0.0
            
            # Trajectory integration
            for n in range(1, len(theta_traj)):
                # rho * (b_i . grad(theta)) dx
                term = rho_traj[n] * dot(b_i, grad(theta_traj[n])) * dx
                total_integral += assemble(term)
                
            gradient_output[i] = total_integral * self.dt
            
        return gradient_output

    def optimize(self, initial_u, lr, max_iterations, tol=1e-4):
        """Main optimization loop."""
        u_coeffs = list(initial_u)
        
        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            # 1. Forward Solve
            print("Running Forward Solve...")
            theta_traj = solve_state(u_coeffs, self.b_exps, self.T, self.num_steps, self.V, i)

            # 2. Terminal Adjoint Condition
            rho_T = solve_terminal_adjoint(theta_traj[-1], self.theta_bar, self.V)

            # 3. Backward Adjoint Solve
            print("Running Backwards Solve...")
            u_eff = self.get_velocity_vector(u_coeffs)
            rho_traj_rev = solve_adjoint_backwards(rho_T, u_eff, self.V, self.T, self.num_steps)
            rho_traj = rho_traj_rev[::-1] 

            # 4. Gradient Calculation
            grad_vec = self.compute_gradient(theta_traj, rho_traj)
            grad_norm = np.linalg.norm(grad_vec)
            
            print(f"Gradient Norm: {grad_norm:.6f}")
            print(f"Coefficients: {u_coeffs}")

            # 5. Convergence Check
            if grad_norm < tol:
                print("Optimality condition satisfied.")
                break

            # 6. Update and Plot
            u_coeffs = [c - lr * g for c, g in zip(u_coeffs, grad_vec)]
            plot_forward(theta_traj[0], theta_traj[-1], self.T, i)

        return u_coeffs

# --- Execution ---

set_log_level(LogLevel.WARNING)

# 1. Setup Parameters
basis_functions = [
    Expression(("-sin(pi*x[0])*cos(pi*x[1])", "cos(pi*x[0])*sin(pi*x[1])"), degree=2),
    Expression(("-sin(2*pi*x[0])*cos(2*pi*x[1])", "cos(2*pi*x[0])*sin(2*pi*x[1])"), degree=2),
    Expression(("-sin(3*pi*x[0])*cos(3*pi*x[1])", "cos(3*pi*x[0])*sin(3*pi*x[1])"), degree=2),
    Expression(("-sin(4*pi*x[0])*cos(4*pi*x[1])", "cos(4*pi*x[0])*sin(4*pi*x[1])"), degree=2)
]

# 2. Initialize Optimizer
optimizer = AdjointOptimizer(
    mesh_size=64, 
    T=1.0, 
    num_steps=100, 
    b_exps=basis_functions, 
    theta_bar=1.0
)

# 3. Run
final_coeffs = optimizer.optimize(
    initial_u=[1.0, 0.0, 0.0, 1.0], 
    lr=10.0, 
    max_iterations=10
)