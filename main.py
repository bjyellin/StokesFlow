from forward_solve import solve_state
from plot_forward_solve import plot_forward
from adjoint_solve import solve_terminal_adjoint, solve_adjoint_backwards
from dolfin import *
import numpy as np

class AdjointOptimizer:
    def __init__(self, mesh_size, T, num_steps, b_exps, lmbda, lr_lmbda, r, theta_bar=1.0):
        """Initializes the optimization environment."""
        self.T = T
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.theta_bar = theta_bar
        self.b_exps = b_exps
        self.lr_lmbda = lr_lmbda 
        self.r = r
        self.lmbda = lmbda
        
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
    
    #Not using r^2C0_^2 as the threshold for the error yet
    def final_time_constraint_satisfied(self, theta_initial, theta_final):
        """
        Checks if the final state is 'r' times better than the initial state.
        r = 0.1 means 90% of the variance has been mixed away.
        """
        # 1. Compute Initial Error (C0)
        e_initial = theta_initial - Constant(self.theta_bar)
        # Using H1 norm as you did before: ||e||^2 + ||grad(e)||^2
        c0_sq = assemble((dot(e_initial, e_initial) + dot(grad(e_initial), grad(e_initial))) * dx)
        c0 = sqrt(c0_sq)

        # 2. Compute Final Error (Ef)
        e_final = theta_final - Constant(self.theta_bar)
        ef_sq = assemble((dot(e_final, e_final) + dot(grad(e_final), grad(e_final))) * dx)
        ef = sqrt(ef_sq)

        # 3. The Condition: Ef <= r * C0
        print(f"Initial Error (C0): {c0:.6f} | Final Error: {ef:.6f}")
        print(f"Target: {self.r * c0:.6f}")
        
        return ef <= (self.r * c0)
    

    def optimality_condition_satisfied(self, theta, rho_traj, u_coeffs):
        n = len(self.b_exps)
        M = np.zeros((n, n))
        p = np.zeros(n) # Use 1D array for easier solving
        
        # Use the most recent state/adjoint for the RHS
        current_rho = rho_traj[-1]
        
        for i in range(n):
            # p_i = integral( b_i · (grad(rho) * theta) )
            p[i] = assemble(dot(self.b_exps[i], grad(current_rho) * theta) * dx)
            
            for j in range(i, n):
                val = assemble(theta * dot(self.b_exps[i], self.b_exps[j]) * dx)
                M[i, j] = val
                M[j, i] = val
                
        try:
            # Solve for the optimal control coefficients 'a'
            u_optimal = np.linalg.solve(M, p)
            return np.linalg.norm(u_optimal-u_coeffs) < 1e-4
        except np.linalg.LinAlgError:
            # If theta is zero in large regions, M might be singular
            print("Warning: Matrix M is singular. Check theta distribution.")
            return np.zeros(n)
        
    # def check_complementary_slackness(self, theta_init, theta_final, tol=1e-5):
    #     """
    #     Checks the condition: lambda * (Final_Error - r * Initial_Error) == 0
    #     """
    #     # 1. Reuse your logic to get the 'gap' (how close we are to the target)
    #     e_initial = theta_init - Constant(self.theta_bar)
    #     c0 = sqrt(assemble((dot(e_initial, e_initial) + dot(grad(e_initial), grad(e_initial))) * dx))
        
    #     e_final = theta_final - Constant(self.theta_bar)
    #     ef = sqrt(assemble((dot(e_final, e_final) + dot(grad(e_final), grad(e_final))) * dx))
        
    #     # The constraint term: (Ef - r*C0)
    #     constraint_gap = ef - (self.r * c0)
    #     print("constraint gap", constraint_gap)
        
    #     # 2. Compute the slackness product
    #     slackness = self.lmbda * constraint_gap
    #     # print("Slackness Value: ", slackness)
        
    #     # 3. We check the absolute value of the product
    #     is_satisfied = abs(slackness) < tol
        
    #     return is_satisfied, slackness

    def check_complementary_slackness(self, theta_init, theta_final, tol=1e-5):
        # 1. Use the built-in FEniCS norm for H1: sqrt( ||e||^2_L2 + ||grad(e)||^2_L2 )
        e_initial = project(theta_init - Constant(self.theta_bar), theta_init.function_space())
        c0 = norm(e_initial, 'H1')
        
        e_final = project(theta_final - Constant(self.theta_bar), theta_final.function_space())
        ef = norm(e_final, 'H1')
        
        # The constraint: g(u) = ef - r*c0
        constraint_gap = ef - (self.r * c0)
        
        # 2. Check the product
        slackness_product = self.lmbda * constraint_gap
        
        # 3. KKT validation
        # Complementary slackness: product is zero
        product_zero = abs(slackness_product) < tol
        
        # Primal Feasibility: constraint is not violated (g <= 0)
        # We use a small epsilon to allow for floating point noise
        is_feasible = constraint_gap <= tol 
        
        # Dual Feasibility: multiplier is non-negative
        lambda_positive = self.lmbda >= -1e-9

        slackness_satisfied = product_zero and is_feasible and lambda_positive
        
        print(f"Gap: {constraint_gap:.2e} | Lambda: {self.lmbda:.2e} | Product: {slackness_product:.2e}")
        
        return slackness_satisfied, constraint_gap


    #My approach using naive gradient descent 
    def optimize_with_gradient_descent(self, initial_u, lr, theta_initial, max_iterations,r,tol=1e-4):
        """Main optimization loop."""
        u_coeffs = list(initial_u)

        # 1. Compute Initial Error (C0)
        e_initial = theta_initial - Constant(self.theta_bar)
        # Using H1 norm as you did before: ||e||^2 + ||grad(e)||^2
        c0_sq = assemble((dot(e_initial, e_initial) + dot(grad(e_initial), grad(e_initial))) * dx)
        c0 = float(c0_sq)**0.5

        # # 2. Compute Final Error (Ef)
        # e_final = theta_final - Constant(self.theta_bar)
        # ef_sq = assemble((dot(e_final, e_final) + dot(grad(e_final), grad(e_final))) * dx)
        # ef = sqrt(ef_sq)
        

        #Still need to replace this for loop with a while eloop that goes until the following three things are satisfied
        # 1. The final time constraint is satisfied (function written)
        # 2. The optimality condition is satisfied (Equation 3.8 in the paper)
        # 3. The complementary slackness condition is satisfied   
        print("About to start training loop")
        # breakpoint()
        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            # breakpoint()
            # 1. Forward Solve
            print("Running Forward Solve...")
            theta_traj = solve_state(u_coeffs, self.b_exps, self.T, self.num_steps, self.V, i)

            #Check progress of theta
            theta_final = theta_traj[-1]

            #Compute final error 
            e_final = theta_final - Constant(self.theta_bar)
            ef_sq = assemble((dot(e_final, e_final) + dot(grad(e_final), grad(e_final))) * dx)
            ef = np.sqrt(ef_sq)

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

            # # 5. Convergence Check
            # if grad_norm < tol:
            #     print("Optimality condition satisfied.")
            #     break

            # 6. Update and Plot
            u_coeffs = [c - lr * (g + self.lmbda * g) for c, g in zip(u_coeffs, grad_vec)]
            plot_forward(theta_traj[0], theta_traj[-1], self.T, i)

            #Update lambda 
            # breakpoint()
            print("About to update lambda")
            self.lmbda = max(0, self.lmbda + self.lr_lmbda * (ef - r * c0))

            #Check final time constraint
            # print("about to check final time constraint")
            final_time_constraint_satisfied = self.final_time_constraint_satisfied(theta_initial, theta_final)

            #Check Optimality condition
            # print("About to check optimality constraint")
            optimality_satisfied = self.optimality_condition_satisfied(theta_traj[-1], rho_traj, u_coeffs)

            #Check complementary slackness condition
            # print("About to check complementary slackness condition")
            slackness_satisfied, constraint_gap = self.check_complementary_slackness(theta_traj[0], theta_traj[-1], self.r)

            if final_time_constraint_satisfied and optimality_satisfied and slackness_satisfied:
                print("All KKT conditions satisfied. Optimization converged.")
                break
            else:
                print("Current KKT Status:")
                print(f"  Final Time Constraint Satisfied: {final_time_constraint_satisfied}")
                print(f"  Optimality Condition Satisfied: {optimality_satisfied}")
                print(f"  Complementary Slackness Satisfied: {slackness_satisfied} (Value: {constraint_gap:.6f})")
                # print("Reached the maximum number of iterations without satisfying all KKT conditions.")
            if i==max_iterations-1 and not (final_time_constraint_satisfied and optimality_satisfied and slackness_satisfied):
                print("Reached the maximum number of iterations without satisfying all KKT conditions.")

        return u_coeffs
    
    def compute_adaptive_alpha(alpha_k, mixing_metric, epsilon_1=1e-3, min_alpha=0.05):
        """
        Computes alpha_{k+1} based on the convergence/mixing metric relative to a threshold.
        
        Parameters:
        -----------
        alpha_k : float
            The alpha parameter from the current iteration (k).
        mixing_metric : float
            The computed performance/error metric (e.g., \mu^{k+1}) from the current step.
        epsilon_1 : float, optional
            The tolerance threshold for the metric. Default is 1e-3.
        min_alpha : float, optional
            The lower bound floor for alpha to prevent it from decaying to zero. Default is 0.05.
            
        Returns:
        --------
        float
            The updated alpha_{k+1} for the next iteration.
        """
        # Check if the mixing metric has dropped below the threshold
        if mixing_metric < epsilon_1:
            # Decay alpha by 50%, but ensure it doesn't drop below the minimum floor
            alpha_next = max(0.5 * alpha_k, min_alpha)
        else:
            # Maintain the current alpha
            alpha_next = alpha_k
            
        return alpha_next
    
    def compute_adaptive_beta(J_val, mu_next, epsilon_1=1e-3):
        """
        Computes beta_{k+1} exactly as defined in the paper's numerical experiments section.
        
        Parameters:
        -----------
        J_val : float
            The current cost functional evaluation J(u^k, \theta^{k+1}).
        mu_next : float
            The mixing metric \mu^{k+1} evaluated at the current step.
        epsilon_1 : float
            The threshold tolerance (same parameter used in the alpha update).
            
        Returns:
        --------
        float
            The exact adaptive beta_{k+1} scaling factor for the next step.
        """
        # Case 3: If mixing threshold is successfully satisfied
        if mu_next < epsilon_1:
            return 250.0
            
        # Cases 1 and 2: If mixing threshold is not yet met (\mu^{k+1} >= \varepsilon_1)
        else:
            if J_val >= 1.0:
                # Dynamic balancing scaling ratio
                return (2.0 * J_val) / mu_next
            else:
                # Constant lower bound alternative for smaller cost regimes
                return 100.0
    

    #STILL WANT TO ADD FORWARD SOLVE PLOTS TO SEE HOW THE FINAL RESULTS CHANGE AFTER EACH ITERATION 
    #Algorithm 1 in Cellular Flow Control Design for Mixing Based on the Least Action Principle 
    def optimize_with_fixed_point(self, initial_u, theta_initial, max_iterations, r, relaxation=0.5, tol=1e-4):
        """
        Optimization loop using Fixed-Point Iteration (FPI).
        Replaces gradient descent with a direct solve for the optimality condition.
        """

        dx = Measure('dx', domain=self.mesh)
        u_coeffs = list(initial_u)
        
        # 1. Compute Initial Error (C0) for constraint tracking
        e_init = project(theta_initial - Constant(self.theta_bar), self.V)
        c0 = norm(e_init, 'H1')

        print(f"Starting Fixed-Point Iteration. Initial C0: {c0:.6f}")

        for i in range(max_iterations):
            print(f"\n--- FPI Iteration {i+1} ---")

            # 2. Forward Solve
            theta_traj = solve_state(u_coeffs, self.b_exps, self.T, self.num_steps, self.V, i)
            theta_final = theta_traj[-1]
            plot_forward(theta_traj[0], theta_traj[-1], self.T, i)

            # 3. Terminal & Backward Adjoint Solve
            rho_T = solve_terminal_adjoint(theta_final, self.theta_bar, self.V)
            u_eff = self.get_velocity_vector(u_coeffs)
            rho_traj_rev = solve_adjoint_backwards(rho_T, u_eff, self.V, self.T, self.num_steps)
            rho_traj = rho_traj_rev[::-1]

            # 4. Construct Matrix M and Vector P for Fixed-Point Update
            n = self.num_coeffs
            M = np.zeros((n, n))
            p = np.zeros(n)
            
            # Trajectory integration over time
            for n_idx in range(1, len(theta_traj)):
                # Accumulate driving forces from the Adjoint-State interaction
                for row in range(n):
                    term_p = rho_traj[n_idx] * dot(self.b_exps[row], grad(theta_traj[n_idx])) * dx
                    p[row] += assemble(term_p) * self.dt
                
                # Accumulate energy costs
                for row in range(n):
                    for col in range(row, n):
                        term_m = dot(self.b_exps[row], self.b_exps[col]) * dx
                        val = assemble(term_m)
                        M[row, col] += val * self.dt
                        M[col, row] += val * self.dt

            # 5. Solve for u_optimal
            try:
                # FIX 1 & 2: Use negative sign AND scale by (1.0 + self.lmbda)
                u_optimal_step = np.linalg.solve(M + 1e-6 * np.eye(n), -p )

                alpha = 0.1
                u_old = np.array(u_coeffs)
                u_new = (1 - alpha) * u_old + alpha * u_optimal_step

                u_coeffs = u_new.tolist()
            except np.linalg.LinAlgError:
                print("Warning: Matrix M is singular. Keeping previous coefficients.")

            # # 4. Construct Matrix M and Vector P for Fixed-Point Update
            # # Based on optimality system: M * u_new = P
            # n = self.num_coeffs
            # M = np.zeros((n, n))
            # p = np.zeros(n)
            
            # # We integrate over the trajectory to find the 'Optimal' u for this state
            # for n_idx in range(1, len(theta_traj)):
            #     # Update P: Sensitivity of the state to the basis functions
            #     # Usually: integral( rho * (b_i . grad(theta)) )
            #     for row in range(n):
            #         term_p = rho_traj[n_idx] * dot(self.b_exps[row], grad(theta_traj[n_idx])) * dx
            #         p[row] += assemble(term_p) * self.dt
                
            #     # Update M: The 'Control Energy' matrix 
            #     # If the paper uses alpha*||u||^2, M is usually alpha * Identity or mass matrix
            #     for row in range(n):
            #         for col in range(row, n):
            #             term_m = dot(self.b_exps[row], self.b_exps[col]) * dx
            #             val = assemble(term_m)
            #             M[row, col] += val * self.dt
            #             M[col, row] += val * self.dt

            # # 5. Solve for u_optimal
            # try:
            #     # Regularize slightly to prevent singular matrix issues
            #     # 1. Calculate the 'optimal' step (M^-1 * p)
            #     # We use linalg.solve because it is numerically more stable than inverting M
            #     u_optimal_step = np.linalg.solve(M + 1e-8 * np.eye(n), -p*(1+self.lmbda))

            #     # 2. Assign your relaxation factor to alpha
            #     alpha = relaxation

            #     # 3. Apply the formula: u_new = (1 - alpha) * u_old + alpha * u_optimal_step
            #     u_old = np.array(u_coeffs)
            #     u_new = (1 - alpha) * u_old + alpha * u_optimal_step

            #     # 4. Update the coefficients
            #     u_coeffs = u_new.tolist()
            # except np.linalg.LinAlgError:
            #     print("Warning: Matrix M is singular. Keeping previous coefficients.")

            # 7. Update Lambda (Augmented Lagrangian)
            ef = norm(project(theta_final - Constant(self.theta_bar), self.V), 'H1')
            # 1. Compute current squared H1 norms
            # Note: if you already have ef_sq and c0_sq from earlier in the loop, use those!
            e_initial = theta_initial - Constant(self.theta_bar)
            e_final = theta_final - Constant(self.theta_bar)
            ef_sq = assemble((inner(e_final, e_final) + inner(grad(e_final), grad(e_final))) * dx)
            c0_sq = assemble((inner(e_initial, e_initial) + inner(grad(e_initial), grad(e_initial))) * dx)

            # 2. Implement the formula exactly: max(lmbda + beta * (ef^2 - r^2 * c0^2), 0)
            # Here, self.lr_lmbda acts as your 'beta'
            self.lmbda = max(0, self.lmbda + self.lr_lmbda * (ef_sq - (self.r**2 * c0_sq)))

            # 8. KKT & Convergence Checks
            final_time_satisfied = self.final_time_constraint_satisfied(theta_initial, theta_final)
            optimality_satisfied = self.optimality_condition_satisfied(theta_final, rho_traj, u_coeffs)
            slackness_satisfied, gap = self.check_complementary_slackness(theta_initial, theta_final, r)

            print(f"Current Coeffs: {[round(c, 4) for c in u_coeffs]}")
            print(f"Gap: {gap:.4e} | Lambda: {self.lmbda:.4e}")

            if final_time_satisfied and optimality_satisfied and slackness_satisfied:
                print("Convergence reached via Fixed-Point Iteration.")
                break

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
# self.T = T
#         self.num_steps = num_steps (got it)
#         self.dt = T / num_steps (got it)
#         self.theta_bar = theta_bar (got it)
#         self.b_exps = b_exps (got it)
#         self.lr_lambda = lr_lambda (got it)
#         self.r = r (got it)
#         self.lmbda = 0.0
        
#         # FEniCS Mesh and Function Space
#         self.mesh = UnitSquareMesh(mesh_size, mesh_size)
#         self.V = FunctionSpace(self.mesh, "Lagrange", 1)
#         self.num_coeffs = len(b_exps)
optimizer = AdjointOptimizer(
    mesh_size=64, 
    T=1.0, 
    num_steps=200, 
    b_exps=basis_functions, 
    lmbda = 0.0,
    lr_lmbda = 0.1,
    theta_bar=1.0,
    r=0.1
)

# 3. Run optimization 
#I should really just think about the initial_u as an initial guess at the coefficients to mix well, 
# and then the optimizer will adjust them to minimize the cost function.

theta_init_expr = Expression("tanh((2*x[1]-1)/0.2)+1", degree=2)
theta_initial_func = interpolate(theta_init_expr, optimizer.V)


print("Hello?")
# breakpoint()
final_coeffs = optimizer.optimize_with_fixed_point(
    initial_u=[0.0, 1.0, 1.0, 1.0], 
    theta_initial = theta_initial_func,
    max_iterations=200, 
    r=0.3   
)