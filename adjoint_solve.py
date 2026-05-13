import numpy as np
import matplotlib.pyplot as plt
import scipy
import dolfin 
from dolfin import *

def solve_terminal_adjoint(theta_curr, theta_bar, V, lam=1.0):
    # Set log level to hide INFO messages
    set_log_level(LogLevel.WARNING)
    """
    Computes the terminal condition for the adjoint state rho(T).
    
    Formula: rho(T) = -2 * lambda * A^-1 * (theta_curr - theta_bar)
    where A is the Helmholtz operator (-Delta + I).
    
    Parameters:
    -----------
    theta_curr : Function
        The concentration state at final time T.
    theta_bar  : Constant or Function
        The target/mean state (perfectly mixed).
    V          : FunctionSpace
        The space for the adjoint function.
    lam        : float
        Lagrange multiplier/weight parameter.
    """
    
    # 1. Define the Helmholtz problem for the Riesz representation (eta)
    # This maps the L2 difference into the H^-1 dual space
    eta = TrialFunction(V)
    v_test = TestFunction(V)
    
    # Bilinear form: <grad(eta), grad(v)> + <eta, v>
    a_dual = (dot(grad(eta), grad(v_test)) + eta * v_test) * dx
    # Linear form: <(theta - theta_bar), v>
    L_dual = (theta_curr - Constant(theta_bar)) * v_test * dx
    
    eta_sol = Function(V)
    
    # 2. Solve the linear system
    # We use a linear solver to find eta where A*eta = (theta - theta_bar)
    solve(a_dual == L_dual, eta_sol, 
          solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
    
    # 3. Scale and assign to rho_tf
    # rho(T) = -2 * lam * eta
    # We create a new Function and manipulate the vector directly for efficiency
    rho_tf = Function(V)
    rho_tf.vector()[:] = -2.0 * lam * eta_sol.vector()[:]
    
    return rho_tf


def solve_adjoint_backwards(rho_tf, u_eff, V, T, num_time_steps):
    dt = T / num_time_steps
    rho_n = Function(V)
    rho_n.assign(rho_tf)
    
    rho_traj = [rho_n] # Start with rho at T
    
    rho = TrialFunction(V)
    v = TestFunction(V)

    # ADJOINT OPERATOR: 
    # Forward was: (theta - theta_n)/dt * v + dot(u_eff, grad(theta)) * v
    # Adjoint is:  (rho_n - rho)/dt * v + dot(u_eff, grad(rho)) * v
    # Note: We solve for 'rho' which is the state at the PREVIOUS time step (t-dt)
    
    # Simple Adjoint Advection (Backwards Euler)
    F = ((rho_n - rho) / dt * v * dx + dot(u_eff, grad(rho)) * v * dx)
    a, L = lhs(F), rhs(F)

    for n in range(num_time_steps):
        rho_prev = Function(V)
        solve(a == L, rho_prev)
        
        rho_traj.append(rho_prev)
        rho_n.assign(rho_prev)

    return rho_traj # [rho_T, rho_{T-dt}, ..., rho_0]