from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pylab as plt
import numpy as np
import basix.ufl
import dolfinx.fem.petsc
from dolfinx import fem, mesh, plot
from dolfinx.fem import Constant
import pyvista

from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    TrialFunctions,
    TestFunctions,
    as_vector,
    as_tensor,
    cos,
    div,
    dx,
    grad,
    inner,
    pi,
    sin,
    curl,
    CellDiameter,
    Measure
)

class LSFEMStokesVorticityPressure:
    def __init__(self, mu):
        self.mu = mu

    def u_ex(self, x):
        sinx = sin(pi * x[0])
        siny = sin(pi * x[1])
        cosx = cos(pi * x[0])
        cosy = cos(pi * x[1])
        c_factor = 2 * pi * sinx * siny
        return c_factor * as_vector((cosy * sinx, -cosx * siny))

    def p_ex(self, x):
        return sin(2 * pi * x[0]) * sin(2 * pi * x[1])

    def omega_ex(self, x):
        # Exact vorticity in 2D: curl(u) = du1/dx0 - du0/dx1
        u = self.u_ex(x)
        return u[1].dx(0) - u[0].dx(1)

    def source_f1(self, x):
        # f1 = mu * curl(omega) + grad(p)
        mu = self.mu
        omega = self.omega_ex(x)
        p = self.p_ex(x)
        # curl of scalar omega in 2D is as_vector((omega.dx(1), -omega.dx(0)))
        curl_omega = as_vector((omega.dx(1), -omega.dx(0)))
        return mu * curl_omega + grad(p)

    def source_f2(self, x):
        # f2 = div(u) (equals 0 for incompressible Stokes)
        u = self.u_ex(x)
        return div(u)

    def source_f3(self, x):
        # f3 = curl(u) - omega (equals 0 by definition)
        u = self.u_ex(x)
        omega = self.omega_ex(x)
        return curl(u) - omega

Stokes_Problem = LSFEMStokesVorticityPressure(mu=1.0)

def create_pressure_pin(pressure_subspace):
    """
    Pins the pressure to 0.0 at the origin (0, 0) on the pressure subspace.
    """
    Q, dof_map = pressure_subspace.collapse()
    
    def at_origin(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    collapsed_dofs = fem.locate_dofs_geometrical(Q, at_origin)
    if len(collapsed_dofs) > 0:
        pin_dof_subspace = dof_map[collapsed_dofs[0:1]]
    else:
        pin_dof_subspace = dof_map[np.array([0], dtype=np.int32)]
        
    pin_dof_subspace = np.ascontiguousarray(pin_dof_subspace, dtype=np.int32)
    val = np.array(0.0, dtype=np.float64)
    
    return [fem.dirichletbc(val, pin_dof_subspace, pressure_subspace)]

def create_bilinear_form(mu, W):
    """
    Creates the weighted LSFEM bilinear form for the Velocity-Vorticity-Pressure
    system based on Page 885 of Deang & Gunzburger (1998).
    Subspaces: omega (0), p (1), u (2)
    """
    omega, p, u = TrialFunctions(W)
    xi, q, v = TestFunctions(W)

    domain = W.mesh
    dx = Measure("dx", domain=domain)
    h = CellDiameter(domain)

    # Residuals (Eq. 2.4 - 2.6)
    # R_mom: mu * curl(omega) + grad(p) - f1
    R_mom = mu * as_vector((omega.dx(1), -omega.dx(0))) + grad(p)
    R_mom_test = mu * as_vector((xi.dx(1), -xi.dx(0))) + grad(q)

    # R_div: div(u) - f2
    R_div = div(u)
    R_div_test = div(v)

    # R_curl: curl(u) - omega - f3
    R_curl = curl(u) - omega
    R_curl_test = curl(v) - xi

    # Weighted LSFEM functional bilinear form (Eq. J_h on page 885)
    a = (inner(R_mom, R_mom_test) + 
         (1.0 / (h**2)) * inner(R_div, R_div_test) + 
         (1.0 / (h**2)) * inner(R_curl, R_curl_test)) * dx

    return fem.form(a)

def create_linear_form(mu, W):
    """
    Creates the weighted LSFEM linear form corresponding to F_h on page 885.
    """
    xi, q, v = TestFunctions(W)
    domain = W.mesh
    dx = Measure("dx", domain=domain)
    h = CellDiameter(domain)

    x = SpatialCoordinate(domain)
    f1 = Stokes_Problem.source_f1(x)
    f2 = Stokes_Problem.source_f2(x)
    f3 = Stokes_Problem.source_f3(x)

    R_mom_test = mu * as_vector((xi.dx(1), -xi.dx(0))) + grad(q)
    R_div_test = div(v)
    R_curl_test = curl(v) - xi

    L = (inner(f1, R_mom_test) + 
         (1.0 / (h**2)) * inner(f2, R_div_test) + 
         (1.0 / (h**2)) * inner(f3, R_curl_test)) * dx

    return fem.form(L)

def create_velocity_bc(velocity_subspace, g_expr=None):
    """
    Creates Dirichlet boundary conditions for the velocity vector subspace.
    """
    V_collapsed, dof_map = velocity_subspace.collapse()
    domain = velocity_subspace.mesh
    
    g = fem.Function(V_collapsed)
    if g_expr is None:
        g.x.array[:] = 0.0
    else:
        g.interpolate(g_expr)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    bdry_facets = mesh.exterior_facet_indices(domain.topology)
    
    dofs = fem.locate_dofs_topological((velocity_subspace, V_collapsed), tdim - 1, bdry_facets)
    return [fem.dirichletbc(g, dofs, velocity_subspace)]

def setup_and_solve_stokes(N=32):
    print(f"Initializing mesh ({N}x{N}) for Velocity-Vorticity-Pressure LSFEM...")
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, mesh.CellType.triangle)

    # Define finite element spaces (e.g., P2 for velocity, P1 for vorticity and pressure)
    element_omega = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    element_p     = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    element_u     = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(2,))

    mixed_element = basix.ufl.mixed_element([element_omega, element_p, element_u])
    W = fem.functionspace(domain, mixed_element)

    mu = 1.0

    print("Assembling weighted LSFEM system...")
    a_form = create_bilinear_form(mu, W)
    L_form = create_linear_form(mu, W)

    # Velocity BCs on subspace 2, pressure gauge pin on subspace 1
    velocity_bcs = create_velocity_bc(W.sub(2), g_expr=None)
    pressure_bcs = create_pressure_pin(W.sub(1))
    bcs = velocity_bcs + pressure_bcs

    A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = dolfinx.fem.petsc.assemble_vector(L_form)
    dolfinx.fem.petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    print("Solving linear system with MUMPS...")
    w = fem.Function(W)
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    solver.solve(b, w.x.petsc_vec)

    if solver.getConvergedReason() < 0:
        raise RuntimeError("MUMPS solver failed to converge.")
    print("Solve complete!")

    w.x.scatter_forward()
    return w, W

def compute_errors(w_solution, W, stokes_problem, domain, comm):
    x = SpatialCoordinate(domain)
    dx = Measure("dx", domain=domain)
    # 1. Velocity Error (subspace 2)
    V_u, dofs_u = W.sub(2).collapse()
    u_h = fem.Function(V_u)
    u_h.x.array[:] = w_solution.x.array[dofs_u]
    u_ex_expr = fem.Expression(stokes_problem.u_ex(x), V_u.element.interpolation_points)
    u_exact = fem.Function(V_u)
    u_exact.interpolate(u_ex_expr)
    u_L2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(u_h - u_exact, u_h - u_exact) * dx)), op=MPI.SUM))
    u_H1_error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(u_h - u_exact), grad(u_h - u_exact)) * dx)), op=MPI.SUM)) + u_L2_error
    # 2. Pressure Error (subspace 1)
    V_p, dofs_p = W.sub(1).collapse()
    p_h = fem.Function(V_p)
    p_h.x.array[:] = w_solution.x.array[dofs_p]
    p_ex_expr = fem.Expression(stokes_problem.p_ex(x), V_p.element.interpolation_points)
    p_exact = fem.Function(V_p)
    p_exact.interpolate(p_ex_expr)
    p_L2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(p_h - p_exact, p_h - p_exact) * dx)), op=MPI.SUM))
    print(f"Velocity L2 Error: {u_L2_error:.5e} | H1 Error: {u_H1_error:.5e}")
    print(f"Pressure L2 Error: {p_L2_error:.5e}")
    # <-- Make sure this return statement is here!
    return u_L2_error, u_H1_error, p_L2_error


def plot_pressure_comparison(w_solution, W, stokes_problem):
    domain = W.mesh
    x = SpatialCoordinate(domain)
    
    # Pressure is at subspace 1
    Q, dofs = W.sub(1).collapse()
    p_h = fem.Function(Q)
    p_h.x.array[:] = w_solution.x.array[dofs]
    
    cells, types, x_pts = plot.vtk_mesh(Q)
    grid_num = pyvista.UnstructuredGrid(cells, types, x_pts)
    grid_num.point_data["Pressure"] = p_h.x.array
    grid_num.set_active_scalars("Pressure")
    
    p_expr = stokes_problem.p_ex(x)
    expression = fem.Expression(p_expr, Q.element.interpolation_points)
    p_true = fem.Function(Q)
    p_true.interpolate(expression)
    
    grid_true = pyvista.UnstructuredGrid(cells, types, x_pts)
    grid_true.point_data["Pressure"] = p_true.x.array
    grid_true.set_active_scalars("Pressure")
    
    grid_diff = pyvista.UnstructuredGrid(cells, types, x_pts)
    grid_diff.point_data["Difference"] = np.abs(p_h.x.array - p_true.x.array)
    grid_diff.set_active_scalars("Difference")
    
    scalar_bar_args = {"vertical": True, "position_x": 0.82, "position_y": 0.1, "width": 0.08, "height": 0.8}
    plotter = pyvista.Plotter(shape=(1, 3), off_screen=True, window_size=[1800, 600])
    
    plotter.subplot(0, 0)
    plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "p_h"})
    plotter.view_xy()
    plotter.add_title("Numerical Pressure", font_size=14, color="black")
    
    plotter.subplot(0, 1)
    plotter.add_mesh(grid_true, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "p_ex"})
    plotter.view_xy()
    plotter.add_title("Exact Pressure", font_size=14, color="black")
    
    plotter.subplot(0, 2)
    plotter.add_mesh(grid_diff, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "Error"})
    plotter.view_xy()
    plotter.add_title("Absolute Difference", font_size=14, color="black")
    
    plotter.screenshot("pressure_comparison.png")
    print("Pressure comparison plot saved to pressure_comparison.png")

def plot_velocity_comparison(w_solution, W, stokes_problem):
    domain = W.mesh
    x = SpatialCoordinate(domain)

    # Velocity is at subspace 2
    R, dofs = W.sub(2).collapse()
    u_h = fem.Function(R)
    u_h.x.array[:] = w_solution.x.array[dofs]

    cells, types, x_pts = plot.vtk_mesh(R)
    grid_num = pyvista.UnstructuredGrid(cells, types, x_pts)
    u_array_num = u_h.x.array.reshape(-1, 2)
    grid_num.point_data["Velocity"] = (u_array_num[:, 0]**2 + u_array_num[:, 1]**2)**0.5
    grid_num.set_active_scalars("Velocity")

    u_expr = stokes_problem.u_ex(x)
    expression = fem.Expression(u_expr, R.element.interpolation_points)
    u_true = fem.Function(R)
    u_true.interpolate(expression)

    grid_true = pyvista.UnstructuredGrid(cells, types, x_pts)
    u_array_true = u_true.x.array.reshape(-1, 2)
    grid_true.point_data["Velocity"] = (u_array_true[:, 0]**2 + u_array_true[:, 1]**2)**0.5
    grid_true.set_active_scalars("Velocity")

    grid_diff = pyvista.UnstructuredGrid(cells, types, x_pts)
    grid_diff.point_data["Difference"] = np.abs(grid_num.point_data["Velocity"] - grid_true.point_data["Velocity"])
    grid_diff.set_active_scalars("Difference")

    scalar_bar_args = {"vertical": True, "position_x": 0.82, "position_y": 0.1, "width": 0.08, "height": 0.8}
    plotter = pyvista.Plotter(shape=(1, 3), off_screen=True, window_size=[1800, 600])

    plotter.subplot(0, 0)
    plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "|u_h|"})
    plotter.view_xy()
    plotter.add_title("Numerical Velocity", font_size=14, color="black")

    plotter.subplot(0, 1)
    plotter.add_mesh(grid_true, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "|u_ex|"})
    plotter.view_xy()
    plotter.add_title("Exact Solution", font_size=14, color="black")

    plotter.subplot(0, 2)
    plotter.add_mesh(grid_diff, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "Error"})
    plotter.view_xy()
    plotter.add_title("Absolute Difference", font_size=14, color="black")

    plotter.screenshot("velocity_comparison.png")
    print("Velocity comparison plot saved to velocity_comparison.png")

def analyze_convergence():
    comm = MPI.COMM_WORLD
    # Define a sequence of grid sizes to test
    grid_sizes = [16, 24, 32, 48]
    
    hs = []
    u_l2_errors = []
    u_h1_errors = []
    p_l2_errors = []
    
    for N in grid_sizes:
        print(f"\n--- Running for N = {N} ---")
        w_solution, W = setup_and_solve_stokes(N=N)
        u_L2, u_H1, p_L2 = compute_errors(w_solution, W, Stokes_Problem, W.mesh, comm)
        
        h = 1.0 / N
        hs.append(h)
        # INDENTATION FIX: These must be inside the loop!
        u_l2_errors.append(u_L2)
        u_h1_errors.append(u_H1)
        p_l2_errors.append(p_L2)
        
    hs = np.array(hs)
    
    # Plotting the convergence curves on a log-log scale
    plt.figure(figsize=(8, 6))
    plt.loglog(hs, u_l2_errors, 'o-', label='Velocity $L^2$ error')
    plt.loglog(hs, u_h1_errors, 's-', label='Velocity $H^1$ error')
    plt.loglog(hs, p_l2_errors, 'd-', label='Pressure $L^2$ error')
    
    # Reference lines to visually verify slopes (e.g., O(h^1) and O(h^2))
    plt.loglog(hs, hs**1 * (u_l2_errors[0] / hs[0]**1), 'k--', alpha=0.5, label='$O(h)$')
    plt.loglog(hs, hs**2 * (u_l2_errors[0] / hs[0]**2), 'k:', alpha=0.5, label='$O(h^2)$')
    
    plt.xlabel('Grid size $h$ (log scale)')
    plt.ylabel('Error norm (log scale)')
    plt.title('LSFEM Stokes Convergence Rates')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.gca().invert_xaxis() # Smaller h (finer mesh) goes to the right
    
    plt.savefig("convergence_plot.png", dpi=300)
    print("\nConvergence plot saved to convergence_plot.png")
    plt.close()

    
if __name__ == "__main__":
    w_solution, W = setup_and_solve_stokes(N=32)
    plot_pressure_comparison(w_solution, W, Stokes_Problem)
    plot_velocity_comparison(w_solution, W, Stokes_Problem)
    analyze_convergence()

