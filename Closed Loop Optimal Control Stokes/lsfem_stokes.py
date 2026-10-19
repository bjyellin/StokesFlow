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



from ufl import sin, cos, pi, as_vector, div, curl

class LSFEMStokesVorticityPressure:
    def __init__(self, mu, problem="cavity"):
        self.mu = mu
        self.problem = problem  # "mms" or "cavity"

    def u_ex(self, x):
        # Kept for MMS verification, or can be bypassed if problem == "cavity"
        sinx = sin(pi * x[0])
        siny = sin(pi * x[1])
        cosx = cos(pi * x[0])
        cosy = cos(pi * x[1])
        c_factor = 2 * pi * sinx * siny
        return c_factor * as_vector((cosy * sinx, -cosx * siny))

    def p_ex(self, x):
        return sin(2 * pi * x[0]) * sin(2 * pi * x[1])

    def omega_ex(self, x):
        u = self.u_ex(x)
        return u[1].dx(0) - u[0].dx(1)

    def source_f1(self, x):
        if self.problem == "cavity":
            return as_vector((0.0 * x[0], 0.0 * x[0]))
        
        mu = self.mu
        omega = self.omega_ex(x)
        p = self.p_ex(x)
        curl_omega = as_vector((omega.dx(1), -omega.dx(0)))
        return mu * curl_omega + grad(p)

    def source_f2(self, x):
        if self.problem == "cavity":
            return 0.0 * x[0]
        
        u = self.u_ex(x)
        return div(u)

    def source_f3(self, x):
        if self.problem == "cavity":
            return 0.0 * x[0]
        
        u = self.u_ex(x)
        omega = self.omega_ex(x)
        return curl(u) - omega

Stokes_Problem = LSFEMStokesVorticityPressure(mu=1.0)
Stokes_Problem.problem = "cavity"  #Options are mms or cavity

problem = Stokes_Problem.problem

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

# def create_linear_form(mu, W):
#     """
#     Creates the weighted LSFEM linear form corresponding to F_h on page 885.
#     """
#     xi, q, v = TestFunctions(W)
#     domain = W.mesh
#     dx = Measure("dx", domain=domain)
#     h = CellDiameter(domain)

#     x = SpatialCoordinate(domain)
#     f1 = Stokes_Problem.source_f1(x)
#     f2 = Stokes_Problem.source_f2(x)
#     f3 = Stokes_Problem.source_f3(x)

#     R_mom_test = mu * as_vector((xi.dx(1), -xi.dx(0))) + grad(q)
#     R_div_test = div(v)
#     R_curl_test = curl(v) - xi

#     L = (inner(f1, R_mom_test) + 
#          (1.0 / (h**2)) * inner(f2, R_div_test) + 
#          (1.0 / (h**2)) * inner(f3, R_curl_test)) * dx

#     return fem.form(L)

def create_linear_form(mu, W):
    """
    Creates the weighted LSFEM linear form corresponding to F_h on page 885.
    """
    xi, q, v = TestFunctions(W)
    domain = W.mesh
    dx = Measure("dx", domain=domain)
    h = CellDiameter(domain)

    R_mom_test = mu * as_vector((xi.dx(1), -xi.dx(0))) + grad(q)
    R_div_test = div(v)
    R_curl_test = curl(v) - xi

    # Use Constant zero vectors for cavity so UFL doesn't optimize out the test functions
    if Stokes_Problem.problem == "cavity":
        f1 = Constant(domain, (0.0, 0.0))
        f2 = Constant(domain, 0.0)
        f3 = Constant(domain, 0.0)
    else:
        x = SpatialCoordinate(domain)
        f1 = Stokes_Problem.source_f1(x)
        f2 = Stokes_Problem.source_f2(x)
        f3 = Stokes_Problem.source_f3(x)

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

def create_cavity_velocity_bcs(velocity_subspace, U_0=1.0):
    """
    Creates the specific Dirichlet boundary conditions for a 2D lid-driven cavity
    on a domain [0, 1] x [0, 1].
    """
    V_collapsed, dof_map = velocity_subspace.collapse()
    
    # 1. Define geometric conditions for boundaries
    def top_wall(x):
        return np.isclose(x[1], 1.0)

    def other_walls(x):
        side_or_bottom = np.isclose(x[1], 0.0) | np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
        return side_or_bottom & ~np.isclose(x[1], 1.0)   # exclude the top row

    # 2. Setup zero function for stationary walls
    g_zero = fem.Function(V_collapsed)
    g_zero.x.array[:] = 0.0
    
    # 3. Setup moving lid expression (u = U_0, v = 0)
    # V_collapsed is a 2D vector function space, so we can interpolate a vector expression
    g_lid = fem.Function(V_collapsed)
    # Using an expression or setting array values directly depending on components
    # Assuming standard vector element where component 0 is u and component 1 is v:
    # A simple way for a constant lid velocity:
    g_lid.sub(0).interpolate(lambda x: np.full(x.shape[1], U_0))
    g_lid.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))
    g_lid.x.scatter_forward()

    # 4. Locate DOFs geometrically
    dofs_stationary = fem.locate_dofs_geometrical((velocity_subspace, V_collapsed), other_walls)
    dofs_lid = fem.locate_dofs_geometrical((velocity_subspace, V_collapsed), top_wall)

    # 5. Create and return the list of Dirichlet BCs
    bc_stationary = fem.dirichletbc(g_zero, dofs_stationary, velocity_subspace)
    bc_lid = fem.dirichletbc(g_lid, dofs_lid, velocity_subspace)
    
    return [bc_stationary, bc_lid] 

def setup_and_solve_stokes(problem, N=64):
    print(f"Initializing mesh ({N}x{N}) for Velocity-Vorticity-Pressure LSFEM...")
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, mesh.CellType.triangle)

    # Define finite element spaces (e.g., P2 for velocity, P1 for vorticity and pressure)
    element_omega = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    element_p     = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    element_u     = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(2,))

    mixed_element = basix.ufl.mixed_element([element_omega, element_p, element_u])
    W = fem.functionspace(domain, mixed_element)

    mu = 1.0

    print("Assembling weighted LSFEM system...")
    a_form = create_bilinear_form(mu, W)
    L_form = create_linear_form(mu, W)

    # Velocity BCs on subspace 2, pressure gauge pin on subspace 1
    if problem == "cavity":
        velocity_bcs = create_cavity_velocity_bcs(W.sub(2), U_0=1.0)
    else:
        velocity_bcs = create_velocity_bc(W.sub(2), g_expr=None)

    pressure_bcs = create_pressure_pin(W.sub(1))
    bcs = velocity_bcs + pressure_bcs

    A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = dolfinx.fem.petsc.assemble_vector(L_form)
    dolfinx.fem.petsc.apply_lifting(b, [a_form], [bcs])
    print("RHS vector min/max:", b.min()[1], b.max()[1])
    
    # breakpoint()
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
    print("Solution vector min/max:", w_solution.x.petsc_vec.min(), w_solution.x.petsc_vec.max())
    # breakpoint()
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

    if problem == 'mms':
    
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "p_h"})
        plotter.view_xy()
        plotter.add_title("Numerical Pressure", font_size=14, color="black")
        
        plotter.subplot(0, 1)
        plotter.add_mesh(grid_true, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "p_ex"})
        plotter.view_xy()
        plotter.add_title("Exact Pressure", font_size=14, color="black")
        
        plotter.subplot(0, 2)
        plotter.add_mesh(grid_diff, show_edges=True, cmap="coolwarm", scalar_bar_args={**scalar_bar_args, "title": "Error"})
        plotter.view_xy()
        plotter.add_title("Absolute Difference", font_size=14, color="black")
        
        plotter.screenshot("pressure_comparison.png")
        print("Pressure comparison plot saved to pressure_comparison.png")

    if problem == 'cavity':
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "p_h"})
        plotter.view_xy()
        plotter.add_title("Numerical Pressure", font_size=14, color="black")
        plotter.screenshot("cavity_pressure.png")
        print("Pressure plot saved to cavity_pressure.png")

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

    if problem == 'mms':

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

    if problem == 'cavity':
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "|u_h|"})
        plotter.view_xy()
        plotter.add_title("Numerical Velocity", font_size=14, color="black")

        plotter.screenshot("cavity_velocity")

def plot_vorticity_comparison(w_solution, W, stokes_problem):
    domain = W.mesh
    x = SpatialCoordinate(domain)

    # Vorticity is at subspace 0 (scalar field)
    Q_omega, dofs_omega = W.sub(0).collapse()
    omega_h = fem.Function(Q_omega)
    omega_h.x.array[:] = w_solution.x.array[dofs_omega]

    cells, types, x_pts = plot.vtk_mesh(Q_omega)
    grid_num = pyvista.UnstructuredGrid(cells, types, x_pts)
    grid_num.point_data["Vorticity"] = omega_h.x.array
    grid_num.set_active_scalars("Vorticity")

    # Only set up exact solution grids if running MMS
    if problem == 'mms':
        omega_expr = stokes_problem.omega_ex(x)
        expression = fem.Expression(omega_expr, Q_omega.element.interpolation_points)
        omega_true = fem.Function(Q_omega)
        omega_true.interpolate(expression)

        grid_true = pyvista.UnstructuredGrid(cells, types, x_pts)
        grid_true.point_data["Vorticity"] = omega_true.x.array
        grid_true.set_active_scalars("Vorticity")

        grid_diff = pyvista.UnstructuredGrid(cells, types, x_pts)
        grid_diff.point_data["Difference"] = np.abs(grid_num.point_data["Vorticity"] - grid_true.point_data["Vorticity"])
        grid_diff.set_active_scalars("Difference")

    scalar_bar_args = {"vertical": True, "position_x": 0.82, "position_y": 0.1, "width": 0.08, "height": 0.8}
    plotter = pyvista.Plotter(shape=(1, 3 if problem == 'mms' else 1), off_screen=True, window_size=[1800 if problem == 'mms' else 600, 600])

    if problem == 'mms':
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "omega_h"})
        plotter.view_xy()
        plotter.add_title("Numerical Vorticity", font_size=14, color="black")

        plotter.subplot(0, 1)
        plotter.add_mesh(grid_true, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "omega_ex"})
        plotter.view_xy()
        plotter.add_title("Exact Vorticity", font_size=14, color="black")

        plotter.subplot(0, 2)
        plotter.add_mesh(grid_diff, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "Error"})
        plotter.view_xy()
        plotter.add_title("Absolute Difference", font_size=14, color="black")

        plotter.screenshot("vorticity_comparison.png")
        print("Vorticity comparison plot saved to vorticity_comparison.png")

    elif problem == 'cavity':
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "omega_h"})
        plotter.view_xy()
        plotter.add_title("Numerical Vorticity (Cavity)", font_size=14, color="black")

        plotter.screenshot("cavity_vorticity.png")
        print("Cavity vorticity plot saved to cavity_vorticity.png")

def plot_cavity_centerlines(w_solution, W):
    domain = W.mesh
    
    # Extract velocity subspace (subspace 2)
    R, dofs = W.sub(2).collapse()
    u_h = fem.Function(R)
    u_h.x.array[:] = w_solution.x.array[dofs]

    # Convert to a PyVista unstructured grid
    cells, types, x_pts = plot.vtk_mesh(R)
    grid = pyvista.UnstructuredGrid(cells, types, x_pts)
    
    # Attach velocity components to the grid point data
    u_array = u_h.x.array.reshape(-1, 2)
    grid.point_data["u_vel"] = u_array[:, 0]
    grid.point_data["v_vel"] = u_array[:, 1]

    # Sample over vertical centerline (x = 0.5, from y = 0.0 to 1.0)
    line_vert = grid.sample_over_line([0.5, 0.0, 0.0], [0.5, 1.0, 0.0], resolution=150)
    y_coords = line_vert.points[:, 1]
    u_profile = line_vert.point_data["u_vel"]

    # Sample over horizontal centerline (y = 0.5, from x = 0.0 to 1.0)
    line_horiz = grid.sample_over_line([0.0, 0.5, 0.0], [1.0, 0.5, 0.0], resolution=150)
    x_coords = line_horiz.points[:, 0]
    v_profile = line_horiz.point_data["v_vel"]

    # Create publication-style matplotlib comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot u velocity along vertical centerline
    axes[0].plot(u_profile, y_coords, 'b-', lw=2, label='LSFEM Stokes ($N=32$)')
    axes[0].set_xlabel('Horizontal Velocity $u$', fontsize=12)
    axes[0].set_ylabel('Vertical Coordinate $y$', fontsize=12)
    axes[0].set_title('Vertical Centerline ($x = 0.5$)', fontsize=14)
    axes[0].grid(True, ls='--', alpha=0.6)
    axes[0].legend()

    # Plot v velocity along horizontal centerline
    axes[1].plot(x_coords, v_profile, 'r-', lw=2, label='LSFEM Stokes ($N=32$)')
    axes[1].set_xlabel('Horizontal Coordinate $x$', fontsize=12)
    axes[1].set_ylabel('Vertical Velocity $v$', fontsize=12)
    axes[1].set_title('Horizontal Centerline ($y = 0.5$)', fontsize=14)
    axes[1].grid(True, ls='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cavity_centerlines.png", dpi=300)
    print("Centerline profile plot saved to cavity_centerlines.png")
    plt.close()


# def analyze_convergence():
#     comm = MPI.COMM_WORLD
#     # Define a sequence of grid sizes to test
#     grid_sizes = [16, 24, 32, 48]
    
#     hs = []
#     u_l2_errors = []
#     u_h1_errors = []
#     p_l2_errors = []
    
#     for N in grid_sizes:
#         print(f"\n--- Running for N = {N} ---")
#         w_solution, W = setup_and_solve_stokes(problem, N=N)
#         u_L2, u_H1, p_L2 = compute_errors(w_solution, W, Stokes_Problem, W.mesh, comm)
        
#         h = 1.0 / N
#         hs.append(h)
#         # INDENTATION FIX: These must be inside the loop!
#         u_l2_errors.append(u_L2)
#         u_h1_errors.append(u_H1)
#         p_l2_errors.append(p_L2)
        
#     hs = np.array(hs)
    
#     # Plotting the convergence curves on a log-log scale
#     plt.figure(figsize=(8, 6))
#     plt.loglog(hs, u_l2_errors, 'o-', label='Velocity $L^2$ error')
#     plt.loglog(hs, u_h1_errors, 's-', label='Velocity $H^1$ error')
#     plt.loglog(hs, p_l2_errors, 'd-', label='Pressure $L^2$ error')
    
#     # Reference lines to visually verify slopes (e.g., O(h^1) and O(h^2))
#     plt.loglog(hs, hs**1 * (u_l2_errors[0] / hs[0]**1), 'k--', alpha=0.5, label='$O(h)$')
#     plt.loglog(hs, hs**2 * (u_l2_errors[0] / hs[0]**2), 'k:', alpha=0.5, label='$O(h^2)$')
    
#     plt.xlabel('Grid size $h$ (log scale)')
#     plt.ylabel('Error norm (log scale)')
#     plt.title('LSFEM Stokes Convergence Rates')
#     plt.grid(True, which="both", ls="--")
#     plt.legend()
#     plt.gca().invert_xaxis() # Smaller h (finer mesh) goes to the right
    
#     plt.savefig("convergence_plot.png", dpi=300)
#     print("\nConvergence plot saved to convergence_plot.png")
#     plt.close()

def analyze_convergence():
    grid_sizes = [16, 24, 32, 48, 64]
    hs = []
    min_u_values = []
    
    for N in grid_sizes:
        print(f"\n--- Running for N = {N} ---")
        w_solution, W = setup_and_solve_stokes(problem, N=N)
        
        # Extract velocity field
        R, dofs = W.sub(2).collapse()
        u_h = fem.Function(R)
        u_h.x.array[:] = w_solution.x.array[dofs]
        
        # Sample along vertical centerline (x = 0.5) using PyVista
        cells, types, x_pts = plot.vtk_mesh(R)
        grid = pyvista.UnstructuredGrid(cells, types, x_pts)
        u_array = u_h.x.array.reshape(-1, 2)
        grid.point_data["u_vel"] = u_array[:, 0]
        
        line_vert = grid.sample_over_line([0.5, 0.0, 0.0], [0.5, 1.0, 0.0], resolution=200)
        u_profile = line_vert.point_data["u_vel"]
        
        # Track the minimum u value (the core vortex backflow peak)
        min_u = np.min(u_profile)
        hs.append(1.0 / N)
        min_u_values.append(min_u)
        print(f"N={N} -> Min centerline u: {min_u:.5f}")

    hs = np.array(hs)
    min_u_values = np.array(min_u_values)
    
    # Plotting grid asymptotic convergence of the QoI
    plt.figure(figsize=(8, 6))
    plt.plot(hs, min_u_values, 'o-', lw=2, color='b', label='Min Centerline $u$')
    plt.xlabel('Grid spacing $h$ ($1/N$)', fontsize=12)
    plt.ylabel('Core Vortex Backflow Velocity ($u_{\min}$)', fontsize=12)
    plt.title('Lid-Driven Cavity Grid Independence Study', fontsize=14)
    plt.grid(True, ls='--')
    plt.legend()
    plt.gca().invert_xaxis() # Finer grids (smaller h) on the right
    
    plt.savefig("cavity_grid_convergence.png", dpi=300)
    print("\nGrid independence plot saved to cavity_grid_convergence.png")
    plt.close()
    
if __name__ == "__main__":
    w_solution, W = setup_and_solve_stokes(problem, N=32)
    if problem == "mms":
        plot_pressure_comparison(w_solution, W, Stokes_Problem)
        plot_velocity_comparison(w_solution, W, Stokes_Problem)
        plot_vorticity_comparison(w_solution, W, Stokes_Problem)
    if problem == 'cavity':
        plot_cavity_centerlines(w_solution, W)
    analyze_convergence()

