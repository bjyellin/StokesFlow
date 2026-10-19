from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pylab as plt
import numpy as np
import basix.ufl
from scipy.interpolate import griddata
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
        self.weighted = 0 # 0 for unweighted, 1 for weighted least-squares functional

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
Stokes_Problem.problem = "mms"  #Options are mms or cavity
Stokes_Problem.weighted = 1  # 0 for unweighted, 1 for weighted least-squares functional

problem = Stokes_Problem.problem


def create_pressure_pin(pressure_subspace):

    Q, dof_map = pressure_subspace.collapse()

    def at_origin(x):
        return (
            np.isclose(x[0], 0.0) &
            np.isclose(x[1], 0.0)
        )

    collapsed_dofs = fem.locate_dofs_geometrical(
        Q,
        at_origin
    )

    if len(collapsed_dofs) == 0:
        raise RuntimeError(
            "Could not locate a pressure DOF at the origin."
        )

    # Map collapsed-space DOF back to the mixed-space DOF.
    mixed_dof = dof_map[collapsed_dofs[0]]

    mixed_dof = np.asarray(
        [mixed_dof],
        dtype=np.int32
    )

    return [
        fem.dirichletbc(
            PETSc.ScalarType(0.0),
            mixed_dof,
            pressure_subspace
        )
    ]

def create_bilinear_form(mu, W):
    """
    Create the unweighted LSFEM bilinear form for the
    velocity-vorticity-pressure first-order Stokes system.

    Subspaces:
        0 = omega
        1 = pressure
        2 = velocity
    """

    omega, p, u = TrialFunctions(W)
    xi, q, v = TestFunctions(W)

    domain = W.mesh
    dx = Measure("dx", domain=domain)

    # ---------------------------------------------------------
    # First-order residuals
    # ---------------------------------------------------------

    # Momentum:
    #
    #   mu * curl(omega) + grad(p) - f1
    #
    R_mom = (
        mu * as_vector(
            (omega.dx(1), -omega.dx(0))
        )
        + grad(p)
    )

    R_mom_test = (
        mu * as_vector(
            (xi.dx(1), -xi.dx(0))
        )
        + grad(q)
    )

    # Incompressibility:
    #
    #   div(u) - f2
    #
    R_div = div(u)
    R_div_test = div(v)

    # Vorticity definition:
    #
    #   curl(u) - omega - f3
    #
    R_curl = curl(u) - omega
    R_curl_test = curl(v) - xi

    # ---------------------------------------------------------
    # UNWEIGHTED least-squares functional
    # ---------------------------------------------------------

    if Stokes_Problem.weighted == 0:
        a = (
            inner(R_mom, R_mom_test)
            + inner(R_div, R_div_test)
            + inner(R_curl, R_curl_test)
        ) * dx

    else:
        h = CellDiameter(domain)

        a = (
            inner(R_mom, R_mom_test)
            + (1.0/h)**2 * inner(R_div, R_div_test)
            + (1.0/h)**2 * inner(R_curl, R_curl_test)
        ) * dx
    

    return fem.form(a)



def create_linear_form(mu, W):
    """
    Create the unweighted LSFEM linear form corresponding to
    the first-order Stokes least-squares functional.
    """

    xi, q, v = TestFunctions(W)

    domain = W.mesh
    dx = Measure("dx", domain=domain)

    R_mom_test = (
        mu * as_vector(
            (xi.dx(1), -xi.dx(0))
        )
        + grad(q)
    )

    R_div_test = div(v)

    R_curl_test = curl(v) - xi

    # ---------------------------------------------------------
    # Right-hand sides
    # ---------------------------------------------------------

    if Stokes_Problem.problem == "cavity":

        f1 = Constant(
            domain,
            PETSc.ScalarType((0.0, 0.0))
        )

        f2 = Constant(
            domain,
            PETSc.ScalarType(0.0)
        )

        f3 = Constant(
            domain,
            PETSc.ScalarType(0.0)
        )

    else:

        x = SpatialCoordinate(domain)

        f1 = Stokes_Problem.source_f1(x)
        f2 = Stokes_Problem.source_f2(x)
        f3 = Stokes_Problem.source_f3(x)

    # ---------------------------------------------------------
    # UNWEIGHTED linear form
    # ---------------------------------------------------------
    if Stokes_Problem.weighted == 0:
        L = (
            inner(f1, R_mom_test)
            + inner(f2, R_div_test)
            + inner(f3, R_curl_test)
        ) * dx
    
    if Stokes_Problem.weighted == 1:
        h = CellDiameter(domain)

        L = (
            inner(f1, R_mom_test)
            + (1.0/h)**2 * inner(f2, R_div_test)
            + (1.0/h)**2 * inner(f3, R_curl_test)
        ) * dx

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


def setup_and_solve_stokes(
    problem,
    N=64,
    degree_omega=1,
    degree_p=1,
    degree_u=1
):
    """
    Solve the velocity-vorticity-pressure LSFEM problem.

    Parameters
    ----------
    problem : str
        "mms" or "cavity"
    N : int
        Number of cells in each coordinate direction.
    degree_omega : int
        Polynomial degree for vorticity.
    degree_p : int
        Polynomial degree for pressure.
    degree_u : int
        Polynomial degree for velocity.
    """

    print(
        f"Initializing mesh ({N}x{N}), "
        f"P{degree_omega}/P{degree_p}/P{degree_u} "
        f"(omega/p/u)"
    )

    comm = MPI.COMM_WORLD

    domain = mesh.create_unit_square(
        comm,
        N,
        N,
        mesh.CellType.triangle
    )

    # ---------------------------------------------------------
    # Finite element spaces
    # ---------------------------------------------------------

    cell = domain.topology.cell_name()

    element_omega = basix.ufl.element(
        "Lagrange",
        cell,
        degree_omega
    )

    element_p = basix.ufl.element(
        "Lagrange",
        cell,
        degree_p
    )

    element_u = basix.ufl.element(
        "Lagrange",
        cell,
        degree_u,
        shape=(2,)
    )

    mixed_element = basix.ufl.mixed_element([
        element_omega,
        element_p,
        element_u
    ])

    W = fem.functionspace(domain, mixed_element)

    mu = 1.0

    print("Assembling LSFEM system...")

    a_form = create_bilinear_form(mu, W)
    L_form = create_linear_form(mu, W)

    # ---------------------------------------------------------
    # Boundary conditions
    # ---------------------------------------------------------

    if problem == "cavity":
        velocity_bcs = create_cavity_velocity_bcs(
            W.sub(2),
            U_0=1.0
        )
    else:
        velocity_bcs = create_velocity_bc(
            W.sub(2),
            g_expr=None
        )

    pressure_bcs = create_pressure_pin(W.sub(1))

    bcs = velocity_bcs + pressure_bcs

    # ---------------------------------------------------------
    # Assemble
    # ---------------------------------------------------------

    A = dolfinx.fem.petsc.assemble_matrix(
        a_form,
        bcs=bcs
    )
    A.assemble()

    b = dolfinx.fem.petsc.assemble_vector(L_form)

    dolfinx.fem.petsc.apply_lifting(
        b,
        [a_form],
        [bcs]
    )

    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE
    )

    dolfinx.fem.petsc.set_bc(b, bcs)

    b.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD
    )

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------

    print("Solving linear system with MUMPS...")

    w = fem.Function(W)

    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)

    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    solver.solve(
        b,
        w.x.petsc_vec
    )

    if solver.getConvergedReason() < 0:
        raise RuntimeError(
            "MUMPS solver failed to converge."
        )

    w.x.scatter_forward()

    print("Solve complete!")

    return w, W

def verify_mms_manufactured_solution(N=8):
    """
    Independently verify that the manufactured solution satisfies
    the three first-order equations used by the LSFEM formulation.

    We check:

        mu curl(omega_ex) + grad(p_ex) - f1 = 0
        div(u_ex) - f2 = 0
        curl(u_ex) - omega_ex - f3 = 0

    This is a verification of the MMS itself, independent of the
    finite-element solve.
    """

    comm = MPI.COMM_WORLD

    domain = mesh.create_unit_square(
        comm,
        N,
        N,
        mesh.CellType.triangle
    )

    x = SpatialCoordinate(domain)

    mu = Stokes_Problem.mu

    u = Stokes_Problem.u_ex(x)
    p = Stokes_Problem.p_ex(x)
    omega = Stokes_Problem.omega_ex(x)

    f1 = Stokes_Problem.source_f1(x)
    f2 = Stokes_Problem.source_f2(x)
    f3 = Stokes_Problem.source_f3(x)

    # ---------------------------------------------------------
    # Exact first-order residuals
    # ---------------------------------------------------------

    R_mom = (
        mu * as_vector(
            (omega.dx(1), -omega.dx(0))
        )
        + grad(p)
        - f1
    )

    R_div = div(u) - f2

    R_curl = curl(u) - omega - f3

    dx = Measure("dx", domain=domain)

    # ---------------------------------------------------------
    # L2 norms
    # ---------------------------------------------------------

    mom_sq = fem.assemble_scalar(
        fem.form(
            inner(R_mom, R_mom) * dx
        )
    )

    div_sq = fem.assemble_scalar(
        fem.form(
            R_div * R_div * dx
        )
    )

    curl_sq = fem.assemble_scalar(
        fem.form(
            R_curl * R_curl * dx
        )
    )

    mom_norm = np.sqrt(
        comm.allreduce(mom_sq, op=MPI.SUM)
    )

    div_norm = np.sqrt(
        comm.allreduce(div_sq, op=MPI.SUM)
    )

    curl_norm = np.sqrt(
        comm.allreduce(curl_sq, op=MPI.SUM)
    )

    if comm.rank == 0:

        print("\n")
        print("=" * 75)
        print("MANUFACTURED-SOLUTION VERIFICATION")
        print("=" * 75)

        print(
            "||mu curl(omega_ex) + grad(p_ex) - f1|| = "
            f"{mom_norm:.6e}"
        )

        print(
            "||div(u_ex) - f2||                    = "
            f"{div_norm:.6e}"
        )

        print(
            "||curl(u_ex) - omega_ex - f3||        = "
            f"{curl_norm:.6e}"
        )

        print("=" * 75)

        tol = 1e-10

        if (
            mom_norm < tol
            and div_norm < tol
            and curl_norm < tol
        ):
            print("MMS verification PASSED.")
        else:
            print(
                "WARNING: MMS verification residuals are not "
                "at machine precision."
            )

        print("=" * 75)

    return mom_norm, div_norm, curl_norm


def run_mms_space_study():
    """
    Run MMS mesh-refinement studies for multiple combinations
    of polynomial orders.

    Tuple:
        (k_omega, k_p, k_u)

    denotes

        P{k_omega} / P{k_p} / P{k_u}.
    """

    comm = MPI.COMM_WORLD

    # =========================================================
    # Verify MMS independently before solving anything
    # =========================================================

    if comm.rank == 0:
        print("\nChecking manufactured solution...")

    verify_mms_manufactured_solution(N=8)

    # =========================================================
    # Mesh resolutions
    # =========================================================

    grid_sizes = [8, 16, 32, 64]

    # =========================================================
    # Polynomial-order combinations
    # =========================================================

    test_cases = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
    ]

    results = {}

    # =========================================================
    # Loop over polynomial spaces
    # =========================================================

    for degrees in test_cases:

        k_omega, k_p, k_u = degrees

        label = (
            f"P{k_omega}/P{k_p}/P{k_u}"
        )

        if comm.rank == 0:
            print("\n")
            print("=" * 80)
            print(f"TEST CASE: {label}")
            print("omega / pressure / velocity")
            print("=" * 80)

        # -----------------------------------------------------
        # Error arrays
        # -----------------------------------------------------

        u_errors = []
        p_errors = []
        p_grad_errors = []
        omega_errors = []

        p_mean_errors = []
        p_gauge_errors = []

        # =====================================================
        # Mesh refinement
        # =====================================================

        for N in grid_sizes:

            if comm.rank == 0:
                print(
                    f"\nRunning {label}, N = {N}"
                )

            w_solution, W = setup_and_solve_stokes(
                problem="mms",
                N=N,
                degree_omega=k_omega,
                degree_p=k_p,
                degree_u=k_u
            )

            (
                u_error,
                p_error,
                p_grad_error,
                omega_error,
                p_mean_error,
                p_gauge_error
            ) = compute_L2_errors(
                w_solution,
                W,
                Stokes_Problem
            )

            # Store errors
            u_errors.append(u_error)
            p_errors.append(p_error)
            p_grad_errors.append(p_grad_error)
            omega_errors.append(omega_error)

            p_mean_errors.append(p_mean_error)
            p_gauge_errors.append(p_gauge_error)

            if comm.rank == 0:

                print(
                    f"  u          = {u_error:.6e}"
                )

                print(
                    f"  p          = {p_error:.6e}"
                )

                print(
                    f"  grad(p)    = {p_grad_error:.6e}"
                )

                print(
                    f"  omega      = {omega_error:.6e}"
                )

                print(
                    f"  mean error = {p_mean_error:.6e}"
                )

                print(
                    f"  p gauge-corrected = "
                    f"{p_gauge_error:.6e}"
                )

        # =====================================================
        # Convert to arrays
        # =====================================================

        u_errors = np.asarray(u_errors)
        p_errors = np.asarray(p_errors)
        p_grad_errors = np.asarray(p_grad_errors)
        omega_errors = np.asarray(omega_errors)

        p_mean_errors = np.asarray(p_mean_errors)
        p_gauge_errors = np.asarray(p_gauge_errors)

        # =====================================================
        # Store results
        # =====================================================

        results[label] = {
            "degrees": degrees,
            "N": np.asarray(grid_sizes),

            "u": u_errors,
            "p": p_errors,
            "p_grad": p_grad_errors,
            "omega": omega_errors,

            "p_mean": p_mean_errors,
            "p_gauge": p_gauge_errors
        }

        # =====================================================
        # Observed convergence rates
        # =====================================================

        if comm.rank == 0:

            print("\nObserved convergence rates")
            print("-" * 75)

            for i in range(1, len(grid_sizes)):

                rate_u = (
                    np.log(
                        u_errors[i - 1] /
                        u_errors[i]
                    )
                    / np.log(2.0)
                )

                rate_p = (
                    np.log(
                        p_errors[i - 1] /
                        p_errors[i]
                    )
                    / np.log(2.0)
                )

                rate_p_grad = (
                    np.log(
                        p_grad_errors[i - 1] /
                        p_grad_errors[i]
                    )
                    / np.log(2.0)
                )

                rate_omega = (
                    np.log(
                        omega_errors[i - 1] /
                        omega_errors[i]
                    )
                    / np.log(2.0)
                )

                print(
                    f"N={grid_sizes[i-1]:2d} -> "
                    f"N={grid_sizes[i]:2d}: "
                    f"u={rate_u:.3f}, "
                    f"p={rate_p:.3f}, "
                    f"grad(p)={rate_p_grad:.3f}, "
                    f"omega={rate_omega:.3f}"
                )

    # =========================================================
    # PLOTS
    # =========================================================

    if comm.rank == 0:

        # =====================================================
        # 1. Individual plots
        # =====================================================

        for label, data in results.items():

            N = data["N"]

            plt.figure(figsize=(8, 6))

            plt.loglog(
                N,
                data["u"],
                "o-",
                linewidth=2,
                markersize=7,
                label=r"$\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$"
            )

            plt.loglog(
                N,
                data["p"],
                "s-",
                linewidth=2,
                markersize=7,
                label=r"$\|p-p_h\|_{L^2}$"
            )

            plt.loglog(
                N,
                data["p_grad"],
                "d-",
                linewidth=2,
                markersize=7,
                label=r"$\|\nabla p-\nabla p_h\|_{L^2}$"
            )

            plt.loglog(
                N,
                data["omega"],
                "^-",
                linewidth=2,
                markersize=7,
                label=r"$\|\omega-\omega_h\|_{L^2}$"
            )

            plt.xlabel(r"Mesh resolution $N$")
            plt.ylabel("Error")

            plt.title(
                f"MMS Convergence: {label}"
            )

            plt.grid(
                True,
                which="both",
                linestyle="--",
                alpha=0.6
            )

            plt.legend()

            plt.tight_layout()

            if Stokes_Problem.weighted ==0:
                filename = (
                    f"unweighted_mms_convergence_"
                    f"{label.replace('/', '_')}.png"
                )

                plt.savefig(
                    f"Unweighted/{filename}",
                    dpi=300
                )

            if Stokes_Problem.weighted ==1:
                filename = (
                    f"weighted_mms_convergence_"
                    f"{label.replace('/', '_')}.png"
                )

                plt.savefig(
                    f"Weighted/{filename}",
                    dpi=300
                )

            plt.close()

            print(
                f"Saved {filename}"
            )

        # =====================================================
        # 2. Velocity
        # =====================================================

        plt.figure(figsize=(9, 7))

        for label, data in results.items():

            plt.loglog(
                data["N"],
                data["u"],
                "o-",
                linewidth=2,
                markersize=6,
                label=label
            )

        plt.xlabel(r"Mesh resolution $N$")
        plt.ylabel(
            r"$\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$"
        )

        plt.title(
            "MMS Velocity Convergence: "
            "Polynomial Space Comparison"
        )

        plt.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.6
        )

        plt.legend(
            title=r"$(P_\omega,P_p,P_u)$",
            fontsize=9
        )

        plt.tight_layout()

        plt.savefig(
            "mms_velocity_all_spaces.png",
            dpi=300
        )

        plt.close()

        # =====================================================
        # 3. Pressure
        # =====================================================

        plt.figure(figsize=(9, 7))

        for label, data in results.items():

            plt.loglog(
                data["N"],
                data["p"],
                "s-",
                linewidth=2,
                markersize=6,
                label=label
            )

        plt.xlabel(r"Mesh resolution $N$")
        plt.ylabel(
            r"$\|p-p_h\|_{L^2}$"
        )

        plt.title(
            "MMS Pressure Convergence: "
            "Polynomial Space Comparison"
        )

        plt.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.6
        )

        plt.legend(
            title=r"$(P_\omega,P_p,P_u)$",
            fontsize=9
        )

        plt.tight_layout()

        plt.savefig(
            "mms_pressure_all_spaces.png",
            dpi=300
        )

        plt.close()

        # =====================================================
        # 4. Pressure gradient
        # =====================================================

        plt.figure(figsize=(9, 7))

        for label, data in results.items():

            plt.loglog(
                data["N"],
                data["p_grad"],
                "d-",
                linewidth=2,
                markersize=6,
                label=label
            )

        plt.xlabel(r"Mesh resolution $N$")
        plt.ylabel(
            r"$\|\nabla p-\nabla p_h\|_{L^2}$"
        )

        plt.title(
            "MMS Pressure Gradient Convergence: "
            "Polynomial Space Comparison"
        )

        plt.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.6
        )

        plt.legend(
            title=r"$(P_\omega,P_p,P_u)$",
            fontsize=9
        )

        plt.tight_layout()

        plt.savefig(
            "mms_pressure_gradient_all_spaces.png",
            dpi=300
        )

        plt.close()

        # =====================================================
        # 5. Vorticity
        # =====================================================

        plt.figure(figsize=(9, 7))

        for label, data in results.items():

            plt.loglog(
                data["N"],
                data["omega"],
                "^-",
                linewidth=2,
                markersize=6,
                label=label
            )

        plt.xlabel(r"Mesh resolution $N$")
        plt.ylabel(
            r"$\|\omega-\omega_h\|_{L^2}$"
        )

        plt.title(
            "MMS Vorticity Convergence: "
            "Polynomial Space Comparison"
        )

        plt.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.6
        )

        plt.legend(
            title=r"$(P_\omega,P_p,P_u)$",
            fontsize=9
        )

        plt.tight_layout()

        plt.savefig(
            "mms_vorticity_all_spaces.png",
            dpi=300
        )

        plt.close()

        # =====================================================
        # 6. Gauge diagnostic
        # =====================================================

        plt.figure(figsize=(9, 7))

        for label, data in results.items():

            plt.loglog(
                data["N"],
                np.abs(data["p_mean"]),
                "o-",
                linewidth=2,
                markersize=6,
                label=label
            )

        plt.xlabel(r"Mesh resolution $N$")
        plt.ylabel(
            r"$|\operatorname{mean}(p_h-p_{\rm ex})|$"
        )

        plt.title(
            "Pressure Gauge Error"
        )

        plt.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.6
        )

        plt.legend(
            title=r"$(P_\omega,P_p,P_u)$",
            fontsize=9
        )

        plt.tight_layout()

        plt.savefig(
            "mms_pressure_gauge_error.png",
            dpi=300
        )

        plt.close()

        # =====================================================
        # 7. Raw vs gauge-corrected pressure
        # =====================================================

        plt.figure(figsize=(9, 7))

        for label, data in results.items():

            plt.loglog(
                data["N"],
                data["p_gauge"],
                "o--",
                linewidth=2,
                markersize=5,
                label=label
            )

        plt.xlabel(r"Mesh resolution $N$")
        plt.ylabel(
            r"Gauge-corrected $\|p-p_h\|_{L^2}$"
        )

        plt.title(
            "Gauge-Corrected Pressure Error"
        )

        plt.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.6
        )

        plt.legend(
            title=r"$(P_\omega,P_p,P_u)$",
            fontsize=9
        )

        plt.tight_layout()

        plt.savefig(
            "mms_pressure_gauge_corrected.png",
            dpi=300
        )

        plt.close()

        # =====================================================
        # FINAL CONVERGENCE SUMMARY
        # =====================================================

        print("\n")
        print("=" * 100)
        print("FINAL MMS CONVERGENCE-RATE SUMMARY")
        print("=" * 100)

        print(
            f"{'Space':<14}"
            f"{'u':>10}"
            f"{'p':>10}"
            f"{'grad(p)':>12}"
            f"{'omega':>12}"
            f"{'p-gauge':>12}"
        )

        print("-" * 100)

        for label, data in results.items():

            u = data["u"]
            p = data["p"]
            p_grad = data["p_grad"]
            omega = data["omega"]
            p_gauge = data["p_gauge"]

            # Finest-mesh convergence rate

            rate_u = (
                np.log(u[-2] / u[-1])
                / np.log(2.0)
            )

            rate_p = (
                np.log(p[-2] / p[-1])
                / np.log(2.0)
            )

            rate_p_grad = (
                np.log(p_grad[-2] / p_grad[-1])
                / np.log(2.0)
            )

            rate_omega = (
                np.log(omega[-2] / omega[-1])
                / np.log(2.0)
            )

            rate_p_gauge = (
                np.log(p_gauge[-2] / p_gauge[-1])
                / np.log(2.0)
            )

            print(
                f"{label:<14}"
                f"{rate_u:>10.3f}"
                f"{rate_p:>10.3f}"
                f"{rate_p_grad:>12.3f}"
                f"{rate_omega:>12.3f}"
                f"{rate_p_gauge:>12.3f}"
            )

        print("=" * 100)

    return results



def compute_L2_errors(w_solution, W, stokes_problem):
    """
    Compute MMS errors:

        ||u-u_h||_L2
        ||p-p_h||_L2
        ||grad(p)-grad(p_h)||_L2
        ||omega-omega_h||_L2

    Also compute two pressure-gauge diagnostics:

        mean(p_h - p_exact)

    and the pressure L2 error after removing that constant offset.

    The gauge-corrected pressure error is diagnostic only; the
    primary pressure error remains the raw error corresponding to
    the pressure pin used in the solve.
    """

    domain = W.mesh
    comm = domain.comm
    x = SpatialCoordinate(domain)
    dx = Measure("dx", domain=domain)

    # =========================================================
    # VELOCITY
    # =========================================================

    V_u, dofs_u = W.sub(2).collapse()

    u_h = fem.Function(V_u)
    u_h.x.array[:] = w_solution.x.array[dofs_u]
    u_h.x.scatter_forward()

    u_exact = fem.Function(V_u)

    u_exact.interpolate(
        fem.Expression(
            stokes_problem.u_ex(x),
            V_u.element.interpolation_points
        )
    )

    u_error_sq = fem.assemble_scalar(
        fem.form(
            inner(
                u_h - u_exact,
                u_h - u_exact
            ) * dx
        )
    )

    u_L2_error = np.sqrt(
        comm.allreduce(
            u_error_sq,
            op=MPI.SUM
        )
    )

    # =========================================================
    # PRESSURE
    # =========================================================

    V_p, dofs_p = W.sub(1).collapse()

    p_h = fem.Function(V_p)
    p_h.x.array[:] = w_solution.x.array[dofs_p]
    p_h.x.scatter_forward()

    p_exact = fem.Function(V_p)

    p_exact.interpolate(
        fem.Expression(
            stokes_problem.p_ex(x),
            V_p.element.interpolation_points
        )
    )

    p_error = p_h - p_exact

    p_mean_error = fem.assemble_scalar(
        fem.form(p_error * dx)
    ) / fem.assemble_scalar(
        fem.form(1.0 * dx)
    )

    # ---------------------------------------------------------
    # Raw pressure L2 error
    # ---------------------------------------------------------

    p_error_sq = fem.assemble_scalar(
        fem.form(
            p_error * p_error * dx
        )
    )

    p_L2_error = np.sqrt(
        comm.allreduce(
            p_error_sq,
            op=MPI.SUM
        )
    )

    # ---------------------------------------------------------
    # Pressure gradient error
    # ---------------------------------------------------------

    grad_p_error_sq = fem.assemble_scalar(
        fem.form(
            inner(
                grad(p_h) - grad(p_exact),
                grad(p_h) - grad(p_exact)
            ) * dx
        )
    )

    p_grad_L2_error = np.sqrt(
        comm.allreduce(
            grad_p_error_sq,
            op=MPI.SUM
        )
    )

    # ---------------------------------------------------------
    # Pressure mean / gauge diagnostic
    # ---------------------------------------------------------

    pressure_error_mean_local = fem.assemble_scalar(
        fem.form(
            p_error * dx
        )
    )

    domain_measure_local = fem.assemble_scalar(
        fem.form(
            1.0 * dx
        )
    )

    pressure_error_mean = (
        comm.allreduce(
            pressure_error_mean_local,
            op=MPI.SUM
        )
        /
        comm.allreduce(
            domain_measure_local,
            op=MPI.SUM
        )
    )

    # ---------------------------------------------------------
    # Gauge-corrected pressure error
    #
    # Remove the constant mean offset:
    #
    #     p_h_corrected = p_h - mean(p_h-p_exact)
    # ---------------------------------------------------------

    p_gauge_error_sq = fem.assemble_scalar(
        fem.form(
            (
                p_error - pressure_error_mean
            )
            *
            (
                p_error - pressure_error_mean
            )
            * dx
        )
    )

    p_gauge_L2_error = np.sqrt(
        comm.allreduce(
            p_gauge_error_sq,
            op=MPI.SUM
        )
    )

    # =========================================================
    # VORTICITY
    # =========================================================

    V_omega, dofs_omega = W.sub(0).collapse()

    omega_h = fem.Function(V_omega)
    omega_h.x.array[:] = w_solution.x.array[dofs_omega]
    omega_h.x.scatter_forward()

    omega_exact = fem.Function(V_omega)

    omega_exact.interpolate(
        fem.Expression(
            stokes_problem.omega_ex(x),
            V_omega.element.interpolation_points
        )
    )

    omega_error_sq = fem.assemble_scalar(
        fem.form(
            (
                omega_h - omega_exact
            )
            *
            (
                omega_h - omega_exact
            )
            * dx
        )
    )

    omega_L2_error = np.sqrt(
        comm.allreduce(
            omega_error_sq,
            op=MPI.SUM
        )
    )

    return (
        u_L2_error,
        p_L2_error,
        p_grad_L2_error,
        omega_L2_error,
        pressure_error_mean,
        p_gauge_L2_error
    )


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
        
        if Stokes_Problem.weighted == 0:
            plotter.screenshot("pressure_comparison_unweighted.png")
        else:
            plotter.screenshot("pressure_comparison_weighted.png")
        print("Pressure comparison plot saved to pressure_comparison.png")


    if problem == 'cavity':
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "p_h"})
        plotter.view_xy()
        plotter.add_title("Numerical Pressure", font_size=14, color="black")
        if Stokes_Problem.weighted == 0:
            plotter.screenshot("cavity_pressure_unweighted.png")
        else:
            plotter.screenshot("cavity_pressure_weighted.png")
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
    grid_diff.point_data["Difference"] = grid_num.point_data["Velocity"] - grid_true.point_data["Velocity"]
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

        if Stokes_Problem.weighted == 0:
            plotter.screenshot("velocity_comparison_unweighted.png")
        else:
            plotter.screenshot("velocity_comparison_weighted.png")
        print("Velocity comparison plot saved to velocity_comparison.png")

    if problem == 'cavity':
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "|u_h|"})
        plotter.view_xy()
        plotter.add_title("Numerical Velocity", font_size=14, color="black")

        if Stokes_Problem.weighted == 0:
            plotter.screenshot("cavity_velocity_unweighted.png")
        else:
            plotter.screenshot("cavity_velocity_weighted")

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
        grid_diff.point_data["Difference"] = grid_num.point_data["Vorticity"] - grid_true.point_data["Vorticity"]
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

        if Stokes_Problem.weighted == 0:
            plotter.screenshot("vorticity_comparison_unweighted.png")
        else:
            plotter.screenshot("vorticity_comparison_weighted.png")
        print("Vorticity comparison plot saved to vorticity_comparison.png")

    elif problem == 'cavity':
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_num, show_edges=True, scalar_bar_args={**scalar_bar_args, "title": "omega_h"})
        plotter.view_xy()
        plotter.add_title("Numerical Vorticity (Cavity)", font_size=14, color="black")
        if Stokes_Problem.weighted == 0:
            plotter.screenshot("cavity_vorticity_unweighted.png")
        else:   
            plotter.screenshot("cavity_vorticity_weighted.png")
        print("Cavity vorticity plot saved to cavity_vorticity.png")

def _extract_velocity(w_solution, W):
    """Return the collapsed velocity Function."""
    R, dofs = W.sub(2).collapse()
    u_h = fem.Function(R)
    u_h.x.array[:] = w_solution.x.array[dofs]
    u_h.x.scatter_forward()
    return u_h, R


def _extract_pressure(w_solution, W):
    """Return the collapsed pressure Function."""
    Q, dofs = W.sub(1).collapse()
    p_h = fem.Function(Q)
    p_h.x.array[:] = w_solution.x.array[dofs]
    p_h.x.scatter_forward()
    return p_h, Q


def _extract_vorticity(w_solution, W):
    """Return the collapsed vorticity Function."""
    Q, dofs = W.sub(0).collapse()
    omega_h = fem.Function(Q)
    omega_h.x.array[:] = w_solution.x.array[dofs]
    omega_h.x.scatter_forward()
    return omega_h, Q


def _make_velocity_grid(u_h, V):
    """Create a PyVista grid containing u and v as point data."""
    cells, types, x_pts = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x_pts)

    u_array = u_h.x.array.reshape(-1, 2)
    grid.point_data["u"] = u_array[:, 0]
    grid.point_data["v"] = u_array[:, 1]
    grid.point_data["Speed"] = np.linalg.norm(u_array, axis=1)
    return grid


def plot_cavity_velocity_field(w_solution, W):
    """
    Plot the cavity velocity magnitude together with velocity vectors.
    The vector field is sampled onto a regular grid for a clean quiver plot.
    """
    u_h, R = _extract_velocity(w_solution, W)
    grid = _make_velocity_grid(u_h, R)

    pts = grid.points[:, :2]
    u_vals = np.asarray(grid.point_data["u"])
    v_vals = np.asarray(grid.point_data["v"])

    nx = 25
    ny = 25
    xq = np.linspace(0.02, 0.98, nx)
    yq = np.linspace(0.02, 0.98, ny)
    X, Y = np.meshgrid(xq, yq)

    U = griddata(pts, u_vals, (X, Y), method="linear")
    V = griddata(pts, v_vals, (X, Y), method="linear")
    speed = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.contourf(X, Y, speed, levels=30)
    fig.colorbar(contour, ax=ax, label=r"$|\mathbf{u}|$")
    ax.quiver(X, Y, U, V, speed, angles="xy", scale_units="xy",
              scale=8, width=0.0025)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Lid-Driven Cavity Velocity Field")
    plt.tight_layout()
    plt.savefig("cavity_velocity_field.png", dpi=300)
    plt.close()
    print("Velocity field plot saved to cavity_velocity_field.png")


def plot_cavity_streamlines(w_solution, W):
    """
    Plot streamlines of the computed cavity velocity field.
    """
    u_h, R = _extract_velocity(w_solution, W)
    grid = _make_velocity_grid(u_h, R)

    pts = grid.points[:, :2]
    u_vals = np.asarray(grid.point_data["u"])
    v_vals = np.asarray(grid.point_data["v"])

    nx = 150
    ny = 150
    xq = np.linspace(0.0, 1.0, nx)
    yq = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xq, yq)

    U = griddata(pts, u_vals, (X, Y), method="linear")
    V = griddata(pts, v_vals, (X, Y), method="linear")
    speed = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.contourf(X, Y, speed, levels=30, alpha=0.7)
    fig.colorbar(contour, ax=ax, label=r"$|\mathbf{u}|$")
    ax.streamplot(xq, yq, U, V, density=1.5, linewidth=0.8,
                  arrowsize=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Lid-Driven Cavity Streamlines")
    plt.tight_layout()
    if Stokes_Problem.weighted == 0:
        plt.savefig("cavity_streamlines_unweighted.png", dpi=300)
    if Stokes_Problem.weighted == 1:
        plt.savefig("cavity_streamlines_weighted.png", dpi=300)
    plt.close()
    print("Streamline plot saved to cavity_streamlines.png")


def plot_cavity_pressure(w_solution, W):
    """Plot the computed cavity pressure."""
    p_h, Q = _extract_pressure(w_solution, W)

    cells, types, x_pts = plot.vtk_mesh(Q)
    grid = pyvista.UnstructuredGrid(cells, types, x_pts)
    grid.point_data["Pressure"] = p_h.x.array
    grid.set_active_scalars("Pressure")

    scalar_bar_args = {
        "vertical": True,
        "position_x": 0.82,
        "position_y": 0.1,
        "width": 0.08,
        "height": 0.8
    }

    plotter = pyvista.Plotter(off_screen=True, window_size=[800, 700])
    plotter.add_mesh(
        grid,
        show_edges=False,
        scalar_bar_args={**scalar_bar_args, "title": "p"}
    )
    plotter.view_xy()
    plotter.add_title("Lid-Driven Cavity Pressure",
                      font_size=16, color="black")
    if Stokes_Problem.weighted == 0:
        plotter.screenshot("cavity_pressure_unweighted.png")
    else: 
        plotter.screenshot("cavity_pressure_weighted.png")
    plotter.close()
    print("Pressure plot saved to cavity_pressure.png")


def plot_cavity_vorticity(w_solution, W):
    """Plot computed cavity vorticity with a symmetric color scale."""
    omega_h, Q = _extract_vorticity(w_solution, W)

    cells, types, x_pts = plot.vtk_mesh(Q)
    grid = pyvista.UnstructuredGrid(cells, types, x_pts)
    grid.point_data["Vorticity"] = omega_h.x.array
    grid.set_active_scalars("Vorticity")

    omega_max = max(abs(np.min(omega_h.x.array)),
                    abs(np.max(omega_h.x.array)))

    scalar_bar_args = {
        "vertical": True,
        "position_x": 0.82,
        "position_y": 0.1,
        "width": 0.08,
        "height": 0.8
    }

    plotter = pyvista.Plotter(off_screen=True, window_size=[800, 700])
    plotter.add_mesh(
        grid,
        show_edges=False,
        clim=[-omega_max, omega_max],
        cmap="coolwarm",
        scalar_bar_args={**scalar_bar_args, "title": r"$\omega$"}
    )
    plotter.view_xy()
    plotter.add_title("Lid-Driven Cavity Vorticity",
                      font_size=16, color="black")
    if Stokes_Problem.weighted == 0:
        plotter.screenshot("cavity_vorticity_unweighted.png")
    else: 
        plotter.screenshot("cavity_vorticity_weighted.png")
    plotter.close()
    print("Vorticity plot saved to cavity_vorticity.png")


def plot_cavity_centerlines(w_solution, W):
    """Plot u on x=0.5 and v on y=0.5."""
    u_h, R = _extract_velocity(w_solution, W)
    grid = _make_velocity_grid(u_h, R)

    line_vert = grid.sample_over_line(
        [0.5, 0.0, 0.0], [0.5, 1.0, 0.0], resolution=200
    )
    y_coords = line_vert.points[:, 1]
    u_profile = line_vert.point_data["u"]

    line_horiz = grid.sample_over_line(
        [0.0, 0.5, 0.0], [1.0, 0.5, 0.0], resolution=200
    )
    x_coords = line_horiz.points[:, 0]
    v_profile = line_horiz.point_data["v"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(u_profile, y_coords, lw=2,
                 label="LSFEM Stokes")
    axes[0].set_xlabel(r"Horizontal velocity $u$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_title(r"Vertical centerline ($x=0.5$)")
    axes[0].grid(True, ls="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(x_coords, v_profile, lw=2,
                 label="LSFEM Stokes")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"Vertical velocity $v$")
    axes[1].set_title(r"Horizontal centerline ($y=0.5$)")
    axes[1].grid(True, ls="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if Stokes_Problem.weighted == 0:
        plt.savefig("cavity_centerlines.png", dpi=300)
    if Stokes_Problem.weighted == 1:
        plt.savefig("cavity_centerlines_weighted.png", dpi=300)
    plt.close()
    print("Centerline plot saved to cavity_centerlines.png")


def _make_dg0_space(domain, shape=None):
    """Create a discontinuous piecewise-constant space for residual plotting."""
    if shape is None:
        element = basix.ufl.element(
            "Discontinuous Lagrange",
            domain.topology.cell_name(),
            0
        )
    else:
        element = basix.ufl.element(
            "Discontinuous Lagrange",
            domain.topology.cell_name(),
            0,
            shape=shape
        )
    return fem.functionspace(domain, element)


def _evaluate_on_dg0(expr, V0):
    """Evaluate a UFL expression at DG0 interpolation points."""
    result = fem.Function(V0)
    expression = fem.Expression(expr, V0.element.interpolation_points)
    result.interpolate(expression)
    result.x.scatter_forward()
    return result


def _make_cell_residual_grid(domain, scalar_function, name):
    """
    Put a DG0 scalar field onto a mesh grid as cell data.
    """
    cells, types, x_pts = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(cells, types, x_pts)

    values = np.asarray(scalar_function.x.array).copy()
    n_cells = grid.n_cells

    # In serial, DG0 has one value per cell.  Keep the local values that
    # correspond to the plotted cells.
    if len(values) != n_cells:
        values = values[:n_cells]

    grid.cell_data[name] = values
    grid.set_active_scalars(name)
    return grid


def compute_cavity_residuals(w_solution, W):
    """
    Compute the three first-order-system residuals used by the LSFEM:

        R_mom  = mu curl(omega) + grad(p)
        R_div  = div(u)
        R_curl = curl(u) - omega

    for the homogeneous cavity problem.
    """
    domain = W.mesh
    mu = Stokes_Problem.mu

    u_h, V_u = _extract_velocity(w_solution, W)
    p_h, V_p = _extract_pressure(w_solution, W)
    omega_h, V_omega = _extract_vorticity(w_solution, W)

    # The UFL expressions are built from the actual FE functions.
    R_div = div(u_h)
    R_curl = curl(u_h) - omega_h
    R_mom = (
        mu * as_vector((omega_h.dx(1), -omega_h.dx(0)))
        + grad(p_h)
    )

    V0 = _make_dg0_space(domain)
    V0_vec = _make_dg0_space(domain, shape=(2,))

    div_h = _evaluate_on_dg0(R_div, V0)
    curl_h = _evaluate_on_dg0(R_curl, V0)
    mom_h = _evaluate_on_dg0(R_mom, V0_vec)

    # Convert the vector momentum residual to a scalar magnitude.
    mom_array = mom_h.x.array.reshape(-1, 2)
    mom_mag = fem.Function(V0)
    mom_mag.x.array[:] = np.linalg.norm(mom_array, axis=1)
    mom_mag.x.scatter_forward()

    return div_h, curl_h, mom_mag


def plot_cavity_residuals(w_solution, W):
    """
    Plot the absolute LSFEM residuals:
        |div u|,
        |curl u - omega|,
        |mu curl(omega) + grad(p)|.
    """
    domain = W.mesh
    div_h, curl_h, mom_h = compute_cavity_residuals(w_solution, W)

    residuals = [
        (div_h, r"$|\nabla\cdot\mathbf{u}|$",
         "cavity_divergence_residual.png",
         "Cavity Divergence Residual"),
        (curl_h, r"$|\nabla\times\mathbf{u}-\omega|$",
         "cavity_curl_residual.png",
         "Cavity Curl-Vorticity Residual"),
        (mom_h, r"$|\mu\nabla\times\omega+\nabla p|$",
         "cavity_momentum_residual.png",
         "Cavity Momentum Residual")
    ]

    for residual, title, filename, plot_title in residuals:
        grid = _make_cell_residual_grid(domain, residual, "Residual")

        scalar_bar_args = {
            "vertical": True,
            "position_x": 0.82,
            "position_y": 0.1,
            "width": 0.08,
            "height": 0.8
        }

        plotter = pyvista.Plotter(
            off_screen=True,
            window_size=[800, 700]
        )
        plotter.add_mesh(
            grid,
            show_edges=False,
            scalar_bar_args={
                **scalar_bar_args,
                "title": title
            }
        )
        plotter.view_xy()
        plotter.add_title(plot_title, font_size=16, color="black")
        plotter.screenshot(filename)
        plotter.close()
        print(f"Residual plot saved to {filename}")


def compute_cavity_residual_norms(w_solution, W):
    """
    Return L2 norms of the three first-order-system residuals.
    These are useful for convergence studies.
    """
    domain = W.mesh
    mu = Stokes_Problem.mu
    dx = Measure("dx", domain=domain)

    u_h, _ = _extract_velocity(w_solution, W)
    p_h, _ = _extract_pressure(w_solution, W)
    omega_h, _ = _extract_vorticity(w_solution, W)

    R_div = div(u_h)
    R_curl = curl(u_h) - omega_h
    R_mom = (
        mu * as_vector((omega_h.dx(1), -omega_h.dx(0)))
        + grad(p_h)
    )

    comm = domain.comm

    div_norm = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(
                fem.form(inner(R_div, R_div) * dx)
            ),
            op=MPI.SUM
        )
    )

    curl_norm = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(
                fem.form(inner(R_curl, R_curl) * dx)
            ),
            op=MPI.SUM
        )
    )

    mom_norm = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(
                fem.form(inner(R_mom, R_mom) * dx)
            ),
            op=MPI.SUM
        )
    )

    return div_norm, curl_norm, mom_norm


def plot_cavity_symmetry_error(w_solution, W):
    """
    Check expected reflection symmetry:
        u(x,y) =  u(1-x,y)
        v(x,y) = -v(1-x,y).
    """
    u_h, R = _extract_velocity(w_solution, W)
    grid = _make_velocity_grid(u_h, R)

    pts = grid.points[:, :2]
    u_vals = np.asarray(grid.point_data["u"])
    v_vals = np.asarray(grid.point_data["v"])

    nx = 150
    ny = 150
    xq = np.linspace(0.0, 1.0, nx)
    yq = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xq, yq)

    U = griddata(pts, u_vals, (X, Y), method="linear")
    V = griddata(pts, v_vals, (X, Y), method="linear")

    U_reflected = np.flip(U, axis=1)
    V_reflected = np.flip(V, axis=1)

    u_error = U - U_reflected
    v_error = np.abs(V + V_reflected)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    c1 = axes[0].contourf(X, Y, u_error, levels=30)
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_aspect("equal")
    axes[0].set_title(r"$|u(x,y)-u(1-x,y)|$")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")

    c2 = axes[1].contourf(X, Y, v_error, levels=30)
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_aspect("equal")
    axes[1].set_title(r"$|v(x,y)+v(1-x,y)|$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")

    plt.tight_layout()
    if Stokes_Problem.weighted == 0:
        plt.savefig("Unweighted/cavity_symmetry_error_unweighted.png", dpi=300)
    else:
        plt.savefig("Weighted/cavity_symmetry_error_weighted.png", dpi=300)
    plt.close()
    print("Symmetry error plot saved to cavity_symmetry_error.png")


def analyze_convergence():
    """
    Grid-independence study for the cavity.

    Tracks:
      1. minimum u on the vertical centerline
      2. L2 norm of the divergence residual
      3. L2 norm of the curl-vorticity residual
      4. L2 norm of the momentum residual
    """
    grid_sizes = [16, 24, 32, 48, 64]

    hs = []
    min_u_values = []
    div_norms = []
    curl_norms = []
    mom_norms = []

    for N in grid_sizes:
        print(f"\n--- Running for N = {N} ---")
        w_solution, W = setup_and_solve_stokes(problem, N=N)

        u_h, R = _extract_velocity(w_solution, W)
        grid = _make_velocity_grid(u_h, R)

        line_vert = grid.sample_over_line(
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            resolution=250
        )

        min_u = np.min(line_vert.point_data["u"])

        div_norm, curl_norm, mom_norm = compute_cavity_residual_norms(
            w_solution, W
        )

        h = 1.0 / N
        hs.append(h)
        min_u_values.append(min_u)
        div_norms.append(div_norm)
        curl_norms.append(curl_norm)
        mom_norms.append(mom_norm)

        print(
            f"N={N}: min(u)={min_u:.6e}, "
            f"||div u||={div_norm:.6e}, "
            f"||curl u-omega||={curl_norm:.6e}, "
            f"||momentum residual||={mom_norm:.6e}"
        )

    hs = np.array(hs)
    min_u_values = np.array(min_u_values)
    div_norms = np.array(div_norms)
    curl_norms = np.array(curl_norms)
    mom_norms = np.array(mom_norms)

    # ---------------------------------------------------------
    # Quantity-of-interest grid independence
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(
        hs, min_u_values, "o-", lw=2,
        label=r"Minimum centerline $u$"
    )
    plt.xlabel(r"Grid spacing $h=1/N$")
    plt.ylabel(r"$u_{\min}$ on $x=0.5$")
    plt.title("Lid-Driven Cavity Grid Independence")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    # plt.gca().invert_xaxis()
    plt.tight_layout()
    if Stokes_Problem.weighted == 1:
        plt.savefig("Weighted/cavity_grid_convergence_weighted.png", dpi=300)
    if Stokes_Problem.weighted == 0:
        plt.savefig("Unweighted/cavity_grid_convergence_unweighted.png", dpi=300)
    plt.close()
    print("Grid-independence plot saved to cavity_grid_convergence.png")

    # ---------------------------------------------------------
    # LSFEM residual convergence
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.loglog(hs, div_norms, "o-", label=r"$\|\nabla\cdot u_h\|_{L^2}$")
    plt.loglog(
        hs, curl_norms, "s-",
        label=r"$\|\nabla\times u_h-\omega_h\|_{L^2}$"
    )
    plt.loglog(
        hs, mom_norms, "d-",
        label=r"$\|\mu\nabla\times\omega_h+\nabla p_h\|_{L^2}$"
    )
    plt.xlabel(r"Grid spacing $h=1/N$")
    plt.ylabel("Residual norm")
    plt.title("LSFEM Residual Convergence")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    # plt.gca().invert_xaxis()
    plt.tight_layout()
    if Stokes_Problem.weighted == 1:
        plt.savefig("Weighted/cavity_residual_convergence_weighted.png", dpi=300)
    if Stokes_Problem.weighted == 0:
        plt.savefig("Unweighted/cavity_residual_convergence_unweighted.png", dpi=300)
    plt.close()
    print("Residual convergence plot saved to cavity_residual_convergence.png")


if __name__ == "__main__":

    if problem == "mms":

        results = run_mms_space_study()

    elif problem == "cavity":

        # Main cavity solve
        w_solution, W = setup_and_solve_stokes(
            problem,
            N=32
        )

        # Physical solution plots
        plot_cavity_velocity_field(w_solution, W)
        plot_cavity_streamlines(w_solution, W)
        plot_cavity_pressure(w_solution, W)
        plot_cavity_vorticity(w_solution, W)
        plot_cavity_centerlines(w_solution, W)

        # LSFEM residual diagnostics
        plot_cavity_residuals(w_solution, W)

        # Symmetry diagnostic
        plot_cavity_symmetry_error(w_solution, W)

        # Cavity convergence
        analyze_convergence()