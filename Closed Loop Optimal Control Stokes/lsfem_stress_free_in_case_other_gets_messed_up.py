# =====================================================================
# Traction BC for the MMS case: all new / modified code in one cell
# =====================================================================
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
import dolfinx.fem.petsc
from dolfinx import fem, mesh
from dolfinx.fem import Constant
from ufl import (
    SpatialCoordinate, TrialFunctions, TestFunctions,
    as_vector, curl, div, grad, inner, dot, sym, Identity,
    FacetNormal, CellDiameter, Measure, sin, cos, pi, exp
)
import pyvista
import matplotlib.pyplot as plt

class LSFEMStokesVorticityPressure:
    def __init__(self, mu=1.0, problem="mms_smooth", weighted=False):
        if problem not in ("mms_smooth", "mms_localized", "cavity"):
            raise ValueError(f"problem must be 'mms' or 'cavity', got {problem!r}")
        self.mu = mu
        self.problem = problem          # "mms" or "cavity"
        self.weighted = int(weighted)   # 0: unweighted, 1: weighted LS functional

    def u_ex(self, x):

        if self.problem == "mms_smooth":
            sinx, siny = sin(pi * x[0]), sin(pi * x[1])
            cosx, cosy = cos(pi * x[0]), cos(pi * x[1])
            c_factor = 2 * pi * sinx * siny
            return c_factor * as_vector((cosy * sinx, -cosx * siny))
        
        if self.problem == "mms_localized":
            r2 = (x[0]-0.5)**2 + (x[1]-0.5)**2
            envelope = exp(-15.0*r2)
            u1 = envelope * sin(2.0 * pi * x[1])
            u2 = -envelope * sin(2.0 * pi * x[0])
            return as_vector((u1, u2))

    def p_ex(self, x):
        
        if self.problem == "mms_smooth":
            return sin(2 * pi * x[0]) * sin(2 * pi * x[1])
        
        if self.problem == "mms_localized":
            # Localized pressure variation with spatial decay
            return cos(3.0 * pi * x[0]) * sin(3.0 * pi * x[1]) * exp(-5.0 * ((x[0] - 0.5)**2))

    def omega_ex(self, x):
        u = self.u_ex(x)
        return u[1].dx(0) - u[0].dx(1)

    def traction_g(self, x, n):
        """Exact Cauchy traction sigma(u_ex, p_ex) n (MMS only)."""
        u, p = self.u_ex(x), self.p_ex(x)
        sigma = 2.0 * self.mu * sym(grad(u)) - p * Identity(2)
        return dot(sigma, n)

    def source_f1(self, x):
        if self.problem == "cavity":
            return as_vector((0.0 * x[0], 0.0 * x[0]))
        omega, p = self.omega_ex(x), self.p_ex(x)
        curl_omega = as_vector((omega.dx(1), -omega.dx(0)))
        return self.mu * curl_omega + grad(p)

    def source_f2(self, x):
        if self.problem == "cavity":
            return 0.0 * x[0]
        return div(self.u_ex(x))

    def source_f3(self, x):
        if self.problem == "cavity":
            return 0.0 * x[0]
        return curl(self.u_ex(x)) - self.omega_ex(x)


Stokes_Problem = LSFEMStokesVorticityPressure(mu=1.0, problem="mms_smooth", weighted=False)
problem = Stokes_Problem.problem


# ---------------------------------------------------------------------
# Traction helpers
# ---------------------------------------------------------------------
def traction(mu, p, u, n):
    """T(p,u) = (2 mu eps(u) - p I) n"""
    return dot(2.0 * mu * sym(grad(u)) - p * Identity(2), n)


def _traction_g(self, x, n):
    """Exact Cauchy traction sigma(u_ex, p_ex) n for the MMS."""
    u = self.u_ex(x)
    p = self.p_ex(x)
    sigma = 2.0 * self.mu * sym(grad(u)) - p * Identity(2)
    return dot(sigma, n)

# attach to the existing class (works on the existing Stokes_Problem instance)
Stokes_Problem = LSFEMStokesVorticityPressure(mu=1.0)
LSFEMStokesVorticityPressure.traction_g = _traction_g
Stokes_Problem.weighted = 1


# ---------------------------------------------------------------------
# Rigid-body kernel removal (pure-traction problem determines u only up
# to 2 translations + 1 rotation). u_ex = 0 at all three points.
# ---------------------------------------------------------------------
def create_rigid_body_pins(W):
    pins = [
        (0, (0.0, 0.0)),   # u_x at (0,0)
        (1, (0.0, 0.0)),   # u_y at (0,0)
        (1, (1.0, 0.0)),   # u_y at (1,0)  -> kills the rotation
    ]
    bcs = []
    for comp, pt in pins:
        sub = W.sub(2).sub(comp)
        Vc, dof_map = sub.collapse()

        def at_pt(x, pt=pt):
            return np.isclose(x[0], pt[0]) & np.isclose(x[1], pt[1])

        dofs = fem.locate_dofs_geometrical(Vc, at_pt)
        if len(dofs) == 0:
            raise RuntimeError(f"Could not locate velocity DOF at {pt}.")

        mixed_dof = np.asarray([dof_map[dofs[0]]], dtype=np.int32)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), mixed_dof, sub))
    return bcs


# ---------------------------------------------------------------------
# Bilinear form (adds boundary traction term for "mms")
# ---------------------------------------------------------------------
def create_bilinear_form(mu, W):
    """
    Subspaces: 0 = omega, 1 = pressure, 2 = velocity
    """
    omega, p, u = TrialFunctions(W)
    xi, q, v = TestFunctions(W)

    domain = W.mesh
    dx = Measure("dx", domain=domain)

    R_mom = mu * as_vector((omega.dx(1), -omega.dx(0))) + grad(p)
    R_mom_test = mu * as_vector((xi.dx(1), -xi.dx(0))) + grad(q)

    R_div, R_div_test = div(u), div(v)
    R_curl, R_curl_test = curl(u) - omega, curl(v) - xi

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
            + (1.0 / h) ** 2 * inner(R_div, R_div_test)
            + (1.0 / h) ** 2 * inner(R_curl, R_curl_test)
        ) * dx

    # Boundary traction residual: w_b * (T(p,u), T(q,v))_{L2(Gamma)}
    # if Stokes_Problem.problem == "mms":
    #     ds = Measure("ds", domain=domain)
    #     n = FacetNormal(domain)
    #     w_b = (1.0 / CellDiameter(domain)) if Stokes_Problem.weighted == 1 else 1.0
    #     #a += w_b * inner(traction(mu, p, u, n), traction(mu, q, v, n)) * ds

    return fem.form(a)


# ---------------------------------------------------------------------
# Linear form (adds boundary traction data for "mms")
# ---------------------------------------------------------------------
def create_linear_form(mu, W):
    xi, q, v = TestFunctions(W)

    domain = W.mesh
    dx = Measure("dx", domain=domain)

    R_mom_test = mu * as_vector((xi.dx(1), -xi.dx(0))) + grad(q)
    R_div_test = div(v)
    R_curl_test = curl(v) - xi

    if Stokes_Problem.problem == "cavity":
        f1 = Constant(domain, PETSc.ScalarType((0.0, 0.0)))
        f2 = Constant(domain, PETSc.ScalarType(0.0))
        f3 = Constant(domain, PETSc.ScalarType(0.0))
    else:
        x = SpatialCoordinate(domain)
        f1 = Stokes_Problem.source_f1(x)
        f2 = Stokes_Problem.source_f2(x)
        f3 = Stokes_Problem.source_f3(x)

    if Stokes_Problem.weighted == 0:
        L = (
            inner(f1, R_mom_test)
            + inner(f2, R_div_test)
            + inner(f3, R_curl_test)
        ) * dx
    else:
        h = CellDiameter(domain)
        L = (
            inner(f1, R_mom_test)
            + (1.0 / h) ** 2 * inner(f2, R_div_test)
            + (1.0 / h) ** 2 * inner(f3, R_curl_test)
        ) * dx

    # Boundary traction data: w_b * (g, T(q,v))_{L2(Gamma)}
    # if Stokes_Problem.problem == "mms":
    #     ds = Measure("ds", domain=domain)
    #     n = FacetNormal(domain)
    #     w_b = (1.0 / CellDiameter(domain)) if Stokes_Problem.weighted == 1 else 1.0
    #     g = Stokes_Problem.traction_g(SpatialCoordinate(domain), n)
    #     L += w_b * inner(g, traction(mu, q, v, n)) * ds

    return fem.form(L)


def create_velocity_bc(velocity_subspace, g_expr=None):
    """
    Creates Dirichlet boundary conditions for the velocity vector subspace.
    """
    V_collapsed, dof_map = velocity_subspace.collapse()
    domain = velocity_subspace.mesh
    
    g = fem.Function(V_collapsed)

    #If you don't set g_expr, then the Dirichlet BC will be zero (traction-free)
    if g_expr is None:
        g.x.array[:] = 0.0
    else:
        g.interpolate(g_expr)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    bdry_facets = mesh.exterior_facet_indices(domain.topology)
    
    dofs = fem.locate_dofs_topological((velocity_subspace, V_collapsed), tdim - 1, bdry_facets)
    return [fem.dirichletbc(g, dofs, velocity_subspace)]

# ---------------------------------------------------------------------
# Solver (MMS: no velocity Dirichlet, no pressure pin; rigid-body pins)
# ---------------------------------------------------------------------
# def setup_and_solve_stokes(
#     problem,
#     N=64,
#     degree_omega=1,
#     degree_p=1,
#     degree_u=1
# ):
#     print(
#         f"Initializing mesh ({N}x{N}), "
#         f"P{degree_omega}/P{degree_p}/P{degree_u} (omega/p/u)"
#     )

#     comm = MPI.COMM_WORLD
#     domain = mesh.create_unit_square(comm, N, N, mesh.CellType.triangle)

#     cell = domain.topology.cell_name()
#     element_omega = basix.ufl.element("Lagrange", cell, degree_omega)
#     element_p = basix.ufl.element("Lagrange", cell, degree_p)
#     element_u = basix.ufl.element("Lagrange", cell, degree_u, shape=(2,))
#     mixed_element = basix.ufl.mixed_element([element_omega, element_p, element_u])
#     W = fem.functionspace(domain, mixed_element)

#     #Set up pressure zero pressure constraint
#     dx = Measure("dx", domain=domain)
#     xi, q, v = TestFunctions(W)
#     c_form = fem.form(q*dx)
#     c = fem.petsc.assemble_vector(c_form)
#     c.ghostUpdate(addv=PETSc.InsertMode.ADD, 
#                   mode=PETSc.ScatterMode.REVERSE)

#     mu = 1.0

#     print("Assembling LSFEM system...")
#     a_form = create_bilinear_form(mu, W)
#     L_form = create_linear_form(mu, W)

#     # Boundary conditions
#     if problem == "cavity":
#         velocity_bcs = create_cavity_velocity_bcs(W.sub(2), U_0=1.0)
#         pressure_bcs = create_pressure_pin(W.sub(1))
#         bcs = velocity_bcs + pressure_bcs
#     else:
#         bcs = create_velocity_bc(W.sub(2))

#     # Assemble system without pressure shift
#     print("Before matrix assembly", flush=True)
#     A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
#     A.assemble()
#     print("After matrix assembly", flush=True)

#     b = dolfinx.fem.petsc.assemble_vector(L_form)

#     print("After vector assembly", flush=True)

#     dolfinx.fem.petsc.apply_lifting(b, [a_form], [bcs])
#     b.ghostUpdate(
#         addv=PETSc.InsertMode.ADD,
#         mode=PETSc.ScatterMode.REVERSE
#     )
#     dolfinx.fem.petsc.set_bc(b, bcs)
#     b.ghostUpdate(
#         addv=PETSc.InsertMode.INSERT,
#         mode=PETSc.ScatterMode.FORWARD
#     )

#     #Assemble system with pressure shift
#     print("Building sparse augmented system...", flush=True)

def setup_and_solve_stokes(
    problem,
    N=64,
    degree_omega=1,
    degree_p=1,
    degree_u=1,
):
    import numpy as np
    import dolfinx
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI
    from petsc4py import PETSc

    # ------------------------------------------------------------
    # Mesh and mixed finite-element space
    # ------------------------------------------------------------
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        N,
        N,
        cell_type=mesh.CellType.triangle,
    )

    cell = domain.basix_cell()

    element_omega = basix.ufl.element(
        "Lagrange", cell, degree_omega
    )
    element_p = basix.ufl.element(
        "Lagrange", cell, degree_p
    )
    element_u = basix.ufl.element(
        "Lagrange", cell, degree_u, shape=(2,)
    )

    mixed_element = basix.ufl.mixed_element(
        [element_omega, element_p, element_u]
    )
    W = fem.functionspace(domain, mixed_element)

    # ------------------------------------------------------------
    # Forms and boundary conditions
    # Keep the same form definitions as in your existing code.
    # ------------------------------------------------------------
    mu = 1.0
    a_form = create_bilinear_form(mu, W)
    L_form = create_linear_form(mu, W)

    # This version is for the non-cavity / else case:
    # impose velocity BCs, but do not pin the pressure.
    bcs = create_velocity_bc(W.sub(2))

    # ------------------------------------------------------------
    # Assemble the original sparse system
    # ------------------------------------------------------------
    A = fem.petsc.assemble_matrix(
        fem.form(a_form),
        bcs=bcs,
    )
    A.assemble()

    b = fem.petsc.assemble_vector(fem.form(L_form))

    fem.petsc.apply_lifting(
        b,
        [fem.form(a_form)],
        bcs=[bcs],
    )
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    fem.petsc.set_bc(b, bcs)

    # ------------------------------------------------------------
    # Assemble the pressure-mean constraint vector c:
    #     c_i = integral of the i-th pressure basis function.
    # ------------------------------------------------------------
    dx = ufl.Measure("dx", domain=domain)
    _, q, _ = ufl.TestFunctions(W)

    c_form = fem.form(q * dx)
    c = fem.petsc.assemble_vector(c_form)
    c.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )

    # ------------------------------------------------------------
    # Sparse augmented system:
    #
    #       [ A   c ] [ x      ]   [ b ]
    #       [ c^T 0 ] [ lambda ] = [ 0 ]
    #
    # This implementation assumes a serial run.
    # ------------------------------------------------------------
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "This augmented-system implementation is serial-only."
        )

    n = A.getSize()[0]

    c_array = c.getArray(readonly=True).copy()
    b_array = b.getArray(readonly=True).copy()

    A_aug = PETSc.Mat().createAIJ(
        size=(n + 1, n + 1),
        nnz=100,
        comm=PETSc.COMM_SELF,
    )
    A_aug.setUp()

    A_aug.setOption(
    PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,
    False,)

    # Copy A into the upper-left block.
    # for i in range(n):
    #     cols, vals = A.getRow(i)
    #     try:
    #         A_aug.setValues(i, cols, vals)
    #     finally:
    #         A.restoreRow(i, cols, vals)
    # Copy A into the upper-left block of the augmented matrix.
    
    for i in range(n):
        cols, vals = A.getRow(i)
        A_aug.setValues(i, cols, vals)

    # Add the constraint column and row.
    # The constraint vector is zero on non-pressure DOFs.
    for i, ci in enumerate(c_array):
        if ci != 0.0:
            A_aug.setValue(i, n, ci)
            A_aug.setValue(n, i, ci)

    A_aug.assemble()

    # Augmented right-hand side.
    b_aug = PETSc.Vec().createSeq(n + 1, comm=PETSc.COMM_SELF)
    b_aug.setValues(np.arange(n, dtype=PETSc.IntType), b_array)
    b_aug.setValue(n, 0.0)
    b_aug.assemble()

    # ------------------------------------------------------------
    # Solve the sparse augmented system with PETSc
    # ------------------------------------------------------------
    solution_vec = b_aug.duplicate()

    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setOperators(A_aug)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")

    # Request MUMPS, as in your existing solver, if available.
    ksp.getPC().setFactorSolverType("mumps")

    ksp.solve(b_aug, solution_vec)

    reason = ksp.getConvergedReason()
    print(f"PETSc converged reason: {reason}", flush=True)

    if reason < 0:
        raise RuntimeError("PETSc failed to solve the augmented system.")

    # ------------------------------------------------------------
    # Transfer the finite-element portion of the solution
    # ------------------------------------------------------------
    solution = solution_vec.getArray(readonly=True).copy()

    w = fem.Function(W)
    w.x.array[:] = solution[:n]
    w.x.scatter_forward()

    # ------------------------------------------------------------
    # Verify the zero-mean pressure constraint
    # ------------------------------------------------------------
    pressure_integral_local = fem.assemble_scalar(
        fem.form(w.sub(1) * dx)
    )
    pressure_integral = MPI.COMM_WORLD.allreduce(
        pressure_integral_local,
        op=MPI.SUM,
    )

    print(
        "Pressure integral (should be zero):",
        pressure_integral,
        flush=True,
    )

    # ------------------------------------------------------------
    # Clean up PETSc objects
    # ------------------------------------------------------------
    ksp.destroy()
    solution_vec.destroy()
    b_aug.destroy()
    A_aug.destroy()
    A.destroy()
    b.destroy()
    c.destroy()

    print("Solve complete!", flush=True)

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


# def run_mms_space_study():
#     """
#     Run MMS mesh-refinement studies for multiple combinations
#     of polynomial orders.

#     Tuple:
#         (k_omega, k_p, k_u)

#     denotes

#         P{k_omega} / P{k_p} / P{k_u}.
#     """

#     comm = MPI.COMM_WORLD

#     # =========================================================
#     # Verify MMS independently before solving anything
#     # =========================================================

#     if comm.rank == 0:
#         print("\nChecking manufactured solution...")

#     verify_mms_manufactured_solution(N=8)

#     # =========================================================
#     # Mesh resolutions
#     # =========================================================

#     grid_sizes = [8, 16, 32, 64]

#     # =========================================================
#     # Polynomial-order combinations
#     # =========================================================

#     test_cases = [
#         (1, 1, 1),
#         (2, 1, 1),
#         (1, 2, 1),
#         (1, 1, 2),
#         (2, 2, 1),
#         (2, 1, 2),
#         (1, 2, 2),
#         (2, 2, 2),
#     ]

#     results = {}

#     # =========================================================
#     # Loop over polynomial spaces
#     # =========================================================

#     for degrees in test_cases:

#         k_omega, k_p, k_u = degrees

#         label = (
#             f"P{k_omega}/P{k_p}/P{k_u}"
#         )

#         if comm.rank == 0:
#             print("\n")
#             print("=" * 80)
#             print(f"TEST CASE: {label}")
#             print("omega / pressure / velocity")
#             print("=" * 80)

#         # -----------------------------------------------------
#         # Error arrays
#         # -----------------------------------------------------

#         u_errors = []
#         u_h1_errors = []
#         p_errors = []
#         p_grad_errors = []
#         omega_errors = []

#         p_mean_errors = []
#         p_gauge_errors = []

#         # =====================================================
#         # Mesh refinement
#         # =====================================================

#         for N in grid_sizes:

#             if comm.rank == 0:
#                 print(
#                     f"\nRunning {label}, N = {N}"
#                 )

#             w_solution, W = setup_and_solve_stokes(
#                 problem="mms",
#                 N=N,
#                 degree_omega=k_omega,
#                 degree_p=k_p,
#                 degree_u=k_u
#             )

#             (
#                 u_error,
#                 u_h1_semi_error,
#                 u_h1_error,
#                 p_error,
#                 p_grad_error,
#                 omega_error,
#                 p_mean_error,
#                 p_gauge_error
#             ) = compute_errors(
#                 w_solution,
#                 W,
#                 Stokes_Problem
#             )

#             # Store errors
#             u_errors.append(u_error)
#             u_h1_errors.append(u_h1_error)
#             p_errors.append(p_error)
#             p_grad_errors.append(p_grad_error)
#             omega_errors.append(omega_error)

#             p_mean_errors.append(p_mean_error)
#             p_gauge_errors.append(p_gauge_error)

#             if comm.rank == 0:

#                 print(
#                     f"  u          = {u_error:.6e}"
#                 )

#                 print(
#                     f"  p          = {p_error:.6e}"
#                 )

#                 print(
#                     f"  grad(p)    = {p_grad_error:.6e}"
#                 )

#                 print(
#                     f"  omega      = {omega_error:.6e}"
#                 )

#                 print(
#                     f"  mean error = {p_mean_error:.6e}"
#                 )

#                 print(
#                     f"  p gauge-corrected = "
#                     f"{p_gauge_error:.6e}"
#                 )

#         # =====================================================
#         # Convert to arrays
#         # =====================================================

#         u_errors = np.asarray(u_errors)
#         p_errors = np.asarray(p_errors)
#         p_grad_errors = np.asarray(p_grad_errors)
#         omega_errors = np.asarray(omega_errors)

#         p_mean_errors = np.asarray(p_mean_errors)
#         p_gauge_errors = np.asarray(p_gauge_errors)

#         # =====================================================
#         # Store results
#         # =====================================================

#         results[label] = {
#             "degrees": degrees,
#             "N": np.asarray(grid_sizes),

#             "u": u_errors,
#             "u_h1": u_h1_errors,
#             "p": p_errors,
#             "p_grad": p_grad_errors,
#             "omega": omega_errors,

#             "p_mean": p_mean_errors,
#             "p_gauge": p_gauge_errors
#         }

#         # =====================================================
#         # Observed convergence rates
#         # =====================================================

#         if comm.rank == 0:

#             print("\nObserved convergence rates")
#             print("-" * 75)

#             for i in range(1, len(grid_sizes)):

#                 rate_u = (
#                     np.log(
#                         u_errors[i - 1] /
#                         u_errors[i]
#                     )
#                     / np.log(2.0)
#                 )

#                 rate_p = (
#                     np.log(
#                         p_errors[i - 1] /
#                         p_errors[i]
#                     )
#                     / np.log(2.0)
#                 )

#                 rate_p_grad = (
#                     np.log(
#                         p_grad_errors[i - 1] /
#                         p_grad_errors[i]
#                     )
#                     / np.log(2.0)
#                 )

#                 rate_omega = (
#                     np.log(
#                         omega_errors[i - 1] /
#                         omega_errors[i]
#                     )
#                     / np.log(2.0)
#                 )

#                 print(
#                     f"N={grid_sizes[i-1]:2d} -> "
#                     f"N={grid_sizes[i]:2d}: "
#                     f"u={rate_u:.3f}, "
#                     f"p={rate_p:.3f}, "
#                     f"grad(p)={rate_p_grad:.3f}, "
#                     f"omega={rate_omega:.3f}"
#                 )

#     # =========================================================
#     # PLOTS
#     # =========================================================

#     if comm.rank == 0:

#         # =====================================================
#         # 1. Individual plots
#         # =====================================================

#         # for label, data in results.items():

#         #     N = data["N"]

#         #     plt.figure(figsize=(8, 6))

#         #     plt.loglog(
#         #         N,
#         #         data["u"],
#         #         "o-",
#         #         linewidth=2,
#         #         markersize=7,
#         #         label=r"$\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$"
#         #     )

#         #     plt.loglog(
#         #         N,
#         #         data["u_h1"],
#         #         "o-",
#         #         linewidth=2,
#         #         markersize=7,
#         #         label=r"$\|\mathbf{u}-\mathbf{u}_h\|_{H^1}$"
#         #     )

#         #     plt.loglog(
#         #         N,
#         #         data["p"],
#         #         "s-",
#         #         linewidth=2,
#         #         markersize=7,
#         #         label=r"$\|p-p_h\|_{L^2}$"
#         #     )

#         #     plt.loglog(
#         #         N,
#         #         data["p_grad"],
#         #         "d-",
#         #         linewidth=2,
#         #         markersize=7,
#         #         label=r"$\|\nabla p-\nabla p_h\|_{L^2}$"
#         #     )

#         #     plt.loglog(
#         #         N,
#         #         data["omega"],
#         #         "^-",
#         #         linewidth=2,
#         #         markersize=7,
#         #         label=r"$\|\omega-\omega_h\|_{L^2}$"
#         #     )

#         #     plt.xlabel(r"Mesh resolution $N$")
#         #     plt.ylabel("Error")

#         #     plt.title(
#         #         f"MMS Convergence: {label}"
#         #     )

#         #     plt.grid(
#         #         True,
#         #         which="both",
#         #         linestyle="--",
#         #         alpha=0.6
#         #     )

#         #     plt.legend()

#         #     plt.tight_layout()

#         # =====================================================
#         # 2. Velocity ($L^2$)
#         # =====================================================

#         plt.figure(figsize=(9, 7))

#         for label, data in results.items():

#             plt.loglog(
#                 data["N"],
#                 data["u"],
#                 "o-",
#                 linewidth=2,
#                 markersize=6,
#                 label=label
#             )

#         plt.xlabel(r"Mesh resolution $N$")
#         plt.ylabel(r"$\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$")
#         plt.title("MMS Velocity Convergence: Polynomial Space Comparison")
#         plt.grid(True, which="both", linestyle="--", alpha=0.6)
#         plt.legend(title=r"$(P_\omega,P_p,P_u)$", fontsize=9)
#         plt.tight_layout()
#         plt.savefig("mms_velocity_all_spaces.png", dpi=300)
#         plt.close()

#         # =====================================================
#         # 2b. Velocity ($H^1$) - Separate Plot using the same loop structure
#         # =====================================================

#         plt.figure(figsize=(9, 7))

#         for label, data in results.items():

#             plt.loglog(
#                 data["N"],
#                 data["u_h1"],
#                 "x-",
#                 linewidth=2,
#                 markersize=6,
#                 label=label
#             )

#         plt.xlabel(r"Mesh resolution $N$")
#         plt.ylabel(r"$\|\mathbf{u}-\mathbf{u}_h\|_{H^1}$")
#         plt.title("MMS Velocity H^1 Convergence: Polynomial Space Comparison")
#         plt.grid(True, which="both", linestyle="--", alpha=0.6)
#         plt.legend(title=r"$(P_\omega,P_p,P_u)$", fontsize=9)
#         plt.tight_layout()
#         plt.savefig("mms_velocity_h1_all_spaces.png", dpi=300)
#         plt.close()

        


#             if Stokes_Problem.weighted ==0:
#                 filename = (
#                     f"unweighted_mms_convergence_"
#                     f"{label.replace('/', '_')}.png"
#                 )

#                 plt.savefig(
#                     f"Unweighted/{filename}",
#                     dpi=300
#                 )



#             if Stokes_Problem.weighted ==1:
#                 filename = (
#                     f"weighted_mms_convergence_"
#                     f"{label.replace('/', '_')}.png"
#                 )

#                 plt.savefig(
#                     f"Weighted/{filename}",
#                     dpi=300
#                 )

#             plt.close()

#             print(
#                 f"Saved {filename}"
#             )

#         # =====================================================
#         # 2. Velocity
#         # =====================================================

#         plt.figure(figsize=(9, 7))

#         for label, data in results.items():

#             plt.loglog(
#                 data["N"],
#                 data["u"],
#                 "o-",
#                 linewidth=2,
#                 markersize=6,
#                 label=label
#             )

#         plt.xlabel(r"Mesh resolution $N$")
#         plt.ylabel(
#             r"$\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$"
#         )

#         plt.title(
#             "MMS Velocity Convergence: "
#             "Polynomial Space Comparison"
#         )

#         plt.grid(
#             True,
#             which="both",
#             linestyle="--",
#             alpha=0.6
#         )

#         plt.legend(
#             title=r"$(P_\omega,P_p,P_u)$",
#             fontsize=9
#         )

#         plt.tight_layout()

#         plt.savefig(
#             "mms_velocity_all_spaces.png",
#             dpi=300
#         )

#         plt.close()

#         # =====================================================
#         # 3. Pressure
#         # =====================================================

#         plt.figure(figsize=(9, 7))

#         for label, data in results.items():

#             plt.loglog(
#                 data["N"],
#                 data["p"],
#                 "s-",
#                 linewidth=2,
#                 markersize=6,
#                 label=label
#             )

#         plt.xlabel(r"Mesh resolution $N$")
#         plt.ylabel(
#             r"$\|p-p_h\|_{L^2}$"
#         )

#         plt.title(
#             "MMS Pressure Convergence: "
#             "Polynomial Space Comparison"
#         )

#         plt.grid(
#             True,
#             which="both",
#             linestyle="--",
#             alpha=0.6
#         )

#         plt.legend(
#             title=r"$(P_\omega,P_p,P_u)$",
#             fontsize=9
#         )

#         plt.tight_layout()

#         plt.savefig(
#             "mms_pressure_all_spaces.png",
#             dpi=300
#         )

#         plt.close()

#         # =====================================================
#         # 4. Pressure gradient
#         # =====================================================

#         plt.figure(figsize=(9, 7))

#         for label, data in results.items():

#             plt.loglog(
#                 data["N"],
#                 data["p_grad"],
#                 "d-",
#                 linewidth=2,
#                 markersize=6,
#                 label=label
#             )

#         plt.xlabel(r"Mesh resolution $N$")
#         plt.ylabel(
#             r"$\|\nabla p-\nabla p_h\|_{L^2}$"
#         )

#         plt.title(
#             "MMS Pressure Gradient Convergence: "
#             "Polynomial Space Comparison"
#         )

#         plt.grid(
#             True,
#             which="both",
#             linestyle="--",
#             alpha=0.6
#         )

#         plt.legend(
#             title=r"$(P_\omega,P_p,P_u)$",
#             fontsize=9
#         )

#         plt.tight_layout()

#         plt.savefig(
#             "mms_pressure_gradient_all_spaces.png",
#             dpi=300
#         )

#         plt.close()

#         # =====================================================
#         # 5. Vorticity
#         # =====================================================

#         plt.figure(figsize=(9, 7))

#         for label, data in results.items():

#             plt.loglog(
#                 data["N"],
#                 data["omega"],
#                 "^-",
#                 linewidth=2,
#                 markersize=6,
#                 label=label
#             )

#         plt.xlabel(r"Mesh resolution $N$")
#         plt.ylabel(
#             r"$\|\omega-\omega_h\|_{L^2}$"
#         )

#         plt.title(
#             "MMS Vorticity Convergence: "
#             "Polynomial Space Comparison"
#         )

#         plt.grid(
#             True,
#             which="both",
#             linestyle="--",
#             alpha=0.6
#         )

#         plt.legend(
#             title=r"$(P_\omega,P_p,P_u)$",
#             fontsize=9
#         )

#         plt.tight_layout()

#         plt.savefig(
#             "mms_vorticity_all_spaces.png",
#             dpi=300
#         )

#         plt.close()

#         # =====================================================
#         # 6. Gauge diagnostic
#         # =====================================================

#         plt.figure(figsize=(9, 7))

#         for label, data in results.items():

#             plt.loglog(
#                 data["N"],
#                 np.abs(data["p_mean"]),
#                 "o-",
#                 linewidth=2,
#                 markersize=6,
#                 label=label
#             )

#         plt.xlabel(r"Mesh resolution $N$")
#         plt.ylabel(
#             r"$|\operatorname{mean}(p_h-p_{\rm ex})|$"
#         )

#         plt.title(
#             "Pressure Gauge Error"
#         )

#         plt.grid(
#             True,
#             which="both",
#             linestyle="--",
#             alpha=0.6
#         )

#         plt.legend(
#             title=r"$(P_\omega,P_p,P_u)$",
#             fontsize=9
#         )

#         plt.tight_layout()

#         plt.savefig(
#             "mms_pressure_gauge_error.png",
#             dpi=300
#         )

#         plt.close()

#         # =====================================================
#         # 7. Raw vs gauge-corrected pressure
#         # =====================================================

#         plt.figure(figsize=(9, 7))

#         for label, data in results.items():

#             plt.loglog(
#                 data["N"],
#                 data["p_gauge"],
#                 "o--",
#                 linewidth=2,
#                 markersize=5,
#                 label=label
#             )

#         plt.xlabel(r"Mesh resolution $N$")
#         plt.ylabel(
#             r"Gauge-corrected $\|p-p_h\|_{L^2}$"
#         )

#         plt.title(
#             "Gauge-Corrected Pressure Error"
#         )

#         plt.grid(
#             True,
#             which="both",
#             linestyle="--",
#             alpha=0.6
#         )

#         plt.legend(
#             title=r"$(P_\omega,P_p,P_u)$",
#             fontsize=9
#         )

#         plt.tight_layout()

#         plt.savefig(
#             "mms_pressure_gauge_corrected.png",
#             dpi=300
#         )

#         plt.close()

#         # =====================================================
#         # FINAL CONVERGENCE SUMMARY
#         # =====================================================

#         print("\n")
#         print("=" * 100)
#         print("FINAL MMS CONVERGENCE-RATE SUMMARY")
#         print("=" * 100)

#         print(
#             f"{'Space':<14}"
#             f"{'u':>10}"
#             f"{'p':>10}"
#             f"{'grad(p)':>12}"
#             f"{'omega':>12}"
#             f"{'p-gauge':>12}"
#         )

#         print("-" * 100)

#         for label, data in results.items():

#             u = data["u"]
#             p = data["p"]
#             p_grad = data["p_grad"]
#             omega = data["omega"]
#             p_gauge = data["p_gauge"]

#             # Finest-mesh convergence rate

#             rate_u = (
#                 np.log(u[-2] / u[-1])
#                 / np.log(2.0)
#             )

#             rate_p = (
#                 np.log(p[-2] / p[-1])
#                 / np.log(2.0)
#             )

#             rate_p_grad = (
#                 np.log(p_grad[-2] / p_grad[-1])
#                 / np.log(2.0)
#             )

#             rate_omega = (
#                 np.log(omega[-2] / omega[-1])
#                 / np.log(2.0)
#             )

#             rate_p_gauge = (
#                 np.log(p_gauge[-2] / p_gauge[-1])
#                 / np.log(2.0)
#             )

#             print(
#                 f"{label:<14}"
#                 f"{rate_u:>10.3f}"
#                 f"{rate_p:>10.3f}"
#                 f"{rate_p_grad:>12.3f}"
#                 f"{rate_omega:>12.3f}"
#                 f"{rate_p_gauge:>12.3f}"
#             )

#         print("=" * 100)

#     return results

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
        u_h1_errors = []
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
                u_h1_semi_error,
                u_h1_error,
                p_error,
                p_grad_error,
                omega_error,
                p_mean_error,
                p_gauge_error
            ) = compute_errors(
                w_solution,
                W,
                Stokes_Problem
            )

            # Store errors
            u_errors.append(u_error)
            u_h1_errors.append(u_h1_error)
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
                    f"  u (H1)     = {u_h1_error:.6e}"
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
        u_h1_errors = np.asarray(u_h1_errors)
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
            "u_h1": u_h1_errors,
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

                rate_u_h1 = (
                    np.log(
                        u_h1_errors[i - 1] /
                        u_h1_errors[i]
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
                    f"u(H1)={rate_u_h1:.3f}, "
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
                data["u_h1"],
                "x-",
                linewidth=2,
                markersize=7,
                label=r"$\|\mathbf{u}-\mathbf{u}_h\|_{H^1}$"
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

            if Stokes_Problem.weighted == 0:
                filename = (
                    f"unweighted_mms_convergence_"
                    f"{label.replace('/', '_')}.png"
                )
                plt.savefig(
                    f"Unweighted/{filename}",
                    dpi=300
                )

            if Stokes_Problem.weighted == 1:
                filename = (
                    f"weighted_mms_convergence_"
                    f"{label.replace('/', '_')}.png"
                )
                plt.savefig(
                    f"Weighted/{filename}",
                    dpi=300
                )

            plt.close()

            print(f"Saved {filename}")

        # =====================================================
        # 2. Velocity ($L^2$) All Spaces
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
            "MMS Velocity L^2 Convergence: "
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
        # 2b. Velocity ($H^1$) All Spaces (Separate Plot)
        # =====================================================

        plt.figure(figsize=(9, 7))

        for label, data in results.items():

            plt.loglog(
                data["N"],
                data["u_h1"],
                "x-",
                linewidth=2,
                markersize=6,
                label=label
            )

        plt.xlabel(r"Mesh resolution $N$")
        plt.ylabel(
            r"$\|\mathbf{u}-\mathbf{u}_h\|_{H^1}$"
        )

        plt.title(
            "MMS Velocity H^1 Convergence"
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
            "mms_velocity_h1_all_spaces.png",
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
            "MMS Pressure Convergence"
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
            "MMS Pressure Gradient Convergence"
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
            "MMS Vorticity Convergence"
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



def compute_errors(w_solution, W, stokes_problem):
    """
    Compute MMS errors:

        ||u - u_h||_L2
        ||u - u_h||_H1 (semi-norm and full norm)
        ||p - p_h||_L2
        ||grad(p) - grad(p_h)||_L2 (Pressure H1 semi-norm)
        ||omega - omega_h||_L2

    Also computes pressure-gauge diagnostics:
        - mean(p_h - p_exact)
        - gauge-corrected pressure L2 error
    """

    domain = W.mesh
    comm = domain.comm
    x = SpatialCoordinate(domain)
    dx = Measure("dx", domain=domain)

    # =========================================================
    # VELOCITY (L2 and H1 errors)
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

    # Velocity L2 error
    u_error_sq = fem.assemble_scalar(
        fem.form(
            inner(
                u_h - u_exact,
                u_h - u_exact
            ) * dx
        )
    )
    u_L2_error = np.sqrt(
        comm.allreduce(u_error_sq, op=MPI.SUM)
    )

    # Velocity H1 semi-norm error (|| grad(u - u_h) ||_L2)
    u_H1_semi_error_sq = fem.assemble_scalar(
        fem.form(
            inner(
                grad(u_h) - grad(u_exact),
                grad(u_h) - grad(u_exact)
            ) * dx
        )
    )
    u_H1_semi_error = np.sqrt(
        comm.allreduce(u_H1_semi_error_sq, op=MPI.SUM)
    )

    # Velocity full H1 error (sqrt(||u - u_h||_L2^2 + ||grad(u - u_h)||_L2^2))
    u_H1_error = np.sqrt(u_L2_error**2 + u_H1_semi_error**2)

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

    # ---------------------------------------------------------
    # Raw pressure L2 error
    # ---------------------------------------------------------
    p_error_sq = fem.assemble_scalar(
        fem.form(
            p_error * p_error * dx
        )
    )
    p_L2_error = np.sqrt(
        comm.allreduce(p_error_sq, op=MPI.SUM)
    )

    # ---------------------------------------------------------
    # Pressure gradient error (Equivalent to Pressure H1 semi-norm)
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
        comm.allreduce(grad_p_error_sq, op=MPI.SUM)
    )

    # ---------------------------------------------------------
    # Pressure mean / gauge diagnostic
    # ---------------------------------------------------------
    pressure_error_mean_local = fem.assemble_scalar(
        fem.form(p_error * dx)
    )
    domain_measure_local = fem.assemble_scalar(
        fem.form(1.0 * dx)
    )

    pressure_error_mean = (
        comm.allreduce(pressure_error_mean_local, op=MPI.SUM)
        /
        comm.allreduce(domain_measure_local, op=MPI.SUM)
    )

    # ---------------------------------------------------------
    # Gauge-corrected pressure L2 error
    # ---------------------------------------------------------
    p_gauge_error_sq = fem.assemble_scalar(
        fem.form(
            (p_error - pressure_error_mean)
            *
            (p_error - pressure_error_mean)
            * dx  # Fixed from - dx
        )
    )
    p_gauge_L2_error = np.sqrt(
        comm.allreduce(p_gauge_error_sq, op=MPI.SUM)
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
            (omega_h - omega_exact)
            *
            (omega_h - omega_exact)
            * dx  # Fixed from - dx
        )
    )
    omega_L2_error = np.sqrt(
        comm.allreduce(omega_error_sq, op=MPI.SUM)
    )

    return (
        u_L2_error,
        u_H1_semi_error,
        u_H1_error,
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

import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_study(h_values, error_records, filename="stokes_convergence.png"):
    """
    Generates a log-log scatter line plot of L^2 and H^1 errors versus mesh size h.
    
    Parameters:
    -----------
    h_values : list or np.ndarray
        List of mesh sizes (e.g., [1/4, 1/8, 1/16, 1/32]).
    error_records : dict
        Dictionary containing error lists for each metric:
        {
            'u_L2': [...], 'u_H1': [...],
            'p_L2': [...], 'p_H1_semi': [...],
            'omega_L2': [...], 'omega_H1': [...]
        }
    filename : str
        Output filename for the saved plot.
    """
    # Set up a 1x3 subplot grid (Velocity, Pressure, Vorticity)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    h_arr = np.array(h_values)

    # =========================================================
    # 1. VELOCITY ERRORS
    # =========================================================
    axes[0].loglog(h_arr, error_records['u_L2'], 'o-', color='tab:blue', label=r'$||\mathbf{u} - \mathbf{u}_h||_{L^2}$')
    axes[0].loglog(h_arr, error_records['u_H1'], 's--', color='tab:blue', label=r'$||\mathbf{u} - \mathbf{u}_h||_{H^1}$')
    axes[0].set_title("Velocity Convergence", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Mesh Size ($h$)", fontsize=12)
    axes[0].set_ylabel("Error Norm", fontsize=12)
    axes[0].grid(True, which="both", ls="--", alpha=0.6)
    axes[0].legend(loc="best", fontsize=10)

    # =========================================================
    # 2. PRESSURE ERRORS
    # =========================================================
    axes[1].loglog(h_arr, error_records['p_L2'], 'o-', color='tab:orange', label=r'$||p - p_h||_{L^2}$')
    axes[1].loglog(h_arr, error_records['p_H1_semi'], 's--', color='tab:orange', label=r'$||\nabla(p - p_h)||_{L^2}$ ($H^1$ semi)')
    axes[1].set_title("Pressure Convergence", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Mesh Size ($h$)", fontsize=12)
    axes[1].set_ylabel("Error Norm", fontsize=12)
    axes[1].grid(True, which="both", ls="--", alpha=0.6)
    axes[1].legend(loc="best", fontsize=10)

    # =========================================================
    # 3. VORTICITY ERRORS
    # =========================================================
    axes[2].loglog(h_arr, error_records['omega_L2'], 'o-', color='tab:green', label=r'$||\omega - \omega_h||_{L^2}$')
    axes[2].loglog(h_arr, error_records['omega_H1'], 's--', color='tab:green', label=r'$||\omega - \omega_h||_{H^1}$')
    axes[2].set_title("Vorticity Convergence", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Mesh Size ($h$)", fontsize=12)
    axes[2].set_ylabel("Error Norm", fontsize=12)
    axes[2].grid(True, which="both", ls="--", alpha=0.6)
    axes[2].legend(loc="best", fontsize=10)

    # Optional: Add reference slope guides (e.g., O(h^2) or O(h^3)) in a corner if helpful
    # c = error_records['u_L2'][0] / (h_arr[0]**2)
    # axes[0].loglog(h_arr, c * h_arr**2, 'k:', label='$O(h^2)$')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Convergence study plot saved to {filename}")

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

# for n in [4, 8, 16, 32]:
#     # 1. Create mesh with resolution corresponding to h ~ 1/n
#     mesh, W = create_stokes_mesh_and_space(n)
#     h_values.append(1.0 / n)
    
#     # 2. Solve forward problem
#     w_solution = solve_stokes(W, Stokes_Problem)
    
#     # 3. Compute errors using your error function
#     u_L2, u_H1_semi, u_H1, p_L2, p_grad_L2, omega_L2, _, _ = compute_errors(w_solution, W, Stokes_Problem)
    
#     # For vorticity H1, if you compute it similarly:
#     # omega_H1 = np.sqrt(omega_L2**2 + omega_H1_semi**2)
    
#     # Append to records
#     error_records['u_L2'].append(u_L2)
#     error_records['u_H1'].append(u_H1)
#     error_records['p_L2'].append(p_L2)
#     error_records['p_H1_semi'].append(p_grad_L2)
#     error_records['omega_L2'].append(omega_L2)
#     # error_records['omega_H1'].append(omega_H1)

# Generate the final convergence plot for your paper
# plot_convergence_study(h_values, error_records, filename="stokes_p2p1_convergence.png")


if __name__ == "__main__":

    if problem in ["mms_smooth", "mms_localized"]:

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
        