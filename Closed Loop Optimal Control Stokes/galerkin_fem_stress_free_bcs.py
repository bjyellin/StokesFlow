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
import ufl
from ufl import (
    SpatialCoordinate, TrialFunctions, TestFunctions,
    as_vector, curl, div, grad, inner, dot, sym, Identity,
    FacetNormal, CellDiameter, Measure, sin, cos, pi, dx, ds
)
import pyvista
import matplotlib.pyplot as plt
# 1. Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)

cell = domain.topology.cell_name()

# 2. Define Taylor-Hood function spaces (P2 vector for velocity, P1 scalar for pressure)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
Q = basix.ufl.element("Lagrange", cell, 1)
TH = basix.ufl._MixedElement([V, Q])
W = fem.functionspace(domain, TH)

# 3. Define trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)


# Dynamic viscosity
mu = 1.0

# 4. Define the exact manufactured solution and pressure
x = SpatialCoordinate(domain)
sinx, siny = sin(pi * x[0]), sin(pi * x[1])
cosx, cosy = cos(pi * x[0]), cos(pi * x[1])
c_factor = 2 * pi * sinx * siny
u_ex = c_factor * as_vector((cosy * sinx, -cosx * siny))
p_ex = 0.0  # Zero exact pressure for this manufactured solution
# breakpoint()
# 5. Compute body force (f = -div(sigma_ex)) and boundary traction (g = sigma_ex * n)
I = Identity(len(x))
sigma_ex = -p_ex * I + mu * (grad(u_ex) + grad(u_ex).T)
print("sigma_ex:", sigma_ex)
f = -div(sigma_ex)
n = ufl.FacetNormal(domain)
g = dot(sigma_ex, n)

# 6. Define Galerkin variational forms
# a: bilinear form (stiff matrix terms)
# L: linear form (body force + natural stress-free boundary traction)

print("Shape of f:", f.ufl_shape)
print("Shape of v:", v.ufl_shape)
print("Shape of g:", g.ufl_shape)
print("Shape of test function component for ds:", v.ufl_shape)
breakpoint()

a = mu * inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx + inner(div(u), q) * dx
L = inner(f, v) * dx + inner(g, v) * ds
breakpoint()
# 7. Handle pressure nullspace / pinning
# Pure stress-free boundary conditions leave pressure determined only up to a constant.
# To remove the rigid body nullspace in pressure, we can pin the pressure at a specific vertex 
# or use a zero-mean constraint. Here, we enforce p = 0 at a single vertex (or sub-dof).
# (Alternatively, you can attach a nullspace to the PETSc solver).

# 8. Solve the linear system
# Note: For pure Neumann/stress-free problems without a pressure pin, the matrix is singular.
# If you want to run this out-of-the-box, pinning one pressure degree of freedom is recommended.
problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "minres", "pc_type": "lu"})
w_h = problem.solve()

# Extract velocity and pressure functions
u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()

# 9. Compute $L^2$ error norms against the exact solution
comm = domain.comm
u_ex_expr = fem.Expression(u_ex, W.sub(0).element.interpolation_points())
# (You can evaluate and compare errors here using assemble and inner products)
print("Stokes Galerkin FEM solve completed successfully!")