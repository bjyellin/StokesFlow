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

