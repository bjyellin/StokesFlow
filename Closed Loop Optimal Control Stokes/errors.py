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
