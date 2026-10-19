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
    if Stokes_Problem.problem == "mms":
        ds = Measure("ds", domain=domain)
        n = FacetNormal(domain)
        w_b = (1.0 / CellDiameter(domain)) if Stokes_Problem.weighted == 1 else 1.0
        g = Stokes_Problem.traction_g(SpatialCoordinate(domain), n)
        L += w_b * inner(g, traction(mu, q, v, n)) * ds

    return fem.form(L)
