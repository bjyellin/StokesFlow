class LSFEMStokesVorticityPressure:
    def __init__(self, mu=1.0, problem="mms", weighted=False):
        if problem not in ("mms", "cavity"):
            raise ValueError(f"problem must be 'mms' or 'cavity', got {problem!r}")
        self.mu = mu
        self.problem = problem          # "mms" or "cavity"
        self.weighted = int(weighted)   # 0: unweighted, 1: weighted LS functional

    def u_ex(self, x):
        sinx, siny = sin(pi * x[0]), sin(pi * x[1])
        cosx, cosy = cos(pi * x[0]), cos(pi * x[1])
        c_factor = 2 * pi * sinx * siny
        return c_factor * as_vector((cosy * sinx, -cosx * siny))

    def p_ex(self, x):
        return sin(2 * pi * x[0]) * sin(2 * pi * x[1])

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


def traction(mu, p, u, n):
    """T(p,u) = (2 mu eps(u) - p I) n"""
    return dot(2.0 * mu * sym(grad(u)) - p * Identity(2), n)
