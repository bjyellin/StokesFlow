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

