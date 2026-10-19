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