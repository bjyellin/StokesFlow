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
