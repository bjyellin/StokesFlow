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
