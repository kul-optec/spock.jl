function get_server_heat_specs(
  N :: TI,
  nx :: TI,
  d :: TI
) where {TI <: Integer}

"""
nx === nu



"""

  nu = nx

  # if (d > 2)
  #   error("Implement this model for d > 2.")
  #   # More specifically, p_ref should be adapted, as well as the dynamics
  # end

  scen_tree = spock.generate_scenario_tree_uniform_branching_factor_v2(N, d, nx, nx)

  # Cost definition (Quadratic, positive definite)
  cost = spock.Cost(
    # Q matrices
    collect([
      LA.I(nx) * 1e-1 for i in 1:scen_tree.n - 1
    ]),
    # R matrices
    collect([
      LA.I(nx) * 1. for i in 1:scen_tree.n - 1
    ]),
    # QN matrices
    collect([
      LA.I(nx) * 1e-1 for i in 1:scen_tree.n_leaf_nodes
    ])
  )

  # cost = spock.Cost(
  #   # Q matrices
  #   collect([
  #     Matrix(LA.diagm(rand(nx)) * 1e-0) for i in 1:scen_tree.n - 1
  #   ]),
  #   # R matrices
  #   collect([
  #     Matrix(LA.diagm(rand(nx)) * 1.) for i in 1:scen_tree.n - 1
  #   ]),
  #   # QN matrices
  #   collect([
  #     Matrix(LA.diagm(rand(nx)) * 1e-0) for i in 1:scen_tree.n_leaf_nodes
  #   ])
  # )

  # Qs = collect([
  #   rand(nx, nx) for i in 1:scen_tree.n - 1
  # ])
  # Rs = collect([
  #   rand(nx, nx) for i in 1:scen_tree.n - 1
  # ])
  # QNs = collect([
  #   rand(nx, nx) for i in 1:scen_tree.n_leaf_nodes
  # ])

  # cost = spock.Cost(
  #   # Q matrices
  #   map(x -> x' * x / LA.opnorm(x' * x), Qs),
  #   # R matrices
  #   map(x -> x' * x / LA.opnorm(x' * x), Rs),
  #   # QN matrices
  #   map(x -> x' * x / LA.opnorm(x' * x), QNs)
  # )

  # Dynamics
  A = [LA.diagm([1. + (1. + (j - 1) / nx) * (i - 1) / d for j in 1:nx]) for i in 1:d]
  f = 0.01
  for k = 1:d
    for i = 1:nx
      if i < nx
        A[k][i, i+1] = f
      end
      if i > 1
        A[k][i, i-1] = f
      end
    end
  end
  B = [Matrix(LA.I(nx) * 1.) for _ in 1:d]
  dynamics = spock.Dynamics(A, B)

  alpha=0.95
  if d == 2
    p_ref = [0.3, 0.7]
  else
    p_ref = spock.rand_probvec2(d)
  end
  rms = spock.get_uniform_rms_avar_v2(p_ref, alpha, d, N);

  constraints = spock.UniformRectangle(
    -1.,
    1.,
    -1.5,
    1.5,
    scen_tree.n_leaf_nodes * nx,
    scen_tree.n_non_leaf_nodes * (nx + nu),
    nx,
    nu,
    scen_tree.n_leaf_nodes,
    scen_tree.n_non_leaf_nodes
  )

  return scen_tree, cost, dynamics, rms, constraints
end

function get_server_heat_specs(
  N :: TI,
  nx :: TI,
  d :: TI,
  alpha :: TF
) where {TI <: Integer, TF <: Real}

"""
nx === nu



"""

  # if (d > 2)
  #   error("Implement this model for d > 2.")
  #   # More specifically, p_ref should be adapted, as well as the dynamics
  # end

  scen_tree = spock.generate_scenario_tree_uniform_branching_factor_v2(N, d, nx, nx)

  # Cost definition (Quadratic, positive definite)
  cost = spock.Cost(
    # Q matrices
    collect([
      Matrix(LA.I(nx) * 1e-1) for i in 1:scen_tree.n - 1
    ]),
    # R matrices
    collect([
      Matrix(LA.I(nx) * 1.) for i in 1:scen_tree.n - 1
    ]),
    # QN matrices
    collect([
      Matrix(LA.I(nx) * 1e-1) for i in 1:scen_tree.n_leaf_nodes
    ])
  )

  # cost = spock.Cost(
  #   # Q matrices
  #   collect([
  #     Matrix(LA.diagm(rand(nx)) * 1e-0) for i in 1:scen_tree.n - 1
  #   ]),
  #   # R matrices
  #   collect([
  #     Matrix(LA.diagm(rand(nx)) * 1.) for i in 1:scen_tree.n - 1
  #   ]),
  #   # QN matrices
  #   collect([
  #     Matrix(LA.diagm(rand(nx)) * 1e-0) for i in 1:scen_tree.n_leaf_nodes
  #   ])
  # )

  # Qs = collect([
  #   rand(nx, nx) for i in 1:scen_tree.n - 1
  # ])
  # Rs = collect([
  #   rand(nx, nx) for i in 1:scen_tree.n - 1
  # ])
  # QNs = collect([
  #   rand(nx, nx) for i in 1:scen_tree.n_leaf_nodes
  # ])

  # cost = spock.Cost(
  #   # Q matrices
  #   map(x -> x' * x / LA.opnorm(x' * x), Qs),
  #   # R matrices
  #   map(x -> x' * x / LA.opnorm(x' * x), Rs),
  #   # QN matrices
  #   map(x -> x' * x / LA.opnorm(x' * x), QNs)
  # )

  # Dynamics
  A = [LA.diagm([1. + (1. + (j - 1) / nx) * (i - 1) / d for j in 1:nx]) for i in 1:d]
  f = 0.01
  for k = 1:d
    for i = 1:nx
      if i < nx
        A[k][i, i+1] = f
      end
      if i > 1
        A[k][i, i-1] = f
      end
    end
  end
  B = [Matrix(LA.I(nx) * 1.) for _ in 1:d]
  dynamics = spock.Dynamics(A, B)

  if d == 2
    p_ref = [0.3, 0.7]
  else
    p_ref = spock.rand_probvec2(d)
  end
  rms = spock.get_uniform_rms_avar_v2(p_ref, alpha, d, N);

  constraints = spock.UniformRectangle(
    -1.,
    1.,
    -1.5,
    1.5,
    scen_tree.n_leaf_nodes * nx,
    scen_tree.n_non_leaf_nodes * (nx + nu),
    nx,
    nu,
    scen_tree.n_leaf_nodes,
    scen_tree.n_non_leaf_nodes
  )

  return scen_tree, cost, dynamics, rms, constraints
end