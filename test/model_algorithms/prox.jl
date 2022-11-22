# Prediction horizon and branching factor
N = 3; d = 2

# Dimensions of state and input vectors
nx = 2; nu = 1

# Scenario tree definition
scen_tree = generate_scenario_tree_uniform_branching_factor_v2(N, d, nx, nu)

# Cost definition (Quadratic, positive definite)
cost = Cost(
  # Q matrices
  collect([
    LA.Matrix([(2.2) 0; 0 (3.7)]) for i in 1:scen_tree.n - 1
  ]),
  # R matrices
  collect([
    reshape([(3.2)], 1, 1) for i in 1:scen_tree.n - 1
  ]),
  # QN matrices
  collect([
    LA.Matrix([(2.2) 0; 0 (3.7)]) for i in 1:scen_tree.n_leaf_nodes
  ])
)

# Dynamics (based on a discretised car model)
T_s = 0.1
A = [[[1.,0.] [T_s, 1.0 - (i - 1) / d * T_s]] for i in 1:d]
B = [reshape([0., T_s], :, 1) for _ in 1:d]
dynamics = Dynamics(A, B)

# Risk measures: AV@R
p_ref = [0.3, 0.7]; alpha=0.95
rms = get_uniform_rms_avar_v2(p_ref, alpha, d, N);

# Box constraints
constraints = UniformRectangle(
  -1.,
  1.,
  -1.,
  1.,
  scen_tree.n_leaf_nodes * nx,
  scen_tree.n_non_leaf_nodes * (nx + nu),
  nx,
  nu,
  scen_tree.n_leaf_nodes,
  scen_tree.n_non_leaf_nodes
)

cp_model = build_model(scen_tree, cost, dynamics, rms, constraints, SolverOptions(CP, nothing))
solve_model!(cp_model, [0.1, .1])

@testset "prox_f is firmly nonexpansive" begin
  for _ = 1:10
    nz = cp_model.state.nz
    prox1 = rand(nz); z1 = copy(prox1)
    prox2 = rand(nz); z2 = copy(prox2)
    gamma = 0.1

    prox_f!(cp_model, prox1, gamma)
    prox_f!(cp_model, prox2, gamma)

    @test (LA.dot(prox1 - prox2, z1 - z2) >= LA.norm(prox1 - prox2)^2) || (
      isapprox(LA.dot(prox1 - prox2, z1 - z2), LA.norm(prox1 - prox2)^2)
    )
  end
end

@testset "prox_{g^*} is firmly nonexpansive" begin
  for _ = 1:20
    nv = cp_model.state.nv
    prox1 = rand(nv); v1 = copy(prox1)
    prox2 = rand(nv); v2 = copy(prox2)
    gamma = 0.1

    prox_h_conj!(cp_model, prox1, gamma)
    prox_h_conj!(cp_model, prox2, gamma)

    @test (LA.dot(prox1 - prox2, v1 - v2) >= LA.norm(prox1 - prox2)^2)
  end
end