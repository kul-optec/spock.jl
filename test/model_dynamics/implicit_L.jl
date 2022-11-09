# Prediction horizon and branching factor
N = 3; d = 2

# Dimensions of state and input vectors
nx = 2; nu = 1

# Scenario tree definition
scen_tree = generate_scenario_tree_uniform_branching_factor_v2(N, d, nx, nu)

# Cost definition (Quadratic, positive definite)
cost = CostV2(
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

cp_model = build_model(scen_tree, cost, dynamics, rms, SolverOptions(L_IMPLICIT, CP))
solve_model!(cp_model, [0.1, .1])

x = cp_model.solver_state.z[1 : 14]
u = cp_model.solver_state.z[15: 17]
s = cp_model.solver_state.z[18:24]
tau = cp_model.solver_state.z[25:30]
y = cp_model.solver_state.z[31:45]

@testset "The solution satisfies the dynamics exactly" begin
  @test isapprox(x[3:4], dynamics.A[1] * x[1:2] + dynamics.B[1] * u[1])
  @test isapprox(x[5:6], dynamics.A[2] * x[1:2] + dynamics.B[2] * u[1])
  @test isapprox(x[7:8], dynamics.A[1] * x[3:4] + dynamics.B[1] * u[2])
  @test isapprox(x[9:10], dynamics.A[2] * x[3:4] + dynamics.B[2] * u[2])
  @test isapprox(x[11:12], dynamics.A[1] * x[5:6] + dynamics.B[1] * u[3])
  @test isapprox(x[13:14], dynamics.A[2] * x[5:6] + dynamics.B[2] * u[3])
end

### TODO: Restructure to find a nice place for these test(s)

@testset "E' y = tau + s holds exactly" begin
  offset = 0
  for i = 1:scen_tree.n_non_leaf_nodes
    children_of_i = scen_tree.child_mapping[i]
    b = rms[i].b; ny = length(b)
    @test isapprox(rms[i].E' * y[offset + 1 : offset + ny], tau[children_of_i .- 1] + s[children_of_i])
    offset += ny
  end
end