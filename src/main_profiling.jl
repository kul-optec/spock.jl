import LinearAlgebra as LA

using spock, Random

# Random.seed!(1)

# Prediction horizon and branching factor
N = 10; d = 2

# Dimensions of state and input vectors
nx = 2; nu = 1

# Scenario tree definition
scen_tree = spock.generate_scenario_tree_uniform_branching_factor_v2(N, d, nx, nu)

# Cost definition (Quadratic, positive definite)
cost = spock.Cost(
  # Q matrices
  collect([
    LA.Matrix([(2.2) 0; 0 (3.7)]) / 3.7 for i in 1:scen_tree.n - 1
  ]),
  # R matrices
  collect([
    reshape([(3.2)], 1, 1) / 3.7 for i in 1:scen_tree.n - 1
  ]),
  # QN matrices
  collect([
    LA.Matrix([(2.2) 0; 0 (3.7)]) / 3.7 for i in 1:scen_tree.n_leaf_nodes
  ])
)

  # # Cost definition (Quadratic, positive definite)
  # Qs = collect([
  #   rand(nx, nx) for i in 1:scen_tree.n - 1
  # ])
  # Rs = collect([
  #   rand(nu, nu) for i in 1:scen_tree.n - 1
  # ])
  # QNs = collect([
  #   rand(nx, nx) for i in 1:scen_tree.n_leaf_nodes
  # ])

  # cost = spock.Cost(
  #   # Q matrices
  #   map(x -> x' * x, Qs),
  #   # R matrices
  #   map(x -> x' * x, Rs),
  #   # QN matrices
  #   map(x -> x' * x, QNs)
  # )

# Dynamics (based on a discretised car model)
T_s = 0.1
A = [[[1.,0.] [T_s, 1.0 - (i - 1) / d * T_s]] for i in 1:d]
B = [reshape([0., T_s], :, 1) for _ in 1:d]
dynamics = spock.Dynamics(A, B)

# Risk measures: AV@R
p_ref = [0.3, 0.7]; alpha=0.95
rms = spock.get_uniform_rms_avar_v2(p_ref, alpha, d, N);
# rms = spock.get_nonuniform_rms_avar_v2(d, N);

# Box constraints
constraints = spock.UniformRectangle(
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

factor = 1e0

cp_model = spock.build_model(scen_tree, cost, dynamics, rms, constraints, spock.SolverOptions(spock.SP, spock.AA))
# @profview spock.solve_model!(cp_model, [0.1, .1], tol=1e-3)
@time spock.solve_model!(cp_model, [0.1, .1] / factor, tol=1e-3 / factor)

cp_model = spock.build_model(scen_tree, cost, dynamics, rms, constraints, spock.SolverOptions(spock.SP, spock.RB))
# @profview spock.solve_model!(cp_model, [0.1, .1], tol=1e-3)
@time spock.solve_model!(cp_model, [0.1, .1] / factor, tol=1e-3 / factor)

cp_model = spock.build_model(scen_tree, cost, dynamics, rms, constraints, spock.SolverOptions(spock.CP, nothing))
# @profview spock.solve_model!(cp_model, [0.1, .1], tol=1e-3)
@time spock.solve_model!(cp_model, [0.1, .1] / factor, tol=1e-3 / factor)