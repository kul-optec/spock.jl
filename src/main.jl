import MathOptInterface as MOI
import MathOptSetDistances as MOD
import LinearAlgebra as LA

using JuMP, SparseArrays

include("cost.jl")
include("dynamics.jl")
include("scenario_tree.jl")
include("risk_measures.jl")
include("model.jl")

## The supported algorithms
include("model_algorithms/cp.jl")
include("model_algorithms/sp.jl")

# Quasi-Newton directions
include("model_algorithms/qnewton_directions/restarted_broyden.jl")

## The supported dynamics
include("model_dynamics/dynamics_in_L.jl")

## The specific models
include("models/cpock.jl")
include("models/spock.jl")
include("models/model_mosek.jl")

# Prediction horizon and branching factor
N = 3; d = 2

# Dimensions of state and input vectors
nx = 2; nu = 1

# Scenario tree definition
scen_tree = generate_scenario_tree_uniform_branching_factor(N, d, nx, nu)

# Cost definition (Quadratic, positive definite)
cost = CostV1(
  # Q matrices
  collect([
    LA.Matrix([(2.2) 0; 0 (3.7)]) / 3.7 for i in 1:scen_tree.n
  ]),
  # R matrices
  collect([
    reshape([(3.2)], 1, 1) / 3.7 for i in 1:scen_tree.n
  ])
)

# Dynamics (based on a discretised car model)
T_s = 0.1
A = [[[1.,0.] [T_s, 1.0 - (i - 1) / d * T_s]] for i in 1:d]
B = [reshape([0., T_s], :, 1) for _ in 1:d]
dynamics = Dynamics(A, B)

# Risk measures: AV@R
p_ref = [0.3, 0.7]; alpha=0.95
rms = get_uniform_rms_avar(p_ref, alpha, d, N);

###########################
### Solving the given problem
###########################

# cp_model = build_model(scen_tree, cost, dynamics, rms, SolverOptions(DYNAMICSL, CP))
# solve_model!(cp_model, [0.1, .1])

# sp_model = build_model(scen_tree, cost, dynamics, rms, SolverOptions(DYNAMICSL, SP))
# solve_model!(sp_model, [0.1, 0.1])
# typeof(sp_model)

reference_model = build_model_mosek(scen_tree, cost, dynamics, rms)
@time solve_model(reference_model, [0.1, .1])

println(value.(reference_model[:x][1:10]))
