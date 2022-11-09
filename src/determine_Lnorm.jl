import MathOptInterface as MOI
import MathOptSetDistances as MOD
import LinearAlgebra as LA

using JuMP, SparseArrays, Test

include("cost.jl")
include("dynamics.jl")
include("scenario_tree.jl")
include("risk_measures.jl")
include("model.jl")

## The supported algorithms
include("model_algorithms/cp.jl")
include("model_algorithms/sp.jl")

# Quasi-Newton directions
include("model_algorithms/qnewton_directions.jl/restarted_broyden.jl")

## The supported dynamics
include("model_dynamics/dynamics_in_L.jl")
include("model_dynamics/implicit_l.jl")

## The specific models
include("models/cpock.jl")
include("models/spock.jl")
include("models/model_cp_implicit.jl")
include("models/model_sp_implicit.jl")
include("models/model_mosek.jl")

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
    0.1 * LA.Matrix([(2.2) 0; 0 (3.7)]) for i in 1:scen_tree.n - 1
  ]),
  # R matrices
  collect([
    0.1 * reshape([(3.2)], 1, 1) for i in 1:scen_tree.n - 1
  ]),
  # QN matrices
  collect([
    0.1 * LA.Matrix([(2.2) 0; 0 (3.7)]) for i in 1:scen_tree.n_leaf_nodes
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

###########################
### Solving the given problem
###########################

sp_model = build_model(scen_tree, cost, dynamics, rms, SolverOptions(L_IMPLICIT, SP))

I = Matrix(1. * LA.I(sp_model.solver_state.nz))
L = zeros(sp_model.solver_state.nv, sp_model.solver_state.nz)

for i = 1:sp_model.solver_state.nz
  L!(sp_model, view(I, :, i), view(L, :, i))
end

maximum(LA.svdvals(collect(L)))^2