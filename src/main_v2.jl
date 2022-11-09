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
include("model_algorithms/qnewton_directions/restarted_broyden.jl")
include("model_algorithms/qnewton_directions/anderson.jl")

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
    LA.Matrix([(2.2) 0; 0 (3.7); ]) / 3.7 for i in 1:scen_tree.n - 1
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

# cp_model = build_model(scen_tree, cost, dynamics, rms, SolverOptions(L_IMPLICIT, CP))
# solve_model!(cp_model, [0.1, .1])

sp_model = build_model(scen_tree, cost, dynamics, rms, SolverOptions(L_IMPLICIT, SP))
@time solve_model!(sp_model, [.1, .1], tol=1e-3)
# typeof(sp_model)

x = sp_model.solver_state.z[sp_model.solver_state_internal.x_inds]
u = sp_model.solver_state.z[sp_model.solver_state_internal.u_inds]
s = sp_model.solver_state.z[sp_model.solver_state_internal.s_inds]
tau = sp_model.solver_state.z[sp_model.solver_state_internal.tau_inds]
y = sp_model.solver_state.z[sp_model.solver_state_internal.y_inds]

reference_model = build_model_mosek(scen_tree, cost, dynamics, rms)
# for i in eachindex(x)
#   set_start_value(reference_model[:x][i], x[i])
# end
# for i in eachindex(u)
#   set_start_value(reference_model[:u][i], u[i])
# end
# for i in eachindex(s)
#   set_start_value(reference_model[:s][i], s[i])
# end
# for i in eachindex(tau)
#   set_start_value(reference_model[:tau][i], tau[i])
# end
# for i in eachindex(y)
#   set_start_value(reference_model[:y][i], y[i])
# end

# TOL = 1e-5
# set_optimizer_attribute(reference_model, "MSK_DPAR_INTPNT_TOL_REL_GAP", TOL)
# set_optimizer_attribute(reference_model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", TOL)
# set_optimizer_attribute(reference_model, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", TOL)
@time solve_model(reference_model, [0.1, .1])

println(sp_model.solver_state.z[1:10])
println(value.(reference_model[:x][1:10]))