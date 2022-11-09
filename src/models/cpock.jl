function build_model_cp_dynamicsl(
  scen_tree :: ScenarioTreeV1, 
  cost :: CostV1, 
  dynamics :: Dynamics, 
  rms :: Vector{RiskMeasureV1},
)

### Problem definition

# nx, nu
nx, nu = size(dynamics.B[1])

# x0
x0 = zeros(nx)

problem_definition = GENERIC_PROBLEM_DEFINITIONV1(
  x0,
  nx,
  nu,
  scen_tree,
  rms,
  cost,
  dynamics,
)

### Solver state

# z
nz = get_nz(problem_definition)
z = zeros(nz); zbar = zeros(nz)

# L
nL = get_nL(problem_definition)
L = get_L(problem_definition, nz, nL)

# L norm
L_norm = maximum(LA.svdvals(collect(L)))^2

# v
nv = size(L)[1]
v = zeros(nv); vbar = zeros(nv)

solver_state = GENERIC_SOLVER_STATE(
  nz,
  nv,
  L_norm,
  z,
  v,
  zbar,
  vbar,
  zeros(nz),
  zeros(nz),
  zeros(nv),
  zeros(nz),
  zeros(nv),
  zeros(nz),
  zeros(nz),
  zeros(nv),
  ones(nz),
  ones(nv),
  zeros(nz),
  zeros(nv),
  [-Inf, -Inf]
)

### Internal solver state
inds_L_risk_a, inds_L_risk_b, inds_L_risk_c, inds_L_cost, inds_L_dynamics = get_L_inds(problem_definition, nz, nL)

solver_state_internal = CP_DYNAMICSL_STATE_INTERNAL(
  L,
  z_to_x(problem_definition),
  z_to_u(problem_definition),
  z_to_s(problem_definition),
  z_to_y(problem_definition),
  zeros(nv),
  inds_L_risk_a,
  inds_L_risk_b,
  inds_L_risk_c,
  inds_L_cost,
  inds_L_dynamics,
  zeros(nv)
)

return MODEL_CP_DYNAMICSL(
  solver_state,
  solver_state_internal,
  problem_definition
)

end

################
### Model solver
################

function solve_model!(
  model :: MODEL_CP_DYNAMICSL,
  x0 :: AbstractArray{TF, 1};
  tol :: TF = 1e-3,
  verbose :: VERBOSE_LEVEL = SILENT,
  z0 :: Union{Vector{Float64}, Nothing} = nothing, 
  v0 :: Union{Vector{Float64}, Nothing} = nothing,
  gamma :: Union{Float64, Nothing} = nothing,
  sigma :: Union{Float64, Nothing} = nothing
) where {TF <: Real}

  if z0 !== nothing || v0 !== nothing
    copyto!(model.solver_state.z, z0)
    copyto!(model.solver_state.v, v0)
  end

  copyto!(model.problem_definition.x0, x0)
  copyto!(model.solver_state.res_0, [-Inf, -Inf])

  run_cp!(model, tol=tol, verbose=verbose, sigma = sigma, gamma = gamma)
end