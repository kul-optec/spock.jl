function build_model_cp_implicitl(
  scen_tree :: ScenarioTreeV2, 
  cost :: CostV2, 
  dynamics :: Dynamics, 
  rms :: Vector{RiskMeasureV2},
)

  ### Problem definition

  # nx, nu
  nx, nu = size(dynamics.B[1])

  # x0
  x0 = zeros(nx)

  problem_definition = GENERIC_PROBLEM_DEFINITIONV2(
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
  nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_11, nv_12, nv_13 = get_nv(problem_definition)
  nv = sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_11, nv_12, nv_13])

  v5_inds = collect(sum([nv_1, nv_2, nv_3, nv_4]) + 1 : sum([nv_1, nv_2, nv_3, nv_4, nv_5]))
  v6_inds = collect(sum([nv_1, nv_2, nv_3, nv_4, nv_5]) + 1 : sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6]))
  v12_inds = collect((sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_11]) + 1 :
    sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_11, nv_12])))
  v13_inds = collect((sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_11, nv_12]) + 1 :
    sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_11, nv_12, nv_13])))


  # Offsets
  n_non_leafs = scen_tree.n_non_leaf_nodes
  n_leafs = scen_tree.n_leaf_nodes
  n = scen_tree.n

  v2_offset = length(z_to_y(problem_definition))
  v3_offset = v2_offset + n_non_leafs
  v4_offset = v3_offset + nx * (n - 1)
  v5_offset = v4_offset + nu * (n - 1)
  v6_offset = v5_offset + (n - 1)
  v11_offset = v6_offset + (n - 1)
  v12_offset = v11_offset + n_leafs * nx
  v13_offset = v12_offset + n_leafs

  # Todo: L norm must be computed efficiently!
  L_norm = 10.

  # v
  v = zeros(nv); vbar = zeros(nv)

  P, K, R_chol, ABK = ricatti_offline(problem_definition)

  ### Kernel projection
  Ms = [
    vcat(
      hcat(
        rms[i].E', -LA.I(length(scen_tree.child_mapping[i])), -LA.I(length(scen_tree.child_mapping[i]))
      ),
      hcat(
        rms[i].F', zeros(length(scen_tree.child_mapping[i]), length(scen_tree.child_mapping[i])), zeros(length(scen_tree.child_mapping[i]), length(scen_tree.child_mapping[i]))
      )
    )
    for i in 1:scen_tree.n_non_leaf_nodes
  ]

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

  # TODO: Support more general cases
  ny = length(problem_definition.rms[1].b)
  n_children = length(problem_definition.dynamics.A)

  solver_state_internal = CP_IMPLICITL_STATE_INTERNAL(
    zeros(nx),
    zeros(nx),
    zeros(nu),
    z_to_x(problem_definition),
    z_to_u(problem_definition),
    z_to_s(problem_definition),
    z_to_tau(problem_definition),
    z_to_y(problem_definition),
    map(x -> sqrt(x), cost.Q),
    map(x -> sqrt(x), cost.R),
    map(x -> sqrt(x), cost.QN),
    zeros(nz),
    zeros(nv),
    Ms,
    map(
      x -> LA.svd(LA.nullspace(x)).U * LA.pinv(LA.svd(LA.nullspace(x)).U),
      Ms
    ),
    zeros(ny + n_children * 2),
    zeros(ny + n_children * 2),
    P,
    K,
    R_chol,
    ABK,
    zeros(scen_tree.n * nx),
    zeros(scen_tree.n * nu),
    zeros(nu),
    zeros(nv),
    v5_inds,
    v6_inds,
    v12_inds,
    v13_inds,
    v2_offset,
    v3_offset,
    v4_offset,
    v5_offset,
    v6_offset,
    v11_offset,
    v12_offset,
    v13_offset,
    zeros(nx + 2),
    zeros(nx + nu + 2)
  )

  return MODEL_CP_IMPLICITL(
    solver_state,
    solver_state_internal,
    problem_definition
  )

end

################
### Model solver
################

function solve_model!(
  model :: MODEL_CP_IMPLICITL,
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