function build_spock(
  scen_tree :: ScenarioTree, 
  cost :: Cost, 
  dynamics :: Dynamics, 
  rms :: Vector{RiskMeasure},
  constraints :: AbstractConvexConstraints,
  qnewton :: Union{QNewtonOptions, Nothing},
)

  ### Problem definition

  # nx, nu
  nx, nu = size(dynamics.B[1])

  # x0
  x0 = zeros(nx)

  problem = GENERIC_PROBLEM_DEFINITION(
    x0,
    nx,
    nu,
    scen_tree,
    rms,
    cost,
    dynamics,
    constraints
  )

  ### Solver state

  # z
  nz = get_nz(problem)
  z = zeros(nz); zbar = zeros(nz)

  # L
  nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_7, nv_11, nv_12, nv_13, nv_14 = get_nv(problem)
  nv = sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_7, nv_11, nv_12, nv_13, nv_14])

  v5_inds = collect(sum([nv_1, nv_2, nv_3, nv_4]) + 1 : sum([nv_1, nv_2, nv_3, nv_4, nv_5]))
  v6_inds = collect(sum([nv_1, nv_2, nv_3, nv_4, nv_5]) + 1 : sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6]))
  v12_inds = collect((sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_7, nv_11]) + 1 :
    sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_7, nv_11, nv_12])))
  v13_inds = collect((sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_7, nv_11, nv_12]) + 1 :
    sum([nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_7, nv_11, nv_12, nv_13])))


  # Offsets
  n_non_leafs = scen_tree.n_non_leaf_nodes
  n_leafs = scen_tree.n_leaf_nodes
  n = scen_tree.n

  v2_offset = length(z_to_y(problem))
  v3_offset = v2_offset + n_non_leafs
  v4_offset = v3_offset + nx * (n - 1)
  v5_offset = v4_offset + nu * (n - 1)
  v6_offset = v5_offset + (n - 1)
  v7_offset = v6_offset + (n - 1)
  v11_offset = v7_offset + constraints.nΓ_nonleaf
  v12_offset = v11_offset + n_leafs * nx
  v13_offset = v12_offset + n_leafs
  v14_offset = v13_offset + n_leafs

  # Todo: L norm must be computed efficiently!
  L_norm = 3.28

  # v
  v = zeros(nv); vbar = zeros(nv)

  P, K, R_chol, ABK = ricatti_offline(problem)

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

  MAX_BROYDEN_K = 1 # todo
  ANDERSON_BUFFER_SIZE = 3

  # TODO: Support more general cases
  ny = length(problem.rms[1].b)
  n_children = length(problem.dynamics.A)

  aa_state = AA_STATE(
    zeros(nz + nv, ANDERSON_BUFFER_SIZE),
    zeros(nz + nv, ANDERSON_BUFFER_SIZE),
    zeros(nz),
    zeros(nv),
    zeros(nz),
    zeros(nv),
    zeros(nz + nv),
    zeros(nz + nv, ANDERSON_BUFFER_SIZE),
    LA.UpperTriangular(LA.Matrix(LA.I(ANDERSON_BUFFER_SIZE) * 1.)),
    zeros(ANDERSON_BUFFER_SIZE),
  )

  solver_state_internal = SP_IMPLICITL_STATE_INTERNAL(
    zeros(nx),
    zeros(nx),
    zeros(nu),
    z_to_x(problem),
    z_to_u(problem),
    z_to_s(problem),
    z_to_tau(problem),
    z_to_y(problem),
    map(x -> sqrt(x), cost.Q),
    map(x -> sqrt(x), cost.R),
    map(x -> sqrt(x), cost.QN),
    zeros(nz),
    zeros(nv),
    # Ms,
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
    v7_offset,
    v11_offset,
    v12_offset,
    v13_offset,
    v14_offset,
    zeros(nx + 2),
    zeros(nx + nu + 2),
    # SuperMann
    zeros(nz),
    zeros(nv),
    zeros(nz),
    zeros(nv),
    zeros(nz),
    zeros(nv),
    zeros(nz),
    zeros(nv),
    zeros(nz),
    zeros(nv),
    zeros(nz),
    zeros(nz),
    zeros(nv),
    # zeros(nz),
    # zeros(nv),
    # zeros(nz),
    # zeros(nv),
    # zeros(nz),
    # zeros(nv),
    # zeros(nz),
    # zeros(nv),
    # zeros(nz * MAX_BROYDEN_K),
    # zeros(nv * MAX_BROYDEN_K),
    # zeros(nz * MAX_BROYDEN_K),
    # zeros(nv * MAX_BROYDEN_K),
    # zeros(nz * MAX_BROYDEN_K),
    # zeros(nv * MAX_BROYDEN_K),
    ones(nz),
    ones(nv),
    # ones(nz),
    # ones(nv),
    # Anderson
    # zeros(nz + nv, ANDERSON_BUFFER_SIZE),
    # zeros(nz + nv, ANDERSON_BUFFER_SIZE),
    # zeros(nz),
    # zeros(nv),
    # zeros(nz),
    # zeros(nv),
    # zeros(nz + nv),
    # zeros(nz + nv, ANDERSON_BUFFER_SIZE),
    # LA.UpperTriangular(LA.Matrix(LA.I(ANDERSON_BUFFER_SIZE) * 1.)),
    # zeros(ANDERSON_BUFFER_SIZE),
  )

  return SPOCK(
    solver_state,
    solver_state_internal,
    aa_state,
    problem
  )

end

function generate_qnewton_direction!(
  model :: SPOCK,
  k :: TI,
  alpha1 :: TF,
  alpha2 :: TF
) where {TI <: Integer, TF <: Real}

  # println("TODO: Function not implemented for this model.")
  # return restarted_broyden!(model, k, alpha1, alpha2)
  k += 1
  anderson!(model, k)
  return k 
end


################
### Model solver
################

function solve_model!(
  model :: SPOCK,
  x0 :: AbstractArray{TF, 1};
  tol :: TF = 1e-3,
  verbose :: VERBOSE_LEVEL = SILENT,
  z0 :: Union{Vector{Float64}, Nothing} = nothing, 
  v0 :: Union{Vector{Float64}, Nothing} = nothing,
  gamma :: Union{Float64, Nothing} = nothing,
  sigma :: Union{Float64, Nothing} = nothing
) where {TF <: Real}

  if z0 !== nothing || v0 !== nothing
    copyto!(model.state.z, z0)
    copyto!(model.state.v, v0)
  end

  copyto!(model.problem.x0, x0)
  copyto!(model.state.res_0, [-Inf, -Inf])

  run_sp!(model, tol=tol, verbose=verbose, sigma = sigma, gamma = gamma)
end