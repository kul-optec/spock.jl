############################################################
# Build step
############################################################

function get_nz(problem_definition :: GENERIC_PROBLEM_DEFINITION)
  nx_total = problem_definition.scen_tree.n * problem_definition.nx   # Every node has a state

  return (problem_definition.scen_tree.n_non_leaf_nodes * problem_definition.nu # One input per non leaf node
              + nx_total                                        # Number of state variables over the tree
              + problem_definition.scen_tree.n                                     # s variable: 1 component per node
              + problem_definition.scen_tree.n_non_leaf_nodes * length(problem_definition.rms[1].b)  # One y variable for each non leaf node
              + problem_definition.scen_tree.n - 1)                                # One tau variable per non root node

  # todo: to support non uniform risk measures, replace the `length(rms[1].b)` by a summation of the lengths of all the risk measures' b vector.
end

function get_nv(problem_definition :: GENERIC_PROBLEM_DEFINITION)

  # todo: to support non uniform risk measures, replace the `length(rms[1].b)` by a summation of the lengths of all the risk measures' b vector.
  ny = length(problem_definition.rms[1].b)

  # for every non-leaf node
  nv_1 = problem_definition.scen_tree.n_non_leaf_nodes * ny
  nv_2 = problem_definition.scen_tree.n_non_leaf_nodes

  # for every child node of non-leaf nodes (stage costs)
  nv_3 = problem_definition.nx * (problem_definition.scen_tree.n - 1)
  nv_4 = problem_definition.nu * (problem_definition.scen_tree.n - 1)
  nv_5 = problem_definition.scen_tree.n - 1
  nv_6 = problem_definition.scen_tree.n - 1

  nv_7 = problem_definition.constraints.nΓ_nonleaf

  # TODO: Skipped 8, 9, 10 -> rename these

  # for every leaf node (terminal costs)
  nv_11 = problem_definition.nx * problem_definition.scen_tree.n_leaf_nodes
  nv_12 = problem_definition.scen_tree.n_leaf_nodes
  nv_13 = problem_definition.scen_tree.n_leaf_nodes

  nv_14 = problem_definition.constraints.nΓ_leaf

  return nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_7, nv_11, nv_12, nv_13, nv_14
end

function ricatti_offline(
  problem_definition :: GENERIC_PROBLEM_DEFINITION
)

  P = [
    zeros(problem_definition.nx, problem_definition.nx)
    for _ = 1:problem_definition.scen_tree.n
  ]
  K = [
    zeros(problem_definition.nu, problem_definition.nx)
    for _ = 1:problem_definition.scen_tree.n_non_leaf_nodes
  ]
  R_chol = [
    LA.cholesky(Matrix(2. * LA.I(problem_definition.nu))) for _ = 1:problem_definition.scen_tree.n_non_leaf_nodes

  ]
  ABK = [
    zeros(problem_definition.nx, problem_definition.nx)
    for _ = 1:problem_definition.scen_tree.n
  ]

  for i = problem_definition.scen_tree.leaf_node_min_index:problem_definition.scen_tree.leaf_node_max_index
    P[i] = LA.I(problem_definition.nx) * 1.
  end

  for i = problem_definition.scen_tree.n_non_leaf_nodes: -1 : 1
    children_of_i = problem_definition.scen_tree.child_mapping[i]
    sum_for_r = zeros(problem_definition.nu, problem_definition.nu)
    sum_for_k = zeros(problem_definition.nu, problem_definition.nx)

    for j in children_of_i
      sum_for_r += problem_definition.dynamics.B[problem_definition.scen_tree.node_info[j].w]' * P[j] * problem_definition.dynamics.B[problem_definition.scen_tree.node_info[j].w]
      sum_for_k += problem_definition.dynamics.B[problem_definition.scen_tree.node_info[j].w]' * P[j] * problem_definition.dynamics.A[problem_definition.scen_tree.node_info[j].w]
    end

    # Symmetrize sum_for_r to avoid numerical issues:
    sum_for_r = 0.5 * (sum_for_r + sum_for_r')

    rtilde = LA.I(problem_definition.nu) + sum_for_r
    chol = LA.cholesky(rtilde); R_chol[i] = chol
    K[i] = chol \ (-sum_for_k)

    sum_for_p = zeros(problem_definition.nx, problem_definition.nx)
    for j in children_of_i
      ABK[j] = problem_definition.dynamics.A[problem_definition.scen_tree.node_info[j].w] + problem_definition.dynamics.B[problem_definition.scen_tree.node_info[j].w] * K[i]
      sum_for_p += ABK[j]' * P[j] * ABK[j]
    end
    P[i] = LA.I(problem_definition.nx) + K[i]' * K[i] + sum_for_p
  end

  return P, K, R_chol, ABK
end

####################
### Internal utility functions
####################

"""
Get the indices of the x variable's components.
"""
function z_to_x(problem_definition :: GENERIC_PROBLEM_DEFINITION)
    return collect(
        1 : problem_definition.nx * problem_definition.scen_tree.n
    )
end    

function z_to_u(problem_definition :: GENERIC_PROBLEM_DEFINITION)
    return collect(
        problem_definition.nx * problem_definition.scen_tree.n + 1 : 
        problem_definition.nx * problem_definition.scen_tree.n + 
          (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu
    )
end

function z_to_s(problem_definition :: GENERIC_PROBLEM_DEFINITION)
    return collect(
      problem_definition.nx * problem_definition.scen_tree.n + 
        (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 1 :
      problem_definition.nx * problem_definition.scen_tree.n + 
        (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 
        problem_definition.scen_tree.n
    )
end

function z_to_tau(problem_definition :: GENERIC_PROBLEM_DEFINITION)

  return collect(
    problem_definition.nx * problem_definition.scen_tree.n + 
      (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 
      problem_definition.scen_tree.n + 1 :
      problem_definition.nx * problem_definition.scen_tree.n + 
      (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 
      problem_definition.scen_tree.n + problem_definition.scen_tree.n - 1
  )
end

function z_to_y(problem_definition :: GENERIC_PROBLEM_DEFINITION)
  """
  TODO: replace n_y by a summation over all ny to support non uniform risk measures
  """

  ny = length(problem_definition.rms[1].b)

    return collect(
      problem_definition.nx * problem_definition.scen_tree.n + 
        (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 
        problem_definition.scen_tree.n + (problem_definition.scen_tree.n - 1) + 1 :
        problem_definition.nx * problem_definition.scen_tree.n + 
          (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 
          problem_definition.scen_tree.n + (problem_definition.scen_tree.n - 1) + 
          problem_definition.scen_tree.n_non_leaf_nodes * ny
    )
end

function node_to_x(problem_definition :: GENERIC_PROBLEM_DEFINITION, i :: Int64)
  return collect(
      (i - 1) * problem_definition.nx + 1 : i * problem_definition.nx
  )
end

function node_to_u(problem_definition :: GENERIC_PROBLEM_DEFINITION, i :: Int64)
  return collect(
      (i - 1) * problem_definition.nu + 1 : i * problem_definition.nu
  )
end


########################################
### Solve step
########################################

function L!(model :: MODEL_IMPLICITL, z :: AbstractArray{TF, 1}, v :: AbstractArray{TF, 1}) where {TF <: Real}
  """
    Updates v such that 
    v <- L z
    without constructing L explicitly.
  """

  if length(z) !== model.solver_state.nz || length(v) !== model.solver_state.nv
    error("z or v has the wrong length")
  end

  ny = length(model.problem_definition.rms[1].b)
  n_non_leafs = model.problem_definition.scen_tree.n_non_leaf_nodes
  n_leafs = model.problem_definition.scen_tree.n_leaf_nodes
  nx = model.problem_definition.nx
  nu = model.problem_definition.nu
  n = model.problem_definition.scen_tree.n
  leaf_offset = model.problem_definition.scen_tree.leaf_node_min_index - 1

  v2_offset = model.solver_state_internal.v2_offset
  v3_offset = model.solver_state_internal.v3_offset
  v4_offset = model.solver_state_internal.v4_offset
  v5_offset = model.solver_state_internal.v5_offset
  v6_offset = model.solver_state_internal.v6_offset
  v7_offset = model.solver_state_internal.v7_offset
  v11_offset = model.solver_state_internal.v11_offset
  v12_offset = model.solver_state_internal.v12_offset
  v13_offset = model.solver_state_internal.v13_offset
  v14_offset = model.solver_state_internal.v14_offset

  # v1
  for k = 1:length(model.solver_state_internal.y_inds)
    v[k] = z[model.solver_state_internal.y_inds[k]]
  end

  # v2
  # v2_offset = length(model.solver_state_internal.y_inds)
  for i = 1:n_non_leafs
    # v2 = s - b' * y
    # v[v2_offset + i] = z[model.solver_state_internal.s_inds[i]] - model.problem_definition.rms[i].b' * z[model.solver_state_internal.y_inds[(i - 1) * ny + 1 : i * ny]]
    dot_p = 0.
    for k = 1:ny
      dot_p += model.problem_definition.rms[i].b[k] * z[model.solver_state_internal.y_inds[(i - 1) * ny + k]]
    end
    v[v2_offset + i] = z[model.solver_state_internal.s_inds[i]] - dot_p
  end

  # v3
  # v3_offset = v2_offset + n_non_leafs
  for i = 1:n_non_leafs
    for j in model.problem_definition.scen_tree.child_mapping[i]
      j_ind = j - 1
      # v3 = sqrt(Q) * x
      LA.mul!(model.solver_state_internal.mul_x_workspace, model.solver_state_internal.sqrtQ[j_ind], view(z, model.solver_state_internal.x_inds[(i - 1) * nx + 1] : model.solver_state_internal.x_inds[i * nx]))
      for k = 1:nx
        v[v3_offset + (j_ind - 1) * nx + k] = model.solver_state_internal.mul_x_workspace[k]
      end
    end
  end

  # v4
  # v4_offset = v3_offset + nx * (n - 1)
  for i = 1:n_non_leafs
    for j in model.problem_definition.scen_tree.child_mapping[i]
      j_ind = j - 1
      # v4 = sqrt(R) * u
      LA.mul!(model.solver_state_internal.mul_u_workspace, model.solver_state_internal.sqrtR[j_ind], view(z, model.solver_state_internal.u_inds[(i - 1) * nu + 1] : model.solver_state_internal.u_inds[i * nu]))
      for k = 1:nu
        v[v4_offset + (j_ind - 1) * nu + k] = model.solver_state_internal.mul_u_workspace[k]
      end
    end
  end

  # v5
  # v5_offset = v4_offset + nu * (n - 1)
  node_counter = 0
  for i = 1:n_non_leafs
    for j in model.problem_definition.scen_tree.child_mapping[i]
      j_ind = j - 1
      # v5 = 0.5 tau
      v[v5_offset + node_counter + 1] = 0.5 * z[model.solver_state_internal.tau_inds[j_ind]]
      node_counter += 1
    end
  end

  # v6
  # v6_offset = v5_offset + (n - 1)
  node_counter = 0
  for i = 1:n_non_leafs
    for j in model.problem_definition.scen_tree.child_mapping[i]
      j_ind = j - 1
      # v6 = 0.5 tau
      v[v6_offset + node_counter + 1] = 0.5 * z[model.solver_state_internal.tau_inds[j_ind]]
      node_counter += 1
    end
  end

  # v7
  if ! isnothing(v7_offset)
    update_nonleaf_constraints_dual!(
      model.problem_definition.constraints, 
      view(v, v7_offset + 1 : v7_offset + model.problem_definition.constraints.nΓ_nonleaf),
      view(z, 1 : n_non_leafs * nx),
      view(z, model.solver_state_internal.u_inds)
    )
  end

  # v8-v10 TODO: Rename

  # v11
  # v11_offset = v6_offset + (n - 1)
  for i = 1:n_leafs
    j = i + leaf_offset
    LA.mul!(model.solver_state_internal.mul_x_workspace, model.solver_state_internal.sqrtQN[i], view(z, model.solver_state_internal.x_inds[(j - 1) * nx + 1] : model.solver_state_internal.x_inds[j * nx]))
    for k = 1:nx
      v[v11_offset + (i - 1) * nx + k] = model.solver_state_internal.mul_x_workspace[k]
    end
  end

  # v12
  # v12_offset = v11_offset + n_leafs * nx
  for i = 1:n_leafs
    j = i + leaf_offset
    v[v12_offset + i] = 0.5 * z[model.solver_state_internal.s_inds[j]]
  end

  # v13
  # v13_offset = v12_offset + n_leafs
  for i = 1:n_leafs
    j = i + leaf_offset
    v[v13_offset + i] = 0.5 * z[model.solver_state_internal.s_inds[j]]
  end

  # v14
  if ! isnothing(v14_offset)
    update_leaf_constraints_dual!(
      model.problem_definition.constraints, 
      view(v, v14_offset + 1 : v14_offset + model.problem_definition.constraints.nΓ_leaf),
      view(z, n_non_leafs * nx + 1 : n * nx)
    )
  end
end

function L_transpose!(model :: MODEL_IMPLICITL, z :: AbstractArray{TF, 1}, v :: AbstractArray{TF, 1}) where {TF <: Real}
  """
    Updates z such that 
    z <- L' v
    without constructing L explicitly.
  """

  if length(z) !== model.solver_state.nz || length(v) !== model.solver_state.nv
    error("z or v has the wrong length")
  end

  nx = model.problem_definition.nx
  nu = model.problem_definition.nu
  ny = length(model.problem_definition.rms[1].b)
  n_non_leafs = model.problem_definition.scen_tree.n_non_leaf_nodes
  n_leafs = model.problem_definition.scen_tree.n_leaf_nodes
  leaf_offset = model.problem_definition.scen_tree.leaf_node_min_index - 1
  n = model.problem_definition.scen_tree.n

  v2_offset = model.solver_state_internal.v2_offset
  v3_offset = model.solver_state_internal.v3_offset
  v4_offset = model.solver_state_internal.v4_offset
  v5_offset = model.solver_state_internal.v5_offset
  v6_offset = model.solver_state_internal.v6_offset
  v7_offset = model.solver_state_internal.v7_offset
  v11_offset = model.solver_state_internal.v11_offset
  v12_offset = model.solver_state_internal.v12_offset
  v13_offset = model.solver_state_internal.v13_offset
  v14_offset = model.solver_state_internal.v14_offset

  # x for non leaf nodes
  if ! isnothing(v7_offset) # Set x = Γ' v7
    update_nonleaf_constraints_primal_x!(
      model.problem_definition.constraints,
      view(z, model.solver_state_internal.x_inds[1] : model.solver_state_internal.x_inds[1] + nx * n_non_leafs),
      view(v, v7_offset + 1 : v7_offset + model.problem_definition.constraints.nΓ_nonleaf)
    )
  else # Set x = 0
    for i = 1:n_non_leafs
      for j = (i - 1) * nx + 1 : i * nx
        z[model.solver_state_internal.x_inds[j]] = 0.
      end
    end
  end
  # Add all sqrt(Q) * v3 terms
  for i = 1:n_non_leafs
    for j in model.problem_definition.scen_tree.child_mapping[i]
      j_ind = j - 1
      LA.mul!(model.solver_state_internal.mul_x_workspace, model.solver_state_internal.sqrtQ[j_ind], view(v, v3_offset + (j_ind - 1) * nx + 1 : v3_offset + j_ind * nx))
      for k = 1:nx
        z[model.solver_state_internal.x_inds[(i - 1) * nx + k]] += model.solver_state_internal.mul_x_workspace[k]
      end
    end
  end

  # x for leaf nodes
  if ! isnothing(v14_offset)
    update_leaf_constraints_primal!(
      model.problem_definition.constraints,
      view(z, nx * n_non_leafs + 1 : model.solver_state_internal.x_inds[end]),
      view(v, v14_offset + 1 : v14_offset + model.problem_definition.constraints.nΓ_leaf)
    )
  else
    for i = 1:n_leafs
      for j = (i + leaf_offset - 1) * nx + 1 : (i + leaf_offset) * nx
        z[model.solver_state_internal.x_inds[j]] = 0.
      end
    end
  end
  # Add sqrt(Q) * v11 term
  for i = 1:n_leafs
    j = leaf_offset + i
    LA.mul!(model.solver_state_internal.mul_x_workspace, model.solver_state_internal.sqrtQN[i], view(v, v11_offset + (i-1) * nx + 1 : v11_offset + i * nx))
    for k = 1:nx
      z[model.solver_state_internal.x_inds[(j-1) * nx + k]] += model.solver_state_internal.mul_x_workspace[k]
    end
  end

  # u (u is only defined for non leaf nodes)
  if ! isnothing(v7_offset) # Set u = Γ' * v7
    update_nonleaf_constraints_primal_u!(
      model.problem_definition.constraints,
      view(z, model.solver_state_internal.u_inds),
      view(v, v7_offset + 1 : v7_offset + model.problem_definition.constraints.nΓ_nonleaf)
    )
  else # Set u = 0
    for i = 1:n_non_leafs
      for j = (i - 1) * nu + 1 : i * nu
        z[model.solver_state_internal.u_inds[j]] = 0.
      end
    end
  end # Add all sqrt(R) * v4 terms
  for i = 1:n_non_leafs
    for j in model.problem_definition.scen_tree.child_mapping[i]
      j_ind = j - 1
      LA.mul!(model.solver_state_internal.mul_u_workspace, model.solver_state_internal.sqrtR[j_ind], view(v, v4_offset + (j_ind - 1) * nu + 1 : v4_offset + j_ind * nu))
      for k = 1:nu
        z[model.solver_state_internal.u_inds[(i - 1) * nu + k]] += model.solver_state_internal.mul_u_workspace[k]
      end
    end
  end

  # y
  for i = 1:n_non_leafs
    # z[model.solver_state_internal.y_inds[(i - 1) * ny + 1 : i * ny]] = v[(i - 1) * ny + 1 : i * ny] - 
    #   model.problem_definition.rms[i].b * v[v2_offset + i]
    for j = 1:ny
      z[model.solver_state_internal.y_inds[(i - 1) * ny + j]] = v[(i - 1) * ny + j] - model.problem_definition.rms[i].b[j] * v[v2_offset + i]
    end
  end

  # tau
  for i = 1:n_non_leafs
    for j in model.problem_definition.scen_tree.child_mapping[i]
      j_ind = j - 1
      z[model.solver_state_internal.tau_inds[j_ind]] = 0.5 * v[v5_offset + j_ind] + 0.5 * v[v6_offset + j_ind]
    end
  end

  # s for non leaf nodes
  for i = 1:n_non_leafs
    z[model.solver_state_internal.s_inds[i]] = v[v2_offset + i]
  end

  # s for leaf nodes
  for i = 1:n_leafs
    j = i + leaf_offset
    z[model.solver_state_internal.s_inds[j]] = 0.5 * v[v12_offset + i] + 0.5 * v[v13_offset + i]
  end
end

function spock_mul!(
  model :: MODEL_IMPLICITL,
  C :: AbstractArray{TF, 1},
  trans :: Bool,
  B :: AbstractArray{TF, 1},
  α :: TF,
  β :: TF
) where {TF <: Real}
"""
In place matrix-vector multiply-add

C <- model.L * B * α + C * β when trans is false, otherwise
C <- model.L' * B * α + C * β
"""

  # C <- L * B * α + C * β
  if !trans
    copyto!(model.solver_state_internal.spock_mul_buffer_v, C)
    L!(model, B, C)
    @simd for i = 1:model.solver_state.nv
      @inbounds C[i] = α * C[i] + β * model.solver_state_internal.spock_mul_buffer_v[i]
    end
  # C <- L' * B * α + C * β
  else
    copyto!(model.solver_state_internal.spock_mul_buffer_z, C)
    L_transpose!(model, C, B)
    @simd for i = 1:model.solver_state.nz
      @inbounds C[i] = α * C[i] + β * model.solver_state_internal.spock_mul_buffer_z[i]
    end
  end
end

precompile(spock_mul!, (MODEL_CP_IMPLICITL, Vector{Float64}, Bool, Vector{Float64}, Float64, Float64))

function spock_dot(
  model :: MODEL_IMPLICITL,
  arg1_z :: AbstractArray{TF, 1}, 
  arg1_v :: AbstractArray{TF, 1}, 
  arg2_z :: AbstractArray{TF, 1}, 
  arg2_v :: AbstractArray{TF, 1}, 
  gamma :: TF, 
  sigma :: TF
) where {TF <: Real}

  L!(model, arg2_z, model.solver_state_internal.workspace_Lv)
  L_transpose!(model, model.solver_state_internal.workspace_Lz, arg2_v)

  for k = 1:model.solver_state.nz
    model.solver_state_internal.workspace_Lz[k] = arg2_z[k] - gamma * model.solver_state_internal.workspace_Lz[k]
  end
  for k = 1:model.solver_state.nv
    model.solver_state_internal.workspace_Lv[k] = - sigma * model.solver_state_internal.workspace_Lv[k] + arg2_v[k]
  end

  res = 0.
  for k = 1:model.solver_state.nz
    res += arg1_z[k] * model.solver_state_internal.workspace_Lz[k]
  end
  for k = 1:model.solver_state.nv
    res += arg1_v[k] * model.solver_state_internal.workspace_Lv[k]
  end
  
  return res

  # return arg1_z' * model.solver_state_internal.workspace_Lz + arg1_v' * model.solver_state_internal.workspace_Lv
end

function spock_dot(
  model :: MODEL_IMPLICITL,
  arg1 :: AbstractArray{TF, 1}, 
  arg2 :: AbstractArray{TF, 1}, 
  gamma :: TF, 
  sigma :: TF
) where {TF <: Real}

  # TODO: Do not index and allocate
  return spock_dot(
    model,
    arg1[1:model.solver_state.nz],
    arg1[model.solver_state.nz + 1 : model.solver_state.nz + model.solver_state.nv],
    arg2[1:model.solver_state.nz],
    arg2[model.solver_state.nz + 1 : model.solver_state.nz + model.solver_state.nv],
    gamma,
    sigma
  )
end

function spock_norm(
  model :: MODEL_IMPLICITL,
  arg :: AbstractArray{TF, 1}, 
  gamma :: TF, 
  sigma :: TF
) where {TF <: Real}

  return sqrt(spock_dot(model, arg, arg, gamma, sigma))
end

function spock_norm(
  model :: MODEL_IMPLICITL,
  arg_z :: AbstractArray{TF, 1}, 
  arg_v :: AbstractArray{TF, 1}, 
  gamma :: TF, 
  sigma :: TF
) where {TF <: Real}

  return sqrt(spock_dot(model, arg_z, arg_v, arg_z, arg_v, gamma, sigma))
end

function projection_S1!(
  model :: MODEL_IMPLICITL,
  z1 :: AbstractArray{TF, 1},
  gamma :: TF
) where {TF <: Real}
  """
  Project z1 := (x, u) onto the set of dynamics S1
  """
  nx = model.problem_definition.nx
  nu = model.problem_definition.nu
  
  u_offset = model.problem_definition.scen_tree.n * model.problem_definition.nx
  q = model.solver_state_internal.ric_q
  d = model.solver_state_internal.ric_d
  ABK = model.solver_state_internal.ABK
  P = model.solver_state_internal.P
  K = model.solver_state_internal.K
  sum_for_d = model.solver_state_internal.sum_for_d

  for i = model.problem_definition.scen_tree.leaf_node_min_index : model.problem_definition.scen_tree.leaf_node_max_index
    for k = 1:nx
      q[(i - 1) * nx + k] = - z1[(i - 1) * nx + k]
    end
  end

  for i = model.problem_definition.scen_tree.n_non_leaf_nodes: -1 : 1
    children_of_i = model.problem_definition.scen_tree.child_mapping[i]
    for k = 1:nu
      sum_for_d[k] = 0.
    end

    for j in children_of_i
      LA.mul!(model.solver_state_internal.mul_u_workspace, model.problem_definition.dynamics.B[model.problem_definition.scen_tree.node_info[j].w]', view(q, (j - 1) * nx + 1 : j * nx))
      for k = 1:nu
        sum_for_d[k] += model.solver_state_internal.mul_u_workspace[k]
      end
      # sum_for_d += model.problem_definition.dynamics.B[model.problem_definition.scen_tree.node_info[j].w]' * q[(j - 1) * nx + 1 : j * nx]
    end
    
    for k = 1:nu
      model.solver_state_internal.mul_u_workspace[k] = z1[u_offset + (i - 1) * nu + k] - sum_for_d[k]
    end
    LA.ldiv!(model.solver_state_internal.R_chol[i], model.solver_state_internal.mul_u_workspace)
    for k = 1:nu
      d[(i - 1) * nu + k] = model.solver_state_internal.mul_u_workspace[k]
    end

    # Set q_i to zero
    for j = (i - 1) * nx + 1 : i * nx
      q[j] = 0.
    end
    for j in children_of_i
      # q[(i - 1) * nx + 1 : i * nx] += ABK[j]' * (
      #   P[j] * model.problem_definition.dynamics.B[model.problem_definition.scen_tree.node_info[j].w] * d[(i - 1) * nu + 1 : i * nu] + q[(j-1) * nx + 1 : j * nx]
      # )
      LA.mul!(model.solver_state_internal.mul_x_workspace, model.problem_definition.dynamics.B[model.problem_definition.scen_tree.node_info[j].w], view(d, (i - 1) * nu + 1 : i * nu))
      LA.mul!(model.solver_state_internal.mul_x_workspace2, P[j], model.solver_state_internal.mul_x_workspace)
      for k = 1:nx
        model.solver_state_internal.mul_x_workspace2[k] += q[(j-1) * nx + k]  
      end
      LA.mul!(model.solver_state_internal.mul_x_workspace, ABK[j]', model.solver_state_internal.mul_x_workspace2)
      for k = 1:nx
        q[(i - 1) * nx + k] += model.solver_state_internal.mul_x_workspace[k]
      end

    end
    # q[(i - 1) * nx + 1 : i * nx] += K[i]' * ( d[(i - 1) * nu + 1 : i * nu] - z1[u_offset + (i - 1) * nu + 1 : u_offset + i * nu])
    for k = 1:nu
      model.solver_state_internal.mul_u_workspace[k] = d[(i - 1) * nu + k] - z1[u_offset + (i - 1) * nu + k]
    end
    LA.mul!(model.solver_state_internal.mul_x_workspace, K[i]', model.solver_state_internal.mul_u_workspace)
    for k = 1:nx
      q[(i - 1) * nx + k] += model.solver_state_internal.mul_x_workspace[k]
    end

    for j = (i - 1) * nx + 1 : i * nx
      q[j] -= z1[j]
    end
  end

  ####################################################################

  # gurobi = Model(Gurobi.Optimizer)
  # set_silent(gurobi)

  # xbar = z1[1:14]; ubar = z1[15:17]

  # @variable(gurobi, x[1:14])
  # @variable(gurobi, u[1:3])
  # @objective(gurobi, Min, (x - xbar)' * (x - xbar) + (u - ubar)' * (u - ubar))
  # @constraint(gurobi, x[3:4] .== dynamics.A[1] * x[1:2] + dynamics.B[1] * u[1])
  # @constraint(gurobi, x[5:6] .== dynamics.A[2] * x[1:2] + dynamics.B[2] * u[1])
  # @constraint(gurobi, x[7:8] .== dynamics.A[1] * x[3:4] + dynamics.B[1] * u[2])
  # @constraint(gurobi, x[9:10] .== dynamics.A[2] * x[3:4] + dynamics.B[2] * u[2])
  # @constraint(gurobi, x[11:12] .== dynamics.A[1] * x[5:6] + dynamics.B[1] * u[3])
  # @constraint(gurobi, x[13:14] .== dynamics.A[2] * x[5:6] + dynamics.B[2] * u[3])
  # @constraint(gurobi, x[1:2] .== model.problem_definition.x0)

  # optimize!(gurobi)
  # xref = value.(gurobi[:x]) 
  # uref = value.(gurobi[:u])

  # z1[1:14] = xref
  # z1[15:17] = uref
  # # return
  
  ####################################################################

  z1[1:nx] = model.problem_definition.x0

  for i = 1:model.problem_definition.scen_tree.n_non_leaf_nodes
    LA.mul!(view(z1, u_offset + (i - 1) * nu + 1 : u_offset + i * nu), K[i], view(z1, (i - 1) * nx + 1 : i * nx))
    for j = (i - 1) * nu + 1 : i * nu
      z1[u_offset + j] += d[j]
    end
    for j in model.problem_definition.scen_tree.child_mapping[i]
      LA.mul!(view(z1, (j - 1) * nx + 1 : j * nx), ABK[j], view(z1, (i - 1) * nx + 1 : i * nx))
      LA.mul!(model.solver_state_internal.mul_x_workspace, model.problem_definition.dynamics.B[model.problem_definition.scen_tree.node_info[j].w], view(d, (i - 1) * nu + 1 : i * nu))
      for k = 1:nx
        z1[(j - 1) * nx + k] += model.solver_state_internal.mul_x_workspace[k]
      end
    end
  end

  # println("----------------")
  # println("$(xref), $(uref)")
  # println("$(z1[1:14]), $(z1[15:17])")
end

function projection_S2!(
  model :: MODEL_IMPLICITL,
  z2 :: AbstractArray{TF, 1},
  gamma :: TF
) where {TF <: Real}

"""
  z2 = (s_2, ... s_n, tau_2, ..., tau_n, y_1, ..., y_{n_non_leafs_max})
"""

  # TODO: It should be possible to reformulate this using only dot products, no?

  n = model.problem_definition.scen_tree.n
  tau_offset = (n - 1)
  y_offset = (n-1 + n - 1)
  ny = length(model.problem_definition.rms[1].b)


  # Project onto the kernel, a.k.a. LS with kernel matrix
  # Note that ls_matrix is the pseudo inverse
  for i = 1:model.problem_definition.scen_tree.n_non_leaf_nodes
    children_of_i = model.problem_definition.scen_tree.child_mapping[i]
    n_children = length(children_of_i)

    copyto!(model.solver_state_internal.ls_b, 1, z2, y_offset + (i - 1) * ny + 1, ny)
    copyto!(model.solver_state_internal.ls_b, ny+1, z2, children_of_i[1] - 1, n_children)
    copyto!(model.solver_state_internal.ls_b, ny + n_children + 1, z2, tau_offset + children_of_i[1] - 1, n_children)
    
    # temp = model.solver_state_internal.ls_matrix[i] * (temp)
    LA.mul!(model.solver_state_internal.ls_b2, model.solver_state_internal.ls_matrix[i], model.solver_state_internal.ls_b)
    # LA.mul!(model.solver_state_internal.ls_b2, model.solver_state_internal.ls_matrix[i], temp)

    # z2[y_offset + (i - 1) * ny + 1 : y_offset + i * ny] = temp[1:ny]
    copyto!(z2, y_offset + (i - 1) * ny + 1, model.solver_state_internal.ls_b2, 1, ny)
    # z2[children_of_i .- 1] = temp[ny+1 : ny + n_children]
    copyto!(z2, children_of_i[1] - 1, model.solver_state_internal.ls_b2, ny+1, n_children)
    # z2[tau_offset .+ children_of_i .- 1] = temp[ny + n_children + 1 : ny + 2 * n_children]
    copyto!(z2, tau_offset + children_of_i[1] - 1, model.solver_state_internal.ls_b2, ny + n_children + 1, n_children)
  end
end

function prox_f!(
  model :: MODEL_IMPLICITL,
  arg :: AbstractArray{TF, 1},
  gamma :: TF
) where {TF <: Real}
"""
This function accepts an argument arg and parameter gamma.
With these, it computes the prox_f^gamma(arg) and stores the result in arg.

arg <- prox_f^gamma(arg)
"""
  # s_0 -= gamma
  arg[model.solver_state_internal.s_inds[1]] -= gamma
  
  # Projection onto S1
  projection_S1!(model, view(arg, 1 : model.solver_state_internal.s_inds[1] - 1), gamma)

  # Projection onto S2
  projection_S2!(model, view(arg, model.solver_state_internal.s_inds[1] + 1 : model.solver_state.nz), gamma)
end

precompile(prox_f!, (MODEL_CP_IMPLICITL, Vector{Float64}, Float64))

function project_on_leaf_constraints!(
  model :: MODEL_IMPLICITL,
  arg :: AbstractArray{TF, 1}
) where {TF <: Real}

  n_leafs = model.problem_definition.scen_tree.n_leaf_nodes
  nx = model.problem_definition.nx
  v11_offset = model.solver_state_internal.v11_offset
  v12_offset = model.solver_state_internal.v12_offset
  v13_offset = model.solver_state_internal.v13_offset
  v14_offset = model.solver_state_internal.v14_offset

  ####
  # !! Mathoptinterface defines the SOC by the vector (t, x), not (x, t)
  ####

  for i = 1:n_leafs
    # v13
    model.solver_state_internal.proj_leafs_workspace[1] = arg[v13_offset + i]
    # v11
    for j = 1:nx
      model.solver_state_internal.proj_leafs_workspace[j + 1] = arg[v11_offset + (i - 1) * nx + j]
    end
    # v12
    model.solver_state_internal.proj_leafs_workspace[nx + 2] = arg[v12_offset + i]

    # model.solver_state_internal.proj_leafs_workspace[1:nx+2] = MOD.projection_on_set(
    #   MOD.DefaultDistance(), 
    #   model.solver_state_internal.proj_leafs_workspace, 
    #   MOI.SecondOrderCone(nx + 2)
    # )
    project_onto_cone!(view(model.solver_state_internal.proj_leafs_workspace, 1 : nx + 2), MOI.SecondOrderCone(nx + 2))

    ## TODO: SOC does not always return t >= |x|... 
    # v13
    arg[v13_offset + i] = model.solver_state_internal.proj_leafs_workspace[1]
    # t_norm = LA.norm(model.solver_state_internal.proj_leafs_workspace[2:nx+2])
    # if t_norm > model.solver_state_internal.proj_leafs_workspace[1]
    #   arg[v13_offset + i] = t_norm
    # else
    #   arg[v13_offset + i] = model.solver_state_internal.proj_leafs_workspace[1]
    # end
    # @assert arg[v13_offset + i] >= t_norm
    # v11
    for j = 1:nx
      arg[v11_offset + (i - 1) * nx + j] = model.solver_state_internal.proj_leafs_workspace[j + 1]
    end
    # v12
    arg[v12_offset + i] = model.solver_state_internal.proj_leafs_workspace[nx + 2]
  end

  ## V14
  if ! isnothing(v14_offset)
    project_onto_leaf_constraints!(
      model.problem_definition.constraints, 
      view(arg, v14_offset + 1 : v14_offset + model.problem_definition.constraints.nΓ_leaf)
    )
  end
end

function project_on_nonleaf_constraints!(
  model :: MODEL_IMPLICITL,
  arg :: AbstractArray{TF, 1}
) where {TF <: Real}

  ####
  # !! Mathoptinterface defines the SOC by the vector (t, x), not (x, t)
  ####

  nx = model.problem_definition.nx
  nu = model.problem_definition.nu

  v2_offset = model.solver_state_internal.v2_offset
  v3_offset = model.solver_state_internal.v3_offset
  v4_offset = model.solver_state_internal.v4_offset
  v5_offset = model.solver_state_internal.v5_offset
  v6_offset = model.solver_state_internal.v6_offset
  v7_offset = model.solver_state_internal.v7_offset
  n_non_leafs = model.problem_definition.scen_tree.n_non_leaf_nodes

  ### v1
  for i = 1:model.problem_definition.scen_tree.n_non_leaf_nodes
    ny = length(model.problem_definition.rms[i].b)

    offset = (i - 1) * ny
    for k = 1:length(model.problem_definition.rms[i].K.subcones)
      cone = model.problem_definition.rms[i].K.subcones[k]
      dim = MOI.dimension(cone)
      # arg[offset + 1 : offset + dim] = MOD.projection_on_set(MOD.DefaultDistance(), arg[offset + 1 : offset + dim], MOI.dual_set(cone))
      project_onto_cone!(view(arg, offset + 1 : offset + dim), MOI.dual_set(cone))
      offset += dim
    end

  end

  ### v2
  # arg[v2_offset + 1 : v2_offset + n_non_leafs] = MOD.projection_on_set(MOD.DefaultDistance(), arg[v2_offset + 1 : v2_offset + n_non_leafs], MOI.Nonnegatives(n_non_leafs))
  for k = 1:n_non_leafs
    arg[v2_offset + k] = arg[v2_offset + k] >= 0 ? arg[v2_offset + k] : 0.
  end

  ### v3 - v6
  for i = 1:n_non_leafs
    children_of_i = model.problem_definition.scen_tree.child_mapping[i]
    for j in children_of_i
      j_ind = j - 1

      # v6
      model.solver_state_internal.proj_nleafs_workspace[1] = arg[v6_offset + j_ind]
      # v3
      for k = 1:nx
        model.solver_state_internal.proj_nleafs_workspace[k + 1] = arg[v3_offset + (j_ind - 1) * nx + k]
      end
      # v4
      for k = 1:nu
        model.solver_state_internal.proj_nleafs_workspace[1 + nx + k] = arg[v4_offset + (j_ind - 1) * nu + k]
      end
      # v5
      model.solver_state_internal.proj_nleafs_workspace[1 + nx + nu + 1] = arg[v5_offset + j_ind]

      # model.solver_state_internal.proj_nleafs_workspace[1:nx+nu+2] = MOD.projection_on_set(
      #   MOD.DefaultDistance(), 
      #   model.solver_state_internal.proj_nleafs_workspace, 
      #   MOI.SecondOrderCone(nx + nu + 2)
      # )
      project_onto_cone!(view(model.solver_state_internal.proj_nleafs_workspace, 1 : nx + nu + 2), MOI.SecondOrderCone(nx + nu + 2))

      ## TODO: SOC does not always return t >= |x|... 
      # v6
      arg[v6_offset + j_ind] = model.solver_state_internal.proj_nleafs_workspace[1]
      # t_norm = LA.norm(model.solver_state_internal.proj_nleafs_workspace[2: nx + nu + 2])
      # if t_norm > model.solver_state_internal.proj_nleafs_workspace[1]
      #   arg[v6_offset + j_ind] = t_norm
      # else
      #   arg[v6_offset + j_ind] = model.solver_state_internal.proj_nleafs_workspace[1]
      # end
      # @assert arg[v6_offset + j_ind] >= LA.norm(model.solver_state_internal.proj_nleafs_workspace[2: nx + nu + 2])
      # v3
      for k = 1:nx
        arg[v3_offset + (j_ind - 1) * nx + k] = model.solver_state_internal.proj_nleafs_workspace[k + 1]
      end
      # v4 
      for k = 1:nu
        arg[v4_offset + (j_ind - 1) * nu + k] = model.solver_state_internal.proj_nleafs_workspace[1 + nx + k]
      end
      # v5
      arg[v5_offset + j_ind]= model.solver_state_internal.proj_nleafs_workspace[1 + nx + nu + 1]
    end
  end

  # v7
  if ! isnothing(v7_offset)
    project_onto_nonleaf_constraints!(
      model.problem_definition.constraints, 
      view(arg, v7_offset + 1 : v7_offset + model.problem_definition.constraints.nΓ_nonleaf),
    )
  end
end

function prox_h_conj!(
  model :: MODEL_IMPLICITL,
  arg :: AbstractArray{TF, 1},
  sigma :: TF
) where {TF <: Real}
"""
This function accepts an argument arg and parameter sigma.
With these, it computes the prox_h_conj^sigma(arg) and stores the result in arg.

arg <- prox_h_conj^sigma(arg)
"""

  @simd for i = 1:model.solver_state.nv
    @inbounds arg[i] = arg[i] / sigma
  end

  ### Add halves
  @simd for i in model.solver_state_internal.v5_inds
    @inbounds arg[i] -= 0.5
  end
  @simd for i in model.solver_state_internal.v6_inds
    @inbounds arg[i] += 0.5
  end
  @simd for i in model.solver_state_internal.v12_inds
    @inbounds arg[i] -= 0.5
  end
  @simd for i in model.solver_state_internal.v13_inds
    @inbounds arg[i] += 0.5
  end

  ### Copy the modified dual vector
  copyto!(model.solver_state_internal.prox_v_workspace, arg)

  # arg <- proj_S3(v_modified)
  project_on_leaf_constraints!(model, arg)
  project_on_nonleaf_constraints!(model, arg)

  @simd for i = 1:model.solver_state.nv
    @inbounds arg[i] = sigma * (model.solver_state_internal.prox_v_workspace[i] - arg[i])
  end
end