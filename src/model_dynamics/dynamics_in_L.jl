############################################################
# Build step
############################################################

function get_nz(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
  nx_total = problem_definition.scen_tree.n * problem_definition.nx   # Every node has a state

  return (problem_definition.scen_tree.n_non_leaf_nodes * problem_definition.nu # One input per non leaf node
              + nx_total                                        # Number of state variables over the tree
              + problem_definition.scen_tree.n                                     # s variable: 1 component per node
              + problem_definition.scen_tree.n_non_leaf_nodes * length(rms[1].b))  # One y variable for each non leaf node

  # todo: to support non uniform risk measures, replace the `length(rms[1].b)` by a summation of the lengths of all the risk measures' b vector.
end

function get_nL(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
  nL = 0

  ### Risk constraints
  for rm in rms
    nL += size(rm.A)[2] # (3.20a)
    nL += length(rm.b)  # (3.20b)
    nL += (1 + length(rm.b)) # (3.20c)
  end

  ### Cost constraints
  for _ = 1:problem_definition.scen_tree.n_leaf_nodes # for each scenario
    nL += problem_definition.nx * problem_definition.scen_tree.N # Select the states along this scenario
    nL += problem_definition.nu * (problem_definition.scen_tree.N - 1) # select the inputs along this scenario
    nL += 1 # select the s variable
  end

  ### Dynamics constraints
  """
  We have that x+ = Ax + Bu for each non root node x+.
  Reformulate to (n-1) equations of the form x+ - Ax - Bu = 0
  """
  nL += problem_definition.nx * (problem_definition.scen_tree.n - 1)

  ### Impose initial state
  nL += problem_definition.nx

  return nL
end

function construct_L_risk_a(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1, nz :: TI) where {TI <: Integer}
  L_I = Float64[]
  L_J = Float64[]
  L_V = Float64[]

  s_inds = z_to_s(problem_definition)
  y_inds = z_to_y(problem_definition)

  for i = 1:scen_tree.n_non_leaf_nodes
    r = problem_definition.rms[i]

    ## Construct submatrix S
    
    # Construct Ss, the matrix that selects the values of s on the child node
    Is = Float64[]
    Js = Float64[]
    Vs = Float64[]

    Js = s_inds[problem_definition.scen_tree.child_mapping[i]]
    append!(Is, [i for i in collect(1:length(Js))])
    append!(Vs, [1 for _ in 1:length(Js)])
    Ss = sparse(Is, Js, Vs, length(Is), y_inds[1] - 1)

    # Construct Sy, the matrix that selects the values of y on the given node
    Is = Float64[]
    Js = Float64[]
    Vs = Float64[]

    ny = length(r.b)

    # TODO: In fact, the use of y_inds here is completely redundant since we substract the offset anyway, reformulate.
    Js = y_inds[(i - 1) * ny + 1 : i * ny] .- (y_inds[1] - 1)
    append!(Is, [k for k in collect(1:length(Js))])
    append!(Vs, [1 for _ in 1:length(Js)])
    Sy = sparse(Is, Js, Vs, length(Is), nz - y_inds[1] + 1)

    # Use Ss and Sy to assemble S
    S = hcat(sparse(r.A') * Ss, sparse(r.B') * Sy)

    ## Add nonzero entries of S to L
    SI, SJ, SV = findnz(S)
    if i > 1
        append!(L_I, SI .+ maximum(L_I))
    else
        append!(L_I, SI)
    end
    append!(L_J, SJ); append!(L_V, SV)
  end

  return L_I, L_J, L_V
end

function construct_L_risk_b(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
  L_I = Float64[]
  L_J = Float64[]
  L_V = Float64[]

  y_inds = z_to_y(problem_definition)
  for i = 1:problem_definition.scen_tree.n_non_leaf_nodes
    # Define ny for the given node
    r = problem_definition.rms[i]
    ny = length(r.b)

    ind = (i - 1) * ny + 1 : i * ny
    append!(L_I, [i > 1 ? j + maximum(L_I) : j for j in collect(1 : ny)])
    append!(L_J, y_inds[ind])
    append!(L_V, [1 for _ in 1:ny])
  end

  return L_I, L_J, L_V
end

function construct_L_risk_c(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
  L_I = Float64[]
  L_J = Float64[]
  L_V = Float64[]

  s_inds = z_to_s(problem_definition)
  y_inds = z_to_y(problem_definition)

  for i = 1:problem_definition.scen_tree.n_non_leaf_nodes
    # Define ny for the given node
    r = problem_definition.rms[i]
    ny = length(r.b)

    append!(L_J, s_inds[i])
    ind = (i - 1) * ny + 1 : i * ny
    append!(L_J, y_inds[ind])

    append!(L_I, [i > 1 ? j + maximum(L_I) : j for j in collect(1 : (ny + 1))])
    append!(L_V, [-1 for _ in 1:(ny + 1)])
  end

  return L_I, L_J, L_V
end

function construct_L_cost(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
  L_I = Float64[]
  L_J = Float64[]
  L_V = Float64[]

  x_inds = z_to_x(problem_definition)
  u_inds = z_to_u(problem_definition)
  s_inds = z_to_s(problem_definition)

  for i = problem_definition.scen_tree.leaf_node_min_index : problem_definition.scen_tree.leaf_node_max_index
    xs = x_inds[node_to_x(problem_definition, i)]
    us = []
    ss = [s_inds[i]]

    n = i
    for _ = problem_definition.scen_tree.N-1:-1:1
        n = problem_definition.scen_tree.anc_mapping[n]
        pushfirst!(xs, x_inds[node_to_x(problem_definition, n)]...)
        pushfirst!(us, u_inds[node_to_u(problem_definition, n)]...)
    end
    append!(L_J, xs)
    append!(L_J, us)
    append!(L_J, ss)
  end
  append!(L_I, [i for i in 1 : length(L_J)])
  append!(L_V, [1 for _ in 1 : length(L_J)])

  return L_I, L_J, L_V
end

function construct_L_dynamics(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
  L_I = Float64[]
  L_J = Float64[]
  L_V = Float64[]

  u_offset = scen_tree.n * problem_definition.nx
  I_offset = 0

  ### For each non root node we have x+ = A x + B u <=> A x - I x+ + B u
  for i = 2 : scen_tree.n
      w = problem_definition.scen_tree.node_info[i].w
      A = problem_definition.dynamics.A[w]; B = problem_definition.dynamics.B[w]
      I = LA.I(problem_definition.nx)
      anc_node = problem_definition.scen_tree.anc_mapping[i]

      # A
      AI, AJ, AV = findnz(sparse(A))
      append!(L_I, AI .+ I_offset)
      append!(L_J, AJ .+ (problem_definition.nx * (anc_node - 1)))
      append!(L_V, AV)

      # -I
      AI, AJ, AV = findnz(sparse(-I))
      append!(L_I, AI .+ I_offset)
      append!(L_J, AJ .+ (problem_definition.nx) * (i - 1))
      append!(L_V, AV)

      # B
      AI, AJ, AV = findnz(sparse(B))
      append!(L_I, AI .+ I_offset)
      append!(L_J, AJ .+ (problem_definition.nu * (i - 1)) .+ u_offset)
      append!(L_V, AV)

      I_offset += size(A)[1]
  end

  return L_I, L_J, L_V
end

function construct_L_initial_cond(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
  L_I = collect(1:problem_definition.nx)
  L_J = collect(1:problem_definition.nx)
  L_V = ones(problem_definition.nx)

  return L_I, L_J, L_V
end

function get_L(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1, nz :: TI, nL :: TI) where {TI <: Integer}
  # Risk constraints
  L_I, L_J, L_V = construct_L_risk_a(problem_definition, nz)

  L_II, L_JJ, L_VV = construct_L_risk_b(problem_definition)
  append!(L_I, L_II .+ maximum(L_I)); append!(L_J, L_JJ); append!(L_V, L_VV)

  L_II, L_JJ, L_VV = construct_L_risk_c(problem_definition)
  append!(L_I, L_II .+ maximum(L_I)); append!(L_J, L_JJ); append!(L_V, L_VV)

  # Cost
  L_II, L_JJ, L_VV = construct_L_cost(problem_definition)
  append!(L_I, L_II .+ maximum(L_I)); append!(L_J, L_JJ); append!(L_V, L_VV)

  # Dynamics
  L_II, L_JJ, L_VV = construct_L_dynamics(problem_definition)
  append!(L_I, L_II .+ maximum(L_I)); append!(L_J, L_JJ); append!(L_V, L_VV)

  # Initial condition
  L_II, L_JJ, L_VV = construct_L_initial_cond(problem_definition)
  append!(L_I, L_II .+ maximum(L_I)); append!(L_J, L_JJ); append!(L_V, L_VV)

  return sparse(L_I, L_J, L_V, nL, nz)
end

function get_L_inds(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1, nz :: TI, nL :: TI) where {TI <: Integer}
  
  offset = 0

  ### inds_L_risk_a
  inds_L_risk_a = Union{UnitRange{Int64}, Int64}[]
  for i = 1:problem_definition.scen_tree.n_non_leaf_nodes
    part = size(rms[i].A)[2]
    inds = offset + 1 : offset + part
    append!(inds_L_risk_a, [inds])

    offset += part
  end

  ### inds_L_risk_b
  inds_L_risk_b = Union{UnitRange{Int64}, Int64}[]
  for i = 1:problem_definition.scen_tree.n_non_leaf_nodes
    part = length(rms[i].b)
    inds = offset + 1 : offset + part
    append!(inds_L_risk_b, [inds])

    offset += part
  end

  ### inds_L_risk_c
  inds_L_risk_c = Union{UnitRange{Int64}, Int64}[]
  for i = 1:problem_definition.scen_tree.n_non_leaf_nodes
    part = length(rms[i].b) + 1
    inds = offset + 1 : offset + part
    append!(inds_L_risk_c, [inds])

    offset +=  part
  end

  ### inds_L_cost
  inds_L_cost = Union{UnitRange{Int64}, Int64}[]
  for _ = 1:problem_definition.scen_tree.n_leaf_nodes
    part = problem_definition.scen_tree.N * problem_definition.nx + (problem_definition.scen_tree.N - 1) * problem_definition.nu + 1
    inds = offset + 1 : offset + part
    append!(inds_L_cost, [inds])

    offset += part
  end

  ### inds_L_dynamics
  part = problem_definition.nx * (problem_definition.scen_tree.n - 1)
  inds_L_dynamics = offset + 1 : offset + part

  return inds_L_risk_a, inds_L_risk_b, inds_L_risk_c, inds_L_cost, inds_L_dynamics

end


####################
### Internal utility functions
####################

"""
Get the indices of the x variable's components.
"""
function z_to_x(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
    return collect(
        1 : problem_definition.nx * problem_definition.scen_tree.n
    )
end    

function z_to_u(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
    return collect(
        problem_definition.nx * problem_definition.scen_tree.n + 1 : 
        problem_definition.nx * problem_definition.scen_tree.n + 
          (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu
    )
end

function z_to_s(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
    return collect(
      problem_definition.nx * problem_definition.scen_tree.n + 
        (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 1 :
      problem_definition.nx * problem_definition.scen_tree.n + 
        (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 
        problem_definition.scen_tree.n
    )
end

function z_to_y(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1)
  """
  TODO: replace n_y by a summation over all ny to support non uniform risk measures
  """

  ny = length(problem_definition.rms[1].b)

    return collect(
      problem_definition.nx * problem_definition.scen_tree.n + 
        (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 
        problem_definition.scen_tree.n + 1 :
        problem_definition.nx * problem_definition.scen_tree.n + 
          (problem_definition.scen_tree.n_non_leaf_nodes) * problem_definition.nu + 
          problem_definition.scen_tree.n + problem_definition.scen_tree.n_non_leaf_nodes * ny
    )
end


function node_to_x(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1, i :: Int64)
  return collect(
      (i - 1) * problem_definition.nx + 1 : i * problem_definition.nx
  )
end

function node_to_u(problem_definition :: GENERIC_PROBLEM_DEFINITIONV1, i :: Int64)
  return collect(
      (i - 1) * problem_definition.nu + 1 : i * problem_definition.nu
  )
end


########################################
### Solve step
########################################

function L!(model :: MODEL_DYNAMICSL, z :: AbstractArray{TF, 1}, v :: AbstractArray{TF, 1}) where {TF <: Real}
  """
    Updates v such that 
    v <- L z
    without constructing L explicitly.
  """

  v[1:model.solver_state.nv] = model.solver_state_internal.L * z

end

function L_transpose!(model :: MODEL_DYNAMICSL, z :: AbstractArray{TF, 1}, v :: AbstractArray{TF, 1}) where {TF <: Real}
  """
    Updates z such that 
    z <- L' v
    without constructing L explicitly.
  """
  z[1:model.solver_state.nz] = model.solver_state_internal.L' * v
end

function spock_mul!(
  model :: MODEL_DYNAMICSL,
  C :: AbstractArray{TF, 1},
  trans :: Bool,
  B :: AbstractArray{TF, 1},
  α :: TF,
  β :: TF
) where {TF <: Real}
"""
In place matrix-vector multiply-Add

C <- model.L * B * α + C * β when trans is false, otherwise
C <- model.L' * B * α + C * β
"""
LA.mul!(C, trans ? model.solver_state_internal.L' : model.solver_state_internal.L, B, α, β)
end

function spock_dot(
  model :: MODEL_DYNAMICSL,
  arg1_z :: AbstractArray{TF, 1}, 
  arg1_v :: AbstractArray{TF, 1}, 
  arg2_z :: AbstractArray{TF, 1}, 
  arg2_v :: AbstractArray{TF, 1}, 
  gamma :: TF, 
  sigma :: TF
) where {TF <: Real}

  # todo: preallocate a workspace to avoid additional allocations
  return arg1_z' * (1. / gamma * arg2_z - model.solver_state_internal.L' *arg2_v) + 
    arg1_v' * (-model.solver_state_internal.L * arg2_z + 1. / sigma * arg2_v)
end

function spock_dot(
  model :: MODEL_DYNAMICSL,
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
  model :: MODEL_DYNAMICSL,
  arg :: AbstractArray{TF, 1}, 
  gamma :: TF, 
  sigma :: TF
) where {TF <: Real}

  return sqrt(spock_dot(model, arg, arg, gamma, sigma))
end

function spock_norm(
  model :: MODEL_DYNAMICSL,
  arg_z :: AbstractArray{TF, 1}, 
  arg_v :: AbstractArray{TF, 1}, 
  gamma :: TF, 
  sigma :: TF
) where {TF <: Real}

  return sqrt(spock_dot(model, arg_z, arg_v, arg_z, arg_v, gamma, sigma))
end

function prox_f!(
  model :: MODEL_DYNAMICSL,
  arg :: AbstractArray{TF, 1},
  gamma :: TF
) where {TF <: Real}
"""
This function accepts an argument arg and parameter gamma.
With these, it computes the prox_f^gamma(arg) and stores the result in arg.

arg <- prox_f^gamma(arg)
"""
  arg[model.solver_state_internal.s_inds[1]] -= gamma    
end

function prox_cost!(
  model :: MODEL_DYNAMICSL,
  arg :: AbstractArray{TF, 1},
  scen_ind :: TI,
  x_inds :: UnitRange{TI},
  u_inds :: UnitRange{TI},
  gamma :: TF
) where {TF <: Real, TI <: Integer}

  node = scen_ind + model.problem_definition.scen_tree.leaf_node_min_index - 1

  for t = model.problem_definition.scen_tree.N : -1 : 1
    # x' Q x
    arg[x_inds[1 + (t - 1) * nx : t * nx]] = arg[x_inds[1 + (t - 1) * nx : t * nx]] ./ (LA.diag(model.problem_definition.cost.Q[node]) .+ 1. / gamma) / gamma
    # u' R u
    if t < model.problem_definition.scen_tree.N
      arg[u_inds[1 + (t - 1) * nu : t * nu]] = arg[u_inds[1 + (t - 1) * nu : t * nu]] ./(LA.diag(model.problem_definition.cost.R[node]) .+ 1. / gamma) / gamma
    end

    if t > 1
      node = model.problem_definition.scen_tree.anc_mapping[node]
    end
  end
end

function psi!(
  model :: MODEL_DYNAMICSL,
  gamma:: TF,
  arg :: AbstractArray{TF, 1},
  scen_ind :: TI,
  x_inds :: UnitRange{TI},
  u_inds :: UnitRange{TI},
  t_ind :: TI
) where {TF <: Real, TI <: Integer}
  node = scen_ind + model.problem_definition.scen_tree.leaf_node_min_index - 1
  nx = model.problem_definition.nx
  nu = model.problem_definition.nu

  ### v_bisection workspace <- prox_f^gamma(arg)
  copyto!(model.solver_state_internal.v_bisection_workspace, arg)
  prox_cost!(model, model.solver_state_internal.v_bisection_workspace, scen_ind, x_inds, u_inds, gamma)

  ### f = 0.5 x' Q x + 0.5 u' R u
  f = 0
  for t = model.problem_definition.scen_tree.N : -1 : 1
    # x' Q x
    f += model.solver_state_internal.v_bisection_workspace[x_inds[1 + (t - 1) * nx : t * nx]]' * model.problem_definition.cost.Q[node] * model.solver_state_internal.v_bisection_workspace[x_inds[1 + (t - 1) * nx : t * nx]]
    
    # u' R u
    if t < model.problem_definition.scen_tree.N
      f += model.solver_state_internal.v_bisection_workspace[u_inds[1 + (t - 1) * nu : t * nu]]' * model.problem_definition.cost.R[node] * model.solver_state_internal.v_bisection_workspace[u_inds[1 + (t - 1) * nu : t * nu]]
    end

    if t > 1
      node = model.problem_definition.scen_tree.anc_mapping[node]
    end
  end
  f *= 0.5

  return f - gamma - arg[t_ind]
end

function bisection_method!(
  model :: MODEL_DYNAMICSL,
  g_lb :: TF,
  g_ub :: TF,
  tol :: TF,
  arg :: AbstractArray{TF, 1},
  scen_ind :: TI,
  x_inds :: UnitRange{TI},
  u_inds :: UnitRange{TI},
  t_ind :: TI
) where {TF <: Real, TI <: Integer}

  g_new = (g_lb + g_ub) / 2
  ps = psi!(  
    model, 
    g_new,
    arg,
    scen_ind,
    x_inds, 
    u_inds, 
    t_ind
  )
  while abs(g_ub - g_lb) > tol * g_lb
      if sign(ps) > 0
          g_lb = g_new
      elseif sign(ps) < 0
          g_ub = g_new
      else
          return g_new
          error("Should never happen")
      end
      g_new = (g_lb + g_ub) / 2
      ps = psi!(  
        model, 
        g_new,
        arg,
        scen_ind,
        x_inds, 
        u_inds, 
        t_ind
      )
  end
  return g_new
end

function epigraph_bisection!(
  model :: MODEL_DYNAMICSL,
  arg :: AbstractArray{TF, 1},
  scen_ind :: TI
) where {TF, TI}
"""
  arg is the vector of dual variables

  for the given scenario, this function projects (xu, t) on the epigraph of 
  the quadratic cost function f_sigma(x, u) = 0.5 x' Q_sigma x + 0.5 u' R_sigma u 
"""
  node = scen_ind + model.problem_definition.scen_tree.leaf_node_min_index - 1
  nx = model.problem_definition.nx
  nu = model.problem_definition.nu
  x_inds = model.solver_state_internal.inds_L_cost[scen_ind][1:nx * model.problem_definition.scen_tree.N]
  u_inds = model.solver_state_internal.inds_L_cost[scen_ind][
    nx * model.problem_definition.scen_tree.N + 1 : nx * model.problem_definition.scen_tree.N + nu * (model.problem_definition.scen_tree.N - 1)
  ]
  t_ind = model.solver_state_internal.inds_L_cost[scen_ind][end]

  ### f = 0.5 x' Q x + 0.5 u' R u
  f = 0
  for t = model.problem_definition.scen_tree.N : -1 : 1
    # x' Q x
    f += arg[x_inds[1 + (t - 1) * nx : t * nx]]' * model.problem_definition.cost.Q[node] * arg[x_inds[1 + (t - 1) * nx : t * nx]]
    
    # u' R u
    if t < model.problem_definition.scen_tree.N
      f += arg[u_inds[1 + (t - 1) * nu : t * nu]]' * model.problem_definition.cost.R[node] * arg[u_inds[1 + (t - 1) * nu : t * nu]]
    end

    if t > 1
      node = model.problem_definition.scen_tree.anc_mapping[node]
    end
  end
  f *= 0.5

  # println("Currently scen $(scen_ind): f = $(f), t = $(arg[t_ind])")

  if f > arg[t_ind]
    # Update arg
    gamma_lb = 1e-8
    gamma_ub = f - arg[t_ind]
    bisection_tol = 1e-8
    gamma_star = bisection_method!(
      model, 
      gamma_lb, 
      gamma_ub, 
      bisection_tol, 
      arg,
      scen_ind,
      x_inds, 
      u_inds, 
      t_ind
    )
    #arg[x_inds], arg[u_inds] <- prox^gamma_star([arg[x_inds], arg[u_inds]])
    prox_cost!(model, arg, scen_ind, x_inds, u_inds, gamma_star)
    arg[t_ind] += gamma_star
  end
end

function projection_cost_epigraph!(
  model :: MODEL_DYNAMICSL,
  arg :: AbstractArray{TF, 1},
) where {TF <: Real}

  for scen_ind = 1:model.problem_definition.scen_tree.n_leaf_nodes
    epigraph_bisection!(model, arg, scen_ind)
  end

end

function projection!(
  model :: MODEL_DYNAMICSL,
  arg :: AbstractArray{TF, 1},
) where {TF <: Real}
"""
arg <- proj_C(arg)
"""
  ### Risk constraints

  # a)
  for (i, ind) in enumerate(model.solver_state_internal.inds_L_risk_a)
    # TODO: Support looping over subcones here to support more general risk measures
    # Proj_{C_polar}(x) = - Proj_{C_dual}(-x)
    arg[ind] = -MOD.projection_on_set(MOD.DefaultDistance(), -arg[ind], MOI.dual_set(model.problem_definition.rms[i].C.subcones[1]))
  end

  # b)
  for (i, ind) in enumerate(model.solver_state_internal.inds_L_risk_b)
    # TODO: Support looping over subcones here to support more general risk measures
    # Proj_{C_polar}(x) = - Proj_{C_dual}(-x)
    arg[ind] = -MOD.projection_on_set(MOD.DefaultDistance(), -arg[ind], model.problem_definition.rms[i].D.subcones[1])
  end

  # c)
  for (i, ind) in enumerate(model.solver_state_internal.inds_L_risk_c)
    dot_p = LA.dot(arg[ind[2:end]], model.problem_definition.rms[i].b)
    dot_p += arg[1]

    if dot_p > 0
      dot_p /= (LA.dot(model.problem_definition.rms[i].b, model.problem_definition.rms[i].b) + 1.)
      
      arg[ind[1]] = arg[ind[1]] - dot_p
      @simd for j = 2:length(ind)
        @inbounds @fastmath arg[ind[j]] = arg[ind[j]] - dot_p * model.problem_definition.rms[i].b[j - 1]
      end
    end
  end

  ### Cost epigraphs
  projection_cost_epigraph!(model, arg)

  ### Dynamics
  @simd for ind in model.solver_state_internal.inds_L_dynamics
    @inbounds @fastmath arg[ind] = 0.
  end

  ### Initial condition
  arg[end - model.problem_definition.nx + 1 : end] = model.problem_definition.x0
end

function prox_h_conj!(
  model :: MODEL_DYNAMICSL,
  arg :: AbstractArray{TF, 1},
  sigma :: TF
) where {TF <: Real}
"""
This function accepts an argument arg and parameter sigma.
With these, it computes the prox_h_conj^sigma(arg) and stores the result in arg.

arg <- prox_h_conj^sigma(arg)
"""
  # v_projection_arg <- arg / sigma
  copyto!(model.solver_state_internal.v_projection_arg, arg)
  LA.BLAS.scal!(
    model.solver_state.nv, 
    1. / sigma, 
    model.solver_state_internal.v_projection_arg, 
    stride(model.solver_state_internal.v_projection_arg, 1)
  )

  projection!(model, model.solver_state_internal.v_projection_arg)

  @simd for i = 1:model.solver_state.nv
    @inbounds @fastmath arg[i] = arg[i] - sigma * model.solver_state_internal.v_projection_arg[i]
  end
end