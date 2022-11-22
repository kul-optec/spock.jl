#########
### Generic Chambolle-Pock implementation
#########

function update_zbar!(
  model :: MODEL_CP,
  gamma :: TF
) where {TF <: Real}

  # zbar = -gamma * L' * v + z
  copyto!(model.solver_state.zbar, model.solver_state.z)
  spock_mul!(model, model.solver_state.zbar, true, model.solver_state.v, -gamma, 1.)
  # zbar = prox_f(zbar)
  prox_f!(model, model.solver_state.zbar, gamma)
end

function update_vbar!(
  model :: MODEL_CP,
  sigma :: TF
) where {TF <: Real}

  # z_workspace = 2 * zbar - z
  copyto!(model.solver_state.z_workspace, model.solver_state.z)
  LA.BLAS.axpby!(2., model.solver_state.zbar, -1., model.solver_state.z_workspace)

  # vbar = sigma * L (z_workspace) + v
  copyto!(model.solver_state.vbar, model.solver_state.v)
  spock_mul!(model, model.solver_state.vbar, false, model.solver_state.z_workspace, sigma, 1.)

  # vbar = prox_h_conj(vbar)
  prox_h_conj!(model, model.solver_state.vbar, sigma)
end

function update_z!(
  model :: MODEL_CP,
  lambda :: TF
) where {TF <: Real}

  @simd for i = 1:model.solver_state.nz
    @inbounds @fastmath model.solver_state.z[i] = lambda * model.solver_state.zbar[i] + (1 - lambda) * model.solver_state.z[i]
  end
end

function update_v!(
  model :: MODEL_CP,
  lambda :: TF
) where {TF <: Real}

  @simd for i = 1:model.solver_state.nv
    @inbounds @fastmath model.solver_state.v[i] = lambda * model.solver_state.vbar[i] + (1 - lambda) * model.solver_state.v[i]
  end
end

function should_terminate!(
  model :: MODEL_CP,
  alpha1 :: TF,
  alpha2 :: TF,
  tol :: TF,
  verbose :: VERBOSE_LEVEL
) where {TF <: Real}

  for i = 1:model.solver_state.nz
    model.solver_state.delta_z[i] = model.solver_state.z[i] - model.solver_state.z_old[i]
  end
  for i = 1:model.solver_state.nv
    model.solver_state.delta_v[i] = model.solver_state.v[i] - model.solver_state.v_old[i]
  end

  copyto!(model.solver_state.xi_1, model.solver_state.delta_z)
  copyto!(model.solver_state.xi_2, model.solver_state.delta_v)

  # xi_1 <- -1/alpha2 xi_1 + L' * delta_v
  spock_mul!(model, model.solver_state.xi_1, true, model.solver_state.delta_v, 1., -1. / alpha1)
  # xi_2 <- -1/alpha2 xi_2 + L * delta_z
  spock_mul!(model, model.solver_state.xi_2, false, model.solver_state.delta_z, 1., -1. / alpha2)

  xi_1 = LA.norm(model.solver_state.xi_1, Inf)
  xi_2 = LA.norm(model.solver_state.xi_2, Inf)

  res = false

  if verbose === LOG
    copyto!(model.solver_state.xi, model.solver_state.xi_1)
    spock_mul!(model, model.solver_state.xi, true, model.solver_state.xi_2, 1., 1.)

    xi = LA.norm(model.solver_state.xi, Inf)

    open("examples/output/xi_cp.dat", "a") do io
      writedlm(io, xi)
    end
    open("examples/output/xi1_cp.dat", "a") do io
      writedlm(io, xi_1)
    end
    open("examples/output/xi2_cp.dat", "a") do io
      writedlm(io, xi_2)
    end

    if xi <= tol
      res = true
    end
  # Only check third condition when first two have succeeded (avoid extra L' if possible)
  elseif xi_1 <= max(tol * model.solver_state.res_0[1], tol) && xi_2 <= max(tol * model.solver_state.res_0[2], tol)
    # copyto!(model.solver_state.xi, model.solver_state.xi_1)
    # spock_mul!(model, model.solver_state.xi, true, model.solver_state.xi_2, 1., 1.)

    # xi = LA.norm(model.solver_state.xi, Inf)

    # if xi <= tol
    #   return true
    # end
    res = true
  end

  if model.solver_state.res_0[1] === -Inf
    model.solver_state.res_0[1] = xi_1
  end
  if model.solver_state.res_0[2] === -Inf
    model.solver_state.res_0[2] = xi_2
  end

  return res

end

# function should_terminate_old!(
#   model :: MODEL_CP,
#   alpha1 :: TF,
#   alpha2 :: TF,
#   tol :: TF,
#   verbose :: VERBOSE_LEVEL
# ) where {TF <: Real}

#   for i = 1:model.solver_state.nz
#     model.solver_state.delta_z[i] = model.solver_state.z[i] - model.solver_state.z_old[i]
#   end
#   for i = 1:model.solver_state.nv
#     model.solver_state.delta_v[i] = model.solver_state.v[i] - model.solver_state.v_old[i]
#   end

#   copyto!(model.solver_state.xi_1, model.solver_state.delta_z)
#   copyto!(model.solver_state.xi_2, model.solver_state.delta_v)

#   # xi_1 <- -1/alpha2 xi_1 + L' * delta_v
#   spock_mul!(model, model.solver_state.xi_1, true, model.solver_state.delta_v, 1., -1. / alpha1)
#   # xi_2 <- -1/alpha2 xi_2 + L * delta_z
#   spock_mul!(model, model.solver_state.xi_2, false, model.solver_state.delta_z, 1., -1. / alpha2)

#   xi_1 = LA.norm(model.solver_state.xi_1, Inf)
#   xi_2 = LA.norm(model.solver_state.xi_2, Inf)

#   if verbose === LOG
#     copyto!(model.solver_state.xi, model.solver_state.xi_1)
#     spock_mul!(model, model.solver_state.xi, true, model.solver_state.xi_2, 1., 1.)

#     xi = LA.norm(model.solver_state.xi, Inf)

#     open("examples/output/xi_cp.dat", "a") do io
#       writedlm(io, xi)
#     end
#     open("examples/output/xi1_cp.dat", "a") do io
#       writedlm(io, xi_1)
#     end
#     open("examples/output/xi2_cp.dat", "a") do io
#       writedlm(io, xi_2)
#     end

#     if xi <= tol
#       return true
#     end
#   # Only check third condition when first two have succeeded (avoid extra L' if possible)
#   elseif xi_1 <= tol && xi_2 <= tol
#     copyto!(model.solver_state.xi, model.solver_state.xi_1)
#     spock_mul!(model, model.solver_state.xi, true, model.solver_state.xi_2, 1., 1.)

#     xi = LA.norm(model.solver_state.xi, Inf)

#     if xi <= tol
#       return true
#     end
#   end

#   return false

# end

# precompile(should_terminate!, (CPOCK, Float64, Float64, Float64))

function run_cp!(
  model :: MODEL_CP;
  MAX_ITER_COUNT :: Int64 = 5000,
  tol :: Float64 = 1e-3,
  verbose :: VERBOSE_LEVEL = SILENT,
  gamma :: Union{Float64, Nothing} = nothing,
  sigma :: Union{Float64, Nothing} = nothing,
  lambda :: Float64 = 1.
)

  if sigma === nothing || gamma === nothing
    sigma = 0.99 / sqrt(model.solver_state.L_norm)
    gamma = sigma
  end

  iter = 0

  while iter < MAX_ITER_COUNT
    # Update the z / v -old variables
    copyto!(model.solver_state.z_old, model.solver_state.z)
    copyto!(model.solver_state.v_old, model.solver_state.v)

    update_zbar!(model, gamma)
    update_vbar!(model, sigma)
    
    update_z!(model, lambda)
    update_v!(model, lambda)

    # Check termination criterion
    if should_terminate!(
      model,
      gamma,
      sigma,
      tol,
      verbose
    )
      break
    end

    iter += 1
  end

  println("Finished in $(iter) iterations.")

end