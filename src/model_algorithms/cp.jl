#########
### Generic Chambolle-Pock implementation
#########

function update_zbar!(
  model :: MODEL_CP,
  gamma :: TF
) where {TF <: Real}

  # zbar = -gamma * L' * v + z
  copyto!(model.state.zbar, model.state.z)
  spock_mul!(model, model.state.zbar, true, model.state.v, -gamma, 1.)
  # zbar = prox_f(zbar)
  prox_f!(model, model.state.zbar, gamma)
end

function update_vbar!(
  model :: MODEL_CP,
  sigma :: TF
) where {TF <: Real}

  # z_workspace = 2 * zbar - z
  copyto!(model.state.z_workspace, model.state.z)
  LA.BLAS.axpby!(2., model.state.zbar, -1., model.state.z_workspace)

  # vbar = sigma * L (z_workspace) + v
  copyto!(model.state.vbar, model.state.v)
  spock_mul!(model, model.state.vbar, false, model.state.z_workspace, sigma, 1.)

  # vbar = prox_h_conj(vbar)
  prox_h_conj!(model, model.state.vbar, sigma)
end

function update_z!(
  model :: MODEL_CP,
  lambda :: TF
) where {TF <: Real}

  @simd for i = 1:model.state.nz
    @inbounds @fastmath model.state.z[i] = lambda * model.state.zbar[i] + (1 - lambda) * model.state.z[i]
  end
end

function update_v!(
  model :: MODEL_CP,
  lambda :: TF
) where {TF <: Real}

  @simd for i = 1:model.state.nv
    @inbounds @fastmath model.state.v[i] = lambda * model.state.vbar[i] + (1 - lambda) * model.state.v[i]
  end
end

function should_terminate!(
  model :: MODEL_CP,
  alpha1 :: TF,
  alpha2 :: TF,
  tol :: TF,
  verbose :: VERBOSE_LEVEL
) where {TF <: Real}

  for i = 1:model.state.nz
    model.state.Δz[i] = model.state.z[i] - model.state.z_old[i]
  end
  for i = 1:model.state.nv
    model.state.Δv[i] = model.state.v[i] - model.state.v_old[i]
  end

  copyto!(model.state.xi_1, model.state.Δz)
  copyto!(model.state.xi_2, model.state.Δv)

  # xi_1 <- -1/alpha2 xi_1 + L' * Δv
  spock_mul!(model, model.state.xi_1, true, model.state.Δv, 1., -1. / alpha1)
  # xi_2 <- -1/alpha2 xi_2 + L * Δz
  spock_mul!(model, model.state.xi_2, false, model.state.Δz, 1., -1. / alpha2)

  xi_1 = LA.norm(model.state.xi_1, Inf)
  xi_2 = LA.norm(model.state.xi_2, Inf)

  res = false

  if verbose === LOG
    copyto!(model.state.xi, model.state.xi_1)
    spock_mul!(model, model.state.xi, true, model.state.xi_2, 1., 1.)

    xi = LA.norm(model.state.xi, Inf)

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
  elseif xi_1 <= max(tol * model.state.res_0[1], tol) && xi_2 <= max(tol * model.state.res_0[2], tol)
    # copyto!(model.state.xi, model.state.xi_1)
    # spock_mul!(model, model.state.xi, true, model.state.xi_2, 1., 1.)

    # xi = LA.norm(model.state.xi, Inf)

    # if xi <= tol
    #   return true
    # end
    res = true
  end

  if model.state.res_0[1] === -Inf
    model.state.res_0[1] = xi_1
  end
  if model.state.res_0[2] === -Inf
    model.state.res_0[2] = xi_2
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

#   for i = 1:model.state.nz
#     model.state.Δz[i] = model.state.z[i] - model.state.z_old[i]
#   end
#   for i = 1:model.state.nv
#     model.state.Δv[i] = model.state.v[i] - model.state.v_old[i]
#   end

#   copyto!(model.state.xi_1, model.state.Δz)
#   copyto!(model.state.xi_2, model.state.Δv)

#   # xi_1 <- -1/alpha2 xi_1 + L' * Δv
#   spock_mul!(model, model.state.xi_1, true, model.state.Δv, 1., -1. / alpha1)
#   # xi_2 <- -1/alpha2 xi_2 + L * Δz
#   spock_mul!(model, model.state.xi_2, false, model.state.Δz, 1., -1. / alpha2)

#   xi_1 = LA.norm(model.state.xi_1, Inf)
#   xi_2 = LA.norm(model.state.xi_2, Inf)

#   if verbose === LOG
#     copyto!(model.state.xi, model.state.xi_1)
#     spock_mul!(model, model.state.xi, true, model.state.xi_2, 1., 1.)

#     xi = LA.norm(model.state.xi, Inf)

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
#     copyto!(model.state.xi, model.state.xi_1)
#     spock_mul!(model, model.state.xi, true, model.state.xi_2, 1., 1.)

#     xi = LA.norm(model.state.xi, Inf)

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
    sigma = 0.99 / sqrt(model.state.L_norm)
    gamma = sigma
  end

  iter = 0

  while iter < MAX_ITER_COUNT
    # Update the z / v -old variables
    copyto!(model.state.z_old, model.state.z)
    copyto!(model.state.v_old, model.state.v)

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