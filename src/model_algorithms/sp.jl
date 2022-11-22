#########
### Generic Chambolle-Pock + SuperMann implementation
#########

function update_zbar!(
  model :: MODEL_SP,
  gamma :: TF
) where {TF <: Real}

  # zbar = -gamma * L' * v + z
  copyto!(model.solver_state.zbar, model.solver_state.z)
  spock_mul!(model, model.solver_state.zbar, true, model.solver_state.v, -gamma, 1.)
  # zbar = prox_f(zbar)
  prox_f!(model, model.solver_state.zbar, gamma)
end

function update_vbar!(
  model :: MODEL_SP,
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

function update_residual!(
  model :: MODEL_SP,
  gamma :: TF,
  sigma :: TF
) where {TF <: Real}
  """
  Updates the fixed point residual r = (z - zbar, v - vbar)
  """

  for i in eachindex(model.solver_state.z)
    model.solver_state.rz[i] = model.solver_state.z[i] - model.solver_state.zbar[i]
  end
  for i in eachindex(model.solver_state.v)
    model.solver_state.rv[i] = model.solver_state.v[i] - model.solver_state.vbar[i]
  end

  return spock_norm(model, model.solver_state.rz, model.solver_state.rv, gamma, sigma)
end

function update_z!(
  model :: MODEL_SP,
  lambda :: TF
) where {TF <: Real}

  @simd for i = 1:model.solver_state.nz
    @inbounds @fastmath model.solver_state.z[i] = lambda * model.solver_state.zbar[i] + (1 - lambda) * model.solver_state.z[i]
  end
end

function update_v!(
  model :: MODEL_SP,
  lambda :: TF
) where {TF <: Real}

  @simd for i = 1:model.solver_state.nv
    @inbounds @fastmath model.solver_state.v[i] = lambda * model.solver_state.vbar[i] + (1 - lambda) * model.solver_state.v[i]
  end
end

function should_perform_k0(
  rnorm :: TF,
  c0 :: TF,
  eta :: TF
) where {TF <: Real}

  # TODO: Provide a model parameter to just disable K0 updates?
  return rnorm <= c0 * eta && false
end

function perform_k0!(
  model :: MODEL_SP,
  rnorm :: TF,
  eta :: TF,
  loop :: Bool
) where {TF <: Real}

  eta = rnorm

  for i in eachindex(model.solver_state.z)
    model.solver_state.z[i] += model.solver_state_internal.dz[i]
  end
  for i in eachindex(model.solver_state.v)
    model.solver_state.v[i] += model.solver_state_internal.dv[i]
  end

  # TODO: to use restarted Broyden correctly, update wbar here too
  copyto!(model.solver_state_internal.w, model.solver_state.z)
  copyto!(model.solver_state_internal.u, model.solver_state.v)

  loop = false

  return eta, loop

end

function should_backtrack_loop(
  loop :: Bool, 
  backtrack_count :: TI, 
  MAX_BACKTRACK_COUNT :: TI
) where {TI <: Integer}

  return loop && backtrack_count <= MAX_BACKTRACK_COUNT
end

function generate_candidate_update!(
  model :: MODEL_SP,
  tau :: TF
) where {TF <: Real}

  ### (w, u) = (z, v) + tau (dz, dv)
  for i in eachindex(model.solver_state_internal.w)
    model.solver_state_internal.w[i] = model.solver_state.z[i] + tau * model.solver_state_internal.dz[i]
  end
  for i in eachindex(model.solver_state_internal.u)
    model.solver_state_internal.u[i] = model.solver_state.v[i] + tau * model.solver_state_internal.dv[i]
  end

end

function update_candidate_residual!(
  model :: MODEL_SP,
  gamma :: TF,
  sigma :: TF
) where {TF <: Real}

  ### wbar = prox_f(w - gamma L' u)
  # wbar = -gamma * L' * u + w
  copyto!(model.solver_state_internal.wbar, model.solver_state_internal.w)
  spock_mul!(model, model.solver_state_internal.wbar, true, model.solver_state_internal.u, -gamma, 1.)
  # wbar = prox_f(wbar)
  prox_f!(model, model.solver_state_internal.wbar, gamma)

  ### ubar = prox_h_conj(u + sigma L(2 wbar - w))
  # w_workspace = 2 * wbar - z
  copyto!(model.solver_state_internal.w_workspace, model.solver_state_internal.w)
  LA.BLAS.axpby!(2., model.solver_state_internal.wbar, -1., model.solver_state_internal.w_workspace)
  # ubar = sigma * L (w_workspace) + u
  copyto!(model.solver_state_internal.ubar, model.solver_state_internal.u)
  spock_mul!(model, model.solver_state_internal.ubar, false, model.solver_state_internal.w_workspace, sigma, 1.)
  # ubar = prox_h_conj(ubar)
  prox_h_conj!(model, model.solver_state_internal.ubar, sigma)

  for i in eachindex(model.solver_state_internal.w)
    model.solver_state_internal.rw[i] = model.solver_state_internal.w[i] - model.solver_state_internal.wbar[i]
  end
  for i in eachindex(model.solver_state_internal.u)
    model.solver_state_internal.ru[i] = model.solver_state_internal.u[i] - model.solver_state_internal.ubar[i]
  end

  return spock_norm(model, model.solver_state_internal.rw, model.solver_state_internal.ru, gamma, sigma)
end

function should_perform_k1(
  model :: MODEL_SP,
  rnorm :: TF,
  rtilde_norm :: TF,
  r_safe :: TF,
  c1 :: TF
) where {TF <: Real}

  return rnorm <= r_safe && rtilde_norm <= c1 * rnorm
end

function perform_k1!(
  model :: MODEL_SP,
  q :: TF,
  iter :: TI,
  rtilde_norm :: TF
) where {TF <: Real, TI <: Integer}

  copyto!(model.solver_state.z, model.solver_state_internal.w)
  copyto!(model.solver_state.v, model.solver_state_internal.u)

  r_safe = rtilde_norm + q^iter
  loop = false

  return r_safe, loop
end

function should_perform_k2(
  model :: MODEL_SP,
  rnorm :: TF,
  rtilde_norm :: TF,
  rho :: TF,
  sigma :: TF
) where {TF <: Real}

  return rho >= sigma * rnorm * rtilde_norm
end

function perform_k2!(
  model :: MODEL_SP,
  rtilde_norm :: TF,
  rho :: TF,
  lambda :: TF,
) where {TF <: Real}
  # Normalize rho
  rho /= rtilde_norm^2

  for i in eachindex(model.solver_state.z)
    model.solver_state.z[i] -= lambda * rho * model.solver_state_internal.rw[i]
  end
  for i in eachindex(model.solver_state.v)
    model.solver_state.v[i] -= lambda * rho * model.solver_state_internal.ru[i]
  end

  loop = false
  return loop
end

function perform_linesearch!(
  tau :: TF,
  beta :: TF
) where {TF <: Real}

  return tau * beta
end

function update_sy!(
  model :: MODEL_SP
)
  # TODO: Only for Broyden
  # nz = model.solver_state.nz
  # nv = model.solver_state.nv

  # for i = 1:nz
  #   model.solver_state_internal.sz[i] = model.solver_state_internal.w[i] - model.solver_state.z_old[i]
  #   model.solver_state_internal.yz[i] = model.solver_state_internal.rw[i] - model.solver_state_internal.rz_old[i]
  # end
  # for i = 1:nv
  #   model.solver_state_internal.sv[i] = model.solver_state_internal.u[i] - model.solver_state.v_old[i]
  #   model.solver_state_internal.yv[i] = model.solver_state_internal.ru[i] - model.solver_state_internal.rv_old[i]
  # end

  # Update the z / v / rz / rv -old variables for next iteration
  copyto!(model.solver_state.z_old, model.solver_state.z)
  copyto!(model.solver_state.v_old, model.solver_state.v)

  copyto!(model.solver_state_internal.rz_old, model.solver_state.rz)
  copyto!(model.solver_state_internal.rv_old, model.solver_state.rv)

end

function update_delta_r!(
  model :: MODEL_SP
)

  for i in eachindex(model.solver_state.delta_z)
    model.solver_state.delta_rz[i] = model.solver_state.rz[i] - model.solver_state_internal.rz_old[i]
  end
  for i in eachindex(model.solver_state.delta_v)
    model.solver_state.delta_rv[i] = model.solver_state.rv[i] - model.solver_state_internal.rv_old[i]
  end

end

function should_terminate!(
  model :: MODEL_SP,
  alpha1 :: TF,
  alpha2 :: TF,
  tol :: TF,
  verbose :: VERBOSE_LEVEL,
  backtrack_count :: TI,
) where {TF <: Real, TI <: Integer}

  for i in eachindex(model.solver_state.delta_z)
    model.solver_state.delta_z[i] = model.solver_state.z[i] - model.solver_state.z_old[i]
  end
  for i in eachindex(model.solver_state.delta_v)
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
    # copyto!(model.solver_state.xi, model.solver_state.xi_1)
    # spock_mul!(model, model.solver_state.xi, true, model.solver_state.xi_2, 1., 1.)

    # xi = LA.norm(model.solver_state.xi, Inf)
    xi = max(xi_1, xi_2)

    open("examples/output/xi_sp.dat", "a") do io
      writedlm(io, xi)
    end
    open("examples/output/xi1_sp.dat", "a") do io
      writedlm(io, xi_1)
    end
    open("examples/output/xi2_sp.dat", "a") do io
      writedlm(io, xi_2)
    end
    open("examples/output/xi_backtrack_count.dat", "a") do io
      writedlm(io, backtrack_count)
    end

    if xi <= tol
      res = true
    end
  # Only check third condition when first two have succeeded (avoid extra L' if possible)
  # elseif xi_1 <= tol * LA.norm(model.solver_state.z_old, Inf) && xi_2 <= tol * LA.norm(model.solver_state.v_old, Inf)
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

function update_delta_old!(
  model :: MODEL_SP
)

  copyto!(model.solver_state_internal.delta_z_old, model.solver_state.delta_z)
  copyto!(model.solver_state_internal.delta_v_old, model.solver_state.delta_v)

  copyto!(model.solver_state_internal.delta_rz_old, model.solver_state.delta_rz)
  copyto!(model.solver_state_internal.delta_rv_old, model.solver_state.delta_rv)

end

function run_sp!(
  model :: MODEL_SP;
  MAX_ITER_COUNT :: Int64 = 1000,
  tol :: Float64 = 1e-3,
  verbose :: VERBOSE_LEVEL = SILENT,
  gamma :: Union{Float64, Nothing} = nothing,
  sigma :: Union{Float64, Nothing} = nothing,
  lambda :: Float64 = 1.,
  c0 :: Float64 = 0.99,
  c1 :: Float64 = 0.99,
  q :: Float64 = 0.99,
  sigma_k2 :: Float64 = 0.1,
  beta :: Float64 = 0.5,
  MAX_BACKTRACK_COUNT :: Int64 = 8,
  lambda_sp :: Float64 = 1.
)

  if sigma === nothing || gamma === nothing
    sigma = 0.99 / sqrt(model.solver_state.L_norm)
    gamma = sigma
  end

  iter = 0

  # Initialize some SuperMann parameters
  rnorm = Inf; rtilde_norm = Inf
  r_safe = Inf
  eta = r_safe
  loop = true
  backtrack_count = 0
  tau = 1.
  broyden_k = 0

  while iter < MAX_ITER_COUNT
    update_zbar!(model, gamma)
    update_vbar!(model, sigma)

    rnorm = update_residual!(model, gamma, sigma)
    # When using Anderson, we need both Δp and Δr. 
    # Δp is computed at the end of previous iteration in the termination check.
    # Compute Δr now (only if Anderson selected).
    update_delta_r!(model)
    update_sy!(model)
    broyden_k = generate_qnewton_direction!(model, broyden_k, gamma, sigma)

    tau = 1.; backtrack_count = 0; loop = true

    # Blind update
    if should_perform_k0(rnorm, c0, eta)
      eta, loop = perform_k0!(model, rnorm, eta, loop)
      rtilde_norm = update_candidate_residual!(model, gamma, sigma)
    # Educated and GKM iterations with linesearch
    else
      while should_backtrack_loop(loop, backtrack_count, MAX_BACKTRACK_COUNT)
        generate_candidate_update!(model, tau)
        rtilde_norm = update_candidate_residual!(model, gamma, sigma)

        if should_perform_k1(model, rnorm, rtilde_norm, r_safe, c1)
          r_safe, loop = perform_k1!(model, q, iter, rtilde_norm)
          continue
        end

        for k = 1:model.solver_state.nz
          model.solver_state_internal.workspace_rho_z[k] = model.solver_state_internal.rw[k] - tau * model.solver_state_internal.dz[k]
        end
        for k = 1:model.solver_state.nv
          model.solver_state_internal.workspace_rho_v[k] = model.solver_state_internal.ru[k] - tau * model.solver_state_internal.dv[k]
        end
        rho = spock_dot(
          model,
          model.solver_state_internal.rw, 
          model.solver_state_internal.ru, 
          model.solver_state_internal.workspace_rho_z,
          model.solver_state_internal.workspace_rho_v,
          gamma,
          sigma
        )
        if should_perform_k2(model, rnorm, rtilde_norm, rho, sigma_k2)
          loop = perform_k2!(model, rtilde_norm, rho, lambda_sp)
          continue
        end
        perform_linesearch!(tau, beta)
        backtrack_count += 1
      end
      # Maximum backtrack_count attained, regular CP update as fallback
      if loop === true
        update_z!(model, lambda)
        update_v!(model, lambda)
      end
    end

    # Only for Anderson: Update old deltas
    update_delta_old!(model)

    # Check termination criterion
    if should_terminate!(
      model,
      gamma,
      sigma,
      tol,
      verbose,
      backtrack_count
    )
      break
    end

    iter += 1
  end


  println("Finished in $(iter) iterations.")
end