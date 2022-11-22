function restarted_broyden!(
  model :: MODEL_SP,
  k :: TI,
  alpha1 :: TF,
  alpha2 :: TF
) where {TI <: Integer, TF <: Real}

  MAX_K = 20
  theta_bar = 0.5

  nz = model.state.nz
  nv = model.state.nv

  # (Psz, Psv) = (L' * sv, L * sz)
  L!(model, model.solver_state_internal.sz, model.solver_state_internal.Psv)
  L_transpose!(model, model.solver_state_internal.Psz, model.solver_state_internal.sv)
  # (Psz, Psv) contains the vector P * s
  for i = 1:nz
    model.solver_state_internal.Psz[i] = model.solver_state_internal.sz[i] - alpha1 * model.solver_state_internal.Psz[i]
  end
  for i = 1:nv
    model.solver_state_internal.Psv[i] = -alpha2 * model.solver_state_internal.Psv[i] + model.solver_state_internal.sv[i]
  end

  # d = -rx
  for k = 1:nz
    model.solver_state_internal.dz[k] = - model.state.rz[k]
  end
  for k = 1:nv
    model.solver_state_internal.dv[k] = - model.state.rv[k]
  end
  
  # stilde = y
  copyto!(model.solver_state_internal.stildez, model.solver_state_internal.yz)
  copyto!(model.solver_state_internal.stildev, model.solver_state_internal.yv)
  
  # stilde and d update
  for i = 1 : k
      # Store s_i - stilde_i in the broyden_workspace vec
      for j = 1:nz
        model.solver_state_internal.broyden_workspace_z[j] = model.solver_state_internal.Sz_buf[(i - 1) * nz + j] - model.solver_state_internal.Stildez_buf[(i - 1) * nz + j]
      end
      for j = 1:nv
        model.solver_state_internal.broyden_workspace_v[j] = model.solver_state_internal.Sv_buf[(i - 1) * nv + j] - model.solver_state_internal.Stildev_buf[(i - 1) * nv + j]
      end

      dot_stilde = 0.
      for j = 1:nz
        dot_stilde += model.solver_state_internal.Psz_buf[(i - 1) * nz + j] * model.solver_state_internal.stildez[j]
      end
      for j = 1:nv
        dot_stilde += model.solver_state_internal.Psv_buf[(i - 1) * nv + j] * model.solver_state_internal.stildev[j]
      end

      dot_d = 0.
      for j = 1:nz
        dot_d += model.solver_state_internal.Psz_buf[(i - 1) * nz + j] * model.solver_state_internal.dz[j]
      end
      for j = 1:nv
        dot_d += model.solver_state_internal.Psv_buf[(i - 1) * nv + j] * model.solver_state_internal.dv[j]
      end

      normalization_factor = 0.
      for j = (i - 1) * nz + 1 : i * nz
        normalization_factor += model.solver_state_internal.Psz_buf[j] * model.solver_state_internal.Stildez_buf[j]
      end
      for j = (i - 1) * nv + 1 : i * nv
        normalization_factor += model.solver_state_internal.Psv_buf[j] * model.solver_state_internal.Stildev_buf[j]
      end

      # inds_z = (i - 1) * nz + 1 : i * nz
      # inds_v = (i - 1) * nv + 1 : i * nv

      # dot_stilde = LA.dot(model.solver_state_internal.Psz_buf[inds_z], model.solver_state_internal.stildez) +
      #   LA.dot(model.solver_state_internal.Psv_buf[inds_v], model.solver_state_internal.stildev)

      # dot_d = LA.dot(model.solver_state_internal.Psz_buf[inds_z], model.solver_state_internal.dz) +
      #       LA.dot(model.solver_state_internal.Psv_buf[inds_v], model.solver_state_internal.dv)

      # normalization_factor = LA.dot(model.solver_state_internal.Psz_buf[inds_z], model.solver_state_internal.Stildez_buf[inds_z]) +
      #       LA.dot(model.solver_state_internal.Psv_buf[inds_v], model.solver_state_internal.Stildev_buf[inds_v])

      dot_stilde /= normalization_factor; dot_d /= normalization_factor

      # stilde update
      for j = 1:nz
        model.solver_state_internal.stildez[j] += dot_stilde * model.solver_state_internal.broyden_workspace_z[j]
      end
      for j = 1:nv
        model.solver_state_internal.stildev[j] += dot_stilde * model.solver_state_internal.broyden_workspace_v[j]
      end
      # d update
      for j = 1:nz
        model.solver_state_internal.dz[j] += dot_d * model.solver_state_internal.broyden_workspace_z[j]
      end
      for j = 1:nv
        model.solver_state_internal.dv[j] += dot_d * model.solver_state_internal.broyden_workspace_v[j]
      end
  end

  # gamma = LA.dot(stilde, Ps) / LA.dot(s, Ps)
  gamma :: TF = (
    LA.dot(model.solver_state_internal.stildez, model.solver_state_internal.Psz) + 
    LA.dot(model.solver_state_internal.stildev, model.solver_state_internal.Psv)
  ) / (
    LA.dot(model.solver_state_internal.sz, model.solver_state_internal.Psz) +
    LA.dot(model.solver_state_internal.sv, model.solver_state_internal.Psv)
  )
  theta  :: TF = 0.
  if abs(gamma) >= theta_bar
      theta = 1.
  elseif gamma === 0 # In Julia, sign(0) = 0, whereas we define it as sign(0) = 1; handle separately
      theta = (1. - theta_bar)
  else
      theta = (1. - sign(gamma) * theta_bar) / (1. - gamma)
  end


  # stilde = (1 - theta) * s + theta * stilde
  for i = 1:nz
    model.solver_state_internal.stildez[i] = (1 - theta) * model.solver_state_internal.sz[i] + theta * model.solver_state_internal.stildez[i]
  end
  for i = 1:nv
    model.solver_state_internal.stildev[i] = (1 - theta) * model.solver_state_internal.sv[i] + theta * model.solver_state_internal.stildev[i]
  end
  # d += LA.dot(Ps, d) / LA.dot(Ps, stilde) * (s - stilde)
  dot_d = (
    LA.dot(model.solver_state_internal.Psz, model.solver_state_internal.dz) +
    LA.dot(model.solver_state_internal.Psv, model.solver_state_internal.dv)
  ) / (
    LA.dot(model.solver_state_internal.Psz, model.solver_state_internal.stildez) + 
    LA.dot(model.solver_state_internal.Psv, model.solver_state_internal.stildev)
  )
  for i = 1:nz
    model.solver_state_internal.dz[i] += dot_d * (model.solver_state_internal.sz[i] - model.solver_state_internal.stildez[i])
  end
  for i = 1:nv
    model.solver_state_internal.dv[i] += dot_d * (model.solver_state_internal.sv[i] - model.solver_state_internal.stildev[i])
  end

  # Update buffers
  if k < MAX_K
    k += 1
    # model.solver_state_internal.Sz_buf[(k - 1) * nz + 1 : k * nz] = model.solver_state_internal.sz
    # model.solver_state_internal.Sv_buf[(k - 1) * nv + 1 : k * nv] = model.solver_state_internal.sv

    copyto!(model.solver_state_internal.Sz_buf, (k - 1) * nz + 1, model.solver_state_internal.sz, 1, nz)
    copyto!(model.solver_state_internal.Sv_buf, (k-1) * nv + 1, model.solver_state_internal.sv, 1, nv)

    # model.solver_state_internal.Stildez_buf[(k-1) * nz + 1 : k * nz] = model.solver_state_internal.stildez
    # model.solver_state_internal.Stildev_buf[(k-1) * nv + 1 : k * nv] = model.solver_state_internal.stildev

    copyto!(model.solver_state_internal.Stildez_buf, (k-1) * nz + 1, model.solver_state_internal.stildez, 1, nz)
    copyto!(model.solver_state_internal.Stildev_buf, (k-1) * nv + 1, model.solver_state_internal.stildev, 1, nv)

    # model.solver_state_internal.Psz_buf[(k-1) * nz + 1 : k * nz] = model.solver_state_internal.Psz
    # model.solver_state_internal.Psv_buf[(k-1) * nv + 1 : k * nv] = model.solver_state_internal.Psv

    copyto!(model.solver_state_internal.Psz_buf, (k-1) * nz + 1, model.solver_state_internal.Psz, 1, nz)
    copyto!(model.solver_state_internal.Psv_buf, (k-1) * nv + 1, model.solver_state_internal.Psv, 1, nv)
  else
    k = 0
  end

  return k
end