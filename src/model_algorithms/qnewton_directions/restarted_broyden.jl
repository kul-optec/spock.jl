function restarted_broyden!(
  model :: SPOCK_RB,
  k :: TI,
  alpha1 :: TF,
  alpha2 :: TF
) where {TI <: Integer, TF <: Real}

  MAX_K = 20
  theta_bar = 0.5

  nz = model.state.nz
  nv = model.state.nv

  # (Psz, Psv) = (L' * sv, L * sz)
  L!(model, model.qn_state.sz, model.qn_state.Psv)
  L_transpose!(model, model.qn_state.Psz, model.qn_state.sv)
  # (Psz, Psv) contains the vector P * s
  for i = 1:nz
    model.qn_state.Psz[i] = model.qn_state.sz[i] - alpha1 * model.qn_state.Psz[i]
  end
  for i = 1:nv
    model.qn_state.Psv[i] = -alpha2 * model.qn_state.Psv[i] + model.qn_state.sv[i]
  end

  # d = -rx
  for k = 1:nz
    model.solver_state_internal.dz[k] = - model.state.rz[k]
  end
  for k = 1:nv
    model.solver_state_internal.dv[k] = - model.state.rv[k]
  end
  
  # stilde = y
  copyto!(model.qn_state.stildez, model.qn_state.yz)
  copyto!(model.qn_state.stildev, model.qn_state.yv)
  
  # stilde and d update
  for i = 1 : k
      # Store s_i - stilde_i in the broyden_wsp vec
      for j = 1:nz
        model.qn_state.broyden_wsp_z[j] = model.qn_state.Sz_buf[(i - 1) * nz + j] - model.qn_state.Stildez_buf[(i - 1) * nz + j]
      end
      for j = 1:nv
        model.qn_state.broyden_wsp_v[j] = model.qn_state.Sv_buf[(i - 1) * nv + j] - model.qn_state.Stildev_buf[(i - 1) * nv + j]
      end

      dot_stilde = 0.
      for j = 1:nz
        dot_stilde += model.qn_state.Psz_buf[(i - 1) * nz + j] * model.qn_state.stildez[j]
      end
      for j = 1:nv
        dot_stilde += model.qn_state.Psv_buf[(i - 1) * nv + j] * model.qn_state.stildev[j]
      end

      dot_d = 0.
      for j = 1:nz
        dot_d += model.qn_state.Psz_buf[(i - 1) * nz + j] * model.solver_state_internal.dz[j]
      end
      for j = 1:nv
        dot_d += model.qn_state.Psv_buf[(i - 1) * nv + j] * model.solver_state_internal.dv[j]
      end

      normalization_factor = 0.
      for j = (i - 1) * nz + 1 : i * nz
        normalization_factor += model.qn_state.Psz_buf[j] * model.qn_state.Stildez_buf[j]
      end
      for j = (i - 1) * nv + 1 : i * nv
        normalization_factor += model.qn_state.Psv_buf[j] * model.qn_state.Stildev_buf[j]
      end

      # inds_z = (i - 1) * nz + 1 : i * nz
      # inds_v = (i - 1) * nv + 1 : i * nv

      # dot_stilde = LA.dot(model.qn_state.Psz_buf[inds_z], model.qn_state.stildez) +
      #   LA.dot(model.qn_state.Psv_buf[inds_v], model.qn_state.stildev)

      # dot_d = LA.dot(model.qn_state.Psz_buf[inds_z], model.solver_state_internal.dz) +
      #       LA.dot(model.qn_state.Psv_buf[inds_v], model.solver_state_internal.dv)

      # normalization_factor = LA.dot(model.qn_state.Psz_buf[inds_z], model.qn_state.Stildez_buf[inds_z]) +
      #       LA.dot(model.qn_state.Psv_buf[inds_v], model.qn_state.Stildev_buf[inds_v])

      dot_stilde /= normalization_factor; dot_d /= normalization_factor

      # stilde update
      for j = 1:nz
        model.qn_state.stildez[j] += dot_stilde * model.qn_state.broyden_wsp_z[j]
      end
      for j = 1:nv
        model.qn_state.stildev[j] += dot_stilde * model.qn_state.broyden_wsp_v[j]
      end
      # d update
      for j = 1:nz
        model.solver_state_internal.dz[j] += dot_d * model.qn_state.broyden_wsp_z[j]
      end
      for j = 1:nv
        model.solver_state_internal.dv[j] += dot_d * model.qn_state.broyden_wsp_v[j]
      end
  end

  # gamma = LA.dot(stilde, Ps) / LA.dot(s, Ps)
  gamma :: TF = (
    LA.dot(model.qn_state.stildez, model.qn_state.Psz) + 
    LA.dot(model.qn_state.stildev, model.qn_state.Psv)
  ) / (
    LA.dot(model.qn_state.sz, model.qn_state.Psz) +
    LA.dot(model.qn_state.sv, model.qn_state.Psv)
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
    model.qn_state.stildez[i] = (1 - theta) * model.qn_state.sz[i] + theta * model.qn_state.stildez[i]
  end
  for i = 1:nv
    model.qn_state.stildev[i] = (1 - theta) * model.qn_state.sv[i] + theta * model.qn_state.stildev[i]
  end
  # d += LA.dot(Ps, d) / LA.dot(Ps, stilde) * (s - stilde)
  dot_d = (
    LA.dot(model.qn_state.Psz, model.solver_state_internal.dz) +
    LA.dot(model.qn_state.Psv, model.solver_state_internal.dv)
  ) / (
    LA.dot(model.qn_state.Psz, model.qn_state.stildez) + 
    LA.dot(model.qn_state.Psv, model.qn_state.stildev)
  )
  for i = 1:nz
    model.solver_state_internal.dz[i] += dot_d * (model.qn_state.sz[i] - model.qn_state.stildez[i])
  end
  for i = 1:nv
    model.solver_state_internal.dv[i] += dot_d * (model.qn_state.sv[i] - model.qn_state.stildev[i])
  end

  # Update buffers
  if k < MAX_K
    k += 1
    # model.qn_state.Sz_buf[(k - 1) * nz + 1 : k * nz] = model.qn_state.sz
    # model.qn_state.Sv_buf[(k - 1) * nv + 1 : k * nv] = model.qn_state.sv

    copyto!(model.qn_state.Sz_buf, (k - 1) * nz + 1, model.qn_state.sz, 1, nz)
    copyto!(model.qn_state.Sv_buf, (k-1) * nv + 1, model.qn_state.sv, 1, nv)

    # model.qn_state.Stildez_buf[(k-1) * nz + 1 : k * nz] = model.qn_state.stildez
    # model.qn_state.Stildev_buf[(k-1) * nv + 1 : k * nv] = model.qn_state.stildev

    copyto!(model.qn_state.Stildez_buf, (k-1) * nz + 1, model.qn_state.stildez, 1, nz)
    copyto!(model.qn_state.Stildev_buf, (k-1) * nv + 1, model.qn_state.stildev, 1, nv)

    # model.qn_state.Psz_buf[(k-1) * nz + 1 : k * nz] = model.qn_state.Psz
    # model.qn_state.Psv_buf[(k-1) * nv + 1 : k * nv] = model.qn_state.Psv

    copyto!(model.qn_state.Psz_buf, (k-1) * nz + 1, model.qn_state.Psz, 1, nz)
    copyto!(model.qn_state.Psv_buf, (k-1) * nv + 1, model.qn_state.Psv, 1, nv)
  else
    k = 0
  end

  return k
end