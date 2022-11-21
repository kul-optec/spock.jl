function anderson!(
  model :: MODEL_SP,
  iter :: TI
) where {TI <: Integer}

  ANDERSON_BUFFER_SIZE = 3
  gamma = model.solver_state_internal.aa_gamma

  # model.solver_state_internal.MP[1:model.solver_state.nz + model.solver_state.nv, 1 : ANDERSON_BUFFER_SIZE] = circshift(model.solver_state_internal.MP, (0, 1))
  # model.solver_state_internal.MR[1:model.solver_state.nz + model.solver_state.nv, 1 : ANDERSON_BUFFER_SIZE] = circshift(model.solver_state_internal.MR, (0, 1))


  n_rows = model.solver_state.nz + model.solver_state.nv
  for j = ANDERSON_BUFFER_SIZE - 1 : -1 : 1
    for i = 1:n_rows
      model.solver_state_internal.MP[i, j + 1] = model.solver_state_internal.MP[i, j]
    end
  end
  n_rows = model.solver_state.nz + model.solver_state.nv
  for j = ANDERSON_BUFFER_SIZE - 1 : -1 : 1
    for i = 1:n_rows
      model.solver_state_internal.MR[i, j + 1] = model.solver_state_internal.MR[i, j]
    end
  end
  
  for i = 1:model.solver_state.nz
    model.solver_state_internal.MR[i, 1] = model.solver_state.delta_rz[i]
    model.solver_state_internal.MP[i, 1] = model.solver_state.delta_z[i] - model.solver_state_internal.MR[i, 1]
  end 
  for i = 1:model.solver_state.nv
    model.solver_state_internal.MR[model.solver_state.nz + i, 1] = model.solver_state.delta_rv[i]
    model.solver_state_internal.MP[model.solver_state.nz + i, 1] = model.solver_state.delta_v[i] - model.solver_state_internal.MR[model.solver_state.nz + i, 1]
  end

  for i = 1:model.solver_state.nz
    model.solver_state_internal.aa_wsp[i] = model.solver_state.rz[i]
  end
  for i = 1:model.solver_state.nv
    model.solver_state_internal.aa_wsp[model.solver_state.nz + i] = model.solver_state.rv[i]
  end

  ##### QR update to solve least squares problem

  ### Shift Q and R
  # Shift R diagonally by one element
  for j = ANDERSON_BUFFER_SIZE - 1: -1:1
    for i = ANDERSON_BUFFER_SIZE -1 : -1 :1
      model.solver_state_internal.aa_R[i+1, j+1] = model.solver_state_internal.aa_R[i, j]
    end
  end
  # shift Q to the right by one element
  for j = ANDERSON_BUFFER_SIZE - 1 : -1 : 1
    for i = 1:n_rows
      model.solver_state_internal.aa_Q[i, j + 1] = model.solver_state_internal.aa_Q[i, j]
    end
  end

  # if iter == 2
  #   println(model.solver_state_internal.aa_Q[1:10, 1:3])
  # end

  # Store the new first column of MR as the first column of Q.
  for i = 1:n_rows
    model.solver_state_internal.aa_Q[i, 1] = model.solver_state_internal.MR[i, 1]
  end

  # Modified GS (orthogonalize against all columns j > 2 of Q, modified for numerical stability)
  if iter == 1
    copyto!(model.solver_state_internal.aa_Q, 1, model.solver_state_internal.MR, 1, n_rows)
    r = LA.norm(view(model.solver_state_internal.aa_Q, :, 1))
    # r = norm_of_column(model.solver_state_internal.aa_Q, 1, n_rows)
    for i = 1:n_rows
      model.solver_state_internal.aa_Q[i, 1] /= r
    end
    model.solver_state_internal.aa_R[1, 1] = r
  else
    for j = 2:min(ANDERSON_BUFFER_SIZE, iter)
      # r = LA.dot(model.solver_state_internal.aa_Q[:, j], model.solver_state_internal.aa_Q[:, 1]) / LA.norm(model.solver_state_internal.aa_Q[:, j])
      r = 0.
      for i = 1:n_rows
        r += model.solver_state_internal.aa_Q[i, j] * model.solver_state_internal.aa_Q[i, 1]
      end
      r /= LA.norm(view(model.solver_state_internal.aa_Q, :, j))
      # r /= norm_of_column(model.solver_state_internal.aa_Q, j, n_rows)
      # Update R
      model.solver_state_internal.aa_R[1, j] = r
      for i = 1:n_rows
        # Update the first column of Q
        model.solver_state_internal.aa_Q[i, 1] = model.solver_state_internal.aa_Q[i, 1] - r * model.solver_state_internal.aa_Q[i, j]
      end
    end
    # Normalize
    r = LA.norm(view(model.solver_state_internal.aa_Q, :, 1))
    # r = norm_of_column(model.solver_state_internal.aa_Q, 1, n_rows)
    for i = 1:n_rows
      model.solver_state_internal.aa_Q[i, 1] /= r
    end
    model.solver_state_internal.aa_R[1, 1] = r
  end

  # if iter == 2
  #   println(model.solver_state_internal.MR[1:10, 1:3])
  #   println(model.solver_state_internal.aa_Q[1:10, 1:3])
  #   println(model.solver_state_internal.aa_R[1:3, 1:3])
  #   t =  model.solver_state_internal.aa_Q * model.solver_state_internal.aa_R
  #   println(t[1:10, 1:3] - model.solver_state_internal.MR[1:10, 1:3])
  # end
  # println(LA.norm(model.solver_state_internal.MR - model.solver_state_internal.aa_Q * model.solver_state_internal.aa_R))

  # gamma = R^{-1} * Q' * aa_wsp
  LA.mul!(gamma, model.solver_state_internal.aa_Q', model.solver_state_internal.aa_wsp)
  # gamma = model.solver_state_internal.aa_R \ gamma
  LA.ldiv!(model.solver_state_internal.aa_R, gamma)

  # gamma = model.solver_state_internal.MP * gamma
  LA.mul!(model.solver_state_internal.aa_wsp, model.solver_state_internal.MP, gamma)
  for i = 1:model.solver_state.nz
    model.solver_state_internal.dz[i] = -model.solver_state.rz[i] - model.solver_state_internal.aa_wsp[i]
  end
  for i = 1:model.solver_state.nv
    model.solver_state_internal.dv[i] = -model.solver_state.rv[i] - model.solver_state_internal.aa_wsp[model.solver_state.nz + i]
  end
end

function norm_of_column(
  Q :: Matrix{TF},
  j :: TI,
  n_rows :: TI
) where {TI <: Integer, TF <: Real}

  res = 0.
  for i = 1:n_rows
    res += Q[i, j]^2
  end

  return sqrt(res)
end