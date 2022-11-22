function anderson!(
  model :: MODEL_SP,
  iter :: TI
) where {TI <: Integer}

  ANDERSON_BUFFER_SIZE = 3
  gamma = model.qn_state.aa_gamma

  # model.qn_state.MP[1:model.state.nz + model.state.nv, 1 : ANDERSON_BUFFER_SIZE] = circshift(model.qn_state.MP, (0, 1))
  # model.qn_state.MR[1:model.state.nz + model.state.nv, 1 : ANDERSON_BUFFER_SIZE] = circshift(model.qn_state.MR, (0, 1))


  n_rows = model.state.nz + model.state.nv
  for j = ANDERSON_BUFFER_SIZE - 1 : -1 : 1
    for i = 1:n_rows
      model.qn_state.MP[i, j + 1] = model.qn_state.MP[i, j]
    end
  end
  n_rows = model.state.nz + model.state.nv
  for j = ANDERSON_BUFFER_SIZE - 1 : -1 : 1
    for i = 1:n_rows
      model.qn_state.MR[i, j + 1] = model.qn_state.MR[i, j]
    end
  end
  
  for i = 1:model.state.nz
    model.qn_state.MR[i, 1] = model.state.Δrz[i]
    model.qn_state.MP[i, 1] = model.state.Δz[i] - model.qn_state.MR[i, 1]
  end 
  for i = 1:model.state.nv
    model.qn_state.MR[model.state.nz + i, 1] = model.state.Δrv[i]
    model.qn_state.MP[model.state.nz + i, 1] = model.state.Δv[i] - model.qn_state.MR[model.state.nz + i, 1]
  end

  for i = 1:model.state.nz
    model.qn_state.aa_wsp[i] = model.state.rz[i]
  end
  for i = 1:model.state.nv
    model.qn_state.aa_wsp[model.state.nz + i] = model.state.rv[i]
  end

  ##### QR update to solve least squares problem

  ### Shift Q and R
  # Shift R diagonally by one element
  for j = ANDERSON_BUFFER_SIZE - 1: -1:1
    for i = ANDERSON_BUFFER_SIZE -1 : -1 :1
      model.qn_state.aa_R[i+1, j+1] = model.qn_state.aa_R[i, j]
    end
  end
  # shift Q to the right by one element
  for j = ANDERSON_BUFFER_SIZE - 1 : -1 : 1
    for i = 1:n_rows
      model.qn_state.aa_Q[i, j + 1] = model.qn_state.aa_Q[i, j]
    end
  end

  # if iter == 2
  #   println(model.qn_state.aa_Q[1:10, 1:3])
  # end

  # Store the new first column of MR as the first column of Q.
  for i = 1:n_rows
    model.qn_state.aa_Q[i, 1] = model.qn_state.MR[i, 1]
  end

  # Modified GS (orthogonalize against all columns j > 2 of Q, modified for numerical stability)
  if iter == 1
    copyto!(model.qn_state.aa_Q, 1, model.qn_state.MR, 1, n_rows)
    r = LA.norm(view(model.qn_state.aa_Q, :, 1))
    # r = norm_of_column(model.qn_state.aa_Q, 1, n_rows)
    for i = 1:n_rows
      model.qn_state.aa_Q[i, 1] /= r
    end
    model.qn_state.aa_R[1, 1] = r
  else
    for j = 2:min(ANDERSON_BUFFER_SIZE, iter)
      # r = LA.dot(model.qn_state.aa_Q[:, j], model.qn_state.aa_Q[:, 1]) / LA.norm(model.qn_state.aa_Q[:, j])
      r = 0.
      for i = 1:n_rows
        r += model.qn_state.aa_Q[i, j] * model.qn_state.aa_Q[i, 1]
      end
      r /= LA.norm(view(model.qn_state.aa_Q, :, j))
      # r /= norm_of_column(model.qn_state.aa_Q, j, n_rows)
      # Update R
      model.qn_state.aa_R[1, j] = r
      for i = 1:n_rows
        # Update the first column of Q
        model.qn_state.aa_Q[i, 1] = model.qn_state.aa_Q[i, 1] - r * model.qn_state.aa_Q[i, j]
      end
    end
    # Normalize
    r = LA.norm(view(model.qn_state.aa_Q, :, 1))
    # r = norm_of_column(model.qn_state.aa_Q, 1, n_rows)
    for i = 1:n_rows
      model.qn_state.aa_Q[i, 1] /= r
    end
    model.qn_state.aa_R[1, 1] = r
  end

  # if iter == 2
  #   println(model.qn_state.MR[1:10, 1:3])
  #   println(model.qn_state.aa_Q[1:10, 1:3])
  #   println(model.qn_state.aa_R[1:3, 1:3])
  #   t =  model.qn_state.aa_Q * model.qn_state.aa_R
  #   println(t[1:10, 1:3] - model.qn_state.MR[1:10, 1:3])
  # end
  # println(LA.norm(model.qn_state.MR - model.qn_state.aa_Q * model.qn_state.aa_R))

  # gamma = R^{-1} * Q' * aa_wsp
  LA.mul!(gamma, model.qn_state.aa_Q', model.qn_state.aa_wsp)
  # gamma = model.qn_state.aa_R \ gamma
  LA.ldiv!(model.qn_state.aa_R, gamma)

  # gamma = model.qn_state.MP * gamma
  LA.mul!(model.qn_state.aa_wsp, model.qn_state.MP, gamma)
  for i = 1:model.state.nz
    model.solver_state_internal.dz[i] = -model.state.rz[i] - model.qn_state.aa_wsp[i]
  end
  for i = 1:model.state.nv
    model.solver_state_internal.dv[i] = -model.state.rv[i] - model.qn_state.aa_wsp[model.state.nz + i]
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