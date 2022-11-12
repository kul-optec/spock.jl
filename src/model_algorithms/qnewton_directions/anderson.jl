function anderson!(
  model :: MODEL_SP,
)

  ANDERSON_BUFFER_SIZE = 3

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

  # temp = vcat(
  #   model.solver_state.rz,
  #   model.solver_state.rv
  # )
  gamma = model.solver_state_internal.MR \ model.solver_state_internal.aa_wsp

  # gamma = model.solver_state_internal.MP * gamma
  LA.mul!(model.solver_state_internal.aa_wsp, model.solver_state_internal.MP, gamma)
  for i = 1:model.solver_state.nz
    model.solver_state_internal.dz[i] = -model.solver_state.rz[i] - model.solver_state_internal.aa_wsp[i]
  end
  for i = 1:model.solver_state.nv
    model.solver_state_internal.dv[i] = -model.solver_state.rv[i] - model.solver_state_internal.aa_wsp[model.solver_state.nz + i]
  end
end