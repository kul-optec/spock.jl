function anderson!(
  model :: MODEL_SP,
)

  ANDERSON_BUFFER_SIZE = 10

  model.solver_state_internal.MP[1:model.solver_state.nz + model.solver_state.nv, 1 : ANDERSON_BUFFER_SIZE] = circshift(model.solver_state_internal.MP, (0, 1))
  model.solver_state_internal.MR[1:model.solver_state.nz + model.solver_state.nv, 1 : ANDERSON_BUFFER_SIZE] = circshift(model.solver_state_internal.MR, (0, 1))
  
  for i = 1:model.solver_state.nz
    model.solver_state_internal.MP[i, 1] = model.solver_state.delta_z[i]
    model.solver_state_internal.MR[i, 1] = model.solver_state.delta_rz[i]
  end 
  for i = 1:model.solver_state.nv
    model.solver_state_internal.MP[model.solver_state.nz + i, 1] = model.solver_state.delta_v[i]
    model.solver_state_internal.MR[model.solver_state.nz + i, 1] = model.solver_state.delta_rv[i]
  end

  gamma = model.solver_state_internal.MR \ vcat(
    model.solver_state.rz,
    model.solver_state.rv
  )

  gamma = (model.solver_state_internal.MP - model.solver_state_internal.MR) * gamma
  for i = 1:model.solver_state.nz
    model.solver_state_internal.dz[i] = -model.solver_state.rz[i] - gamma[i]
  end
  for i = 1:model.solver_state.nv
    model.solver_state_internal.dv[i] = -model.solver_state.rv[i] - gamma[model.solver_state.nz + i]
  end
end