###
# Model
###

abstract type CUSTOM_SOLVER_MODEL end

# JuMP defines the Model type, that would be confusing to us
const REFERENCE_MODEL = Model

# Type union for all available solver models
const SOLVER_MODEL = Union{REFERENCE_MODEL, CUSTOM_SOLVER_MODEL}

"""
The idea is that only the internal state has to differ between models.
"""
# used to store e.g. the variables of the CP algorithm
abstract type SOLVER_STATE end 
# used to store some specific variables for this model
abstract type SOLVER_STATE_INTERNAL end
# Part of the problem definition that remains accessible during the solve step
abstract type PROBLEM_DEFINITION end

struct GENERIC_SOLVER_STATE{TF, TI} <: SOLVER_STATE
  nz :: TI
  nv :: TI
  L_norm :: TF
  z :: Vector{TF}
  v :: Vector{TF}
  zbar :: Vector{TF}
  vbar :: Vector{TF}
  z_workspace :: Vector{TF}
  rz :: Vector{TF}
  rv :: Vector{TF}
  xi_1 :: Vector{TF}
  xi_2 :: Vector{TF}
  xi :: Vector{TF}
  delta_z :: Vector{TF}
  delta_v :: Vector{TF}
  z_old :: Vector{TF}
  v_old :: Vector{TF}
  delta_rz :: Vector{TF}
  delta_rv :: Vector{TF}
  res_0 :: Vector{TF}
end

struct GENERIC_PROBLEM_DEFINITIONV1{TF <: Real, TI <: Integer} <: PROBLEM_DEFINITION
  x0 :: Vector{TF}
  nx :: TI
  nu :: TI
  scen_tree :: ScenarioTreeV1{TI}
  rms :: Vector{RiskMeasureV1}
  cost:: Cost
  dynamics :: Dynamics{TF}
end

struct GENERIC_PROBLEM_DEFINITIONV2{TF <: Real, TI <: Integer} <: PROBLEM_DEFINITION
  x0 :: Vector{TF}
  nx :: TI
  nu :: TI
  scen_tree :: ScenarioTreeV2{TI}
  rms :: Vector{RiskMeasureV2}
  cost:: Cost
  dynamics :: Dynamics{TF}
end

"""
Supported options to construct various solvers
"""

@enum DynamicsOptions begin
  DYNAMICSL = 1
  L_IMPLICIT = 2
end

@enum AlgorithmOptions begin
  CP = 1 # Plain Chambolle-Pock
  SP = 2 # Chambolle-Pock + SuperMann
end

struct SolverOptions
  dynamics :: DynamicsOptions
  algorithm :: AlgorithmOptions
end

##########
# CP Model with the dynamics in L
##########
struct CP_DYNAMICSL_STATE_INTERNAL{TM, TI, TF} <: SOLVER_STATE_INTERNAL
  L :: TM
  x_inds :: Vector{TI}
  u_inds :: Vector{TI}
  s_inds :: Vector{TI}
  y_inds :: Vector{TI}
  v_projection_arg :: AbstractArray{TF, 1}
  inds_L_risk_a :: Vector{Union{UnitRange{TI}, TI}}
  inds_L_risk_b :: Vector{Union{UnitRange{TI}, TI}}
  inds_L_risk_c :: Vector{Union{UnitRange{TI}, TI}}
  inds_L_cost :: Vector{Union{UnitRange{TI}, TI}}
  inds_L_dynamics :: UnitRange{TI}
  v_bisection_workspace :: AbstractArray{TF, 1}
end

struct MODEL_CP_DYNAMICSL{TM, TI, TF} <: CUSTOM_SOLVER_MODEL
  solver_state :: GENERIC_SOLVER_STATE{TF, TI}
  solver_state_internal :: CP_DYNAMICSL_STATE_INTERNAL{TM, TI, TF}
  problem_definition :: GENERIC_PROBLEM_DEFINITIONV1{TF, TI}
end

##########
# SP Model with the dynamics in L
##########
struct SP_DYNAMICSL_STATE_INTERNAL{TM, TI, TF} <: SOLVER_STATE_INTERNAL
  L :: TM
  x_inds :: Vector{TI}
  u_inds :: Vector{TI}
  s_inds :: Vector{TI}
  y_inds :: Vector{TI}
  v_projection_arg :: AbstractArray{TF, 1}
  inds_L_risk_a :: Vector{Union{UnitRange{TI}, TI}}
  inds_L_risk_b :: Vector{Union{UnitRange{TI}, TI}}
  inds_L_risk_c :: Vector{Union{UnitRange{TI}, TI}}
  inds_L_cost :: Vector{Union{UnitRange{TI}, TI}}
  inds_L_dynamics :: UnitRange{TI}
  v_bisection_workspace :: AbstractArray{TF, 1}
  # For SuperMann
  dz :: AbstractArray{TF, 1}
  dv :: AbstractArray{TF, 1}
  w :: AbstractArray{TF, 1}
  u :: AbstractArray{TF, 1}
  wbar :: AbstractArray{TF, 1}
  ubar :: AbstractArray{TF, 1}
  rw :: AbstractArray{TF, 1}
  ru :: AbstractArray{TF, 1}
  w_workspace :: AbstractArray{TF, 1}
  # Restarted Broyden
  sz :: AbstractArray{TF, 1}
  sv :: AbstractArray{TF, 1}
  stildez :: AbstractArray{TF, 1}
  stildev :: AbstractArray{TF, 1}
  yz :: AbstractArray{TF, 1}
  yv :: AbstractArray{TF, 1}
  Psz :: AbstractArray{TF, 1}
  Psv :: AbstractArray{TF, 1}
  Sz_buf :: AbstractArray{TF, 1}
  Sv_buf :: AbstractArray{TF, 1}
  Stildez_buf :: AbstractArray{TF, 1}
  Stildev_buf :: AbstractArray{TF, 1}
  Psz_buf :: AbstractArray{TF, 1}
  Psv_buf :: AbstractArray{TF, 1}
  rz_old :: AbstractArray{TF, 1}
  rv_old :: AbstractArray{TF, 1}
end

struct MODEL_SP_DYNAMICSL{TM, TI, TF} <: CUSTOM_SOLVER_MODEL
  solver_state :: GENERIC_SOLVER_STATE{TF, TI}
  solver_state_internal :: SP_DYNAMICSL_STATE_INTERNAL{TM, TI, TF}
  problem_definition :: GENERIC_PROBLEM_DEFINITIONV1{TF, TI}
end

########
# CP model with implicit L
########

struct CP_IMPLICITL_STATE_INTERNAL{TI, TF, TM} <: SOLVER_STATE_INTERNAL
  mul_x_workspace :: Vector{TF}
  mul_x_workspace2 :: Vector{TF}
  mul_u_workspace :: Vector{TF}
  x_inds :: Vector{TI}
  u_inds :: Vector{TI}
  s_inds :: Vector{TI}
  tau_inds :: Vector{TI}
  y_inds :: Vector{TI}
  sqrtQ :: Vector{TM}
  sqrtR :: Vector{TM}
  sqrtQN :: Vector{TM}
  spock_mul_buffer_z :: Vector{TF}
  spock_mul_buffer_v :: Vector{TF}
  M :: Vector{TM} # TODO: Unnecessary?
  ls_matrix :: Vector{TM}
  ls_b :: Vector{TF}
  ls_b2 :: Vector{TF}
  # Ricatti
  P :: Vector{TM}
  K :: Vector{TM}
  R_chol :: Vector{LA.Cholesky{TF, TM}}
  ABK :: Vector{TM}
  ric_q :: Vector{TF}
  ric_d :: Vector{TF}
  sum_for_d :: Vector{TF}
  # prox_g*
  prox_v_workspace :: Vector{TF}
  v5_inds :: Vector{TI}
  v6_inds :: Vector{TI}
  v12_inds :: Vector{TI}
  v13_inds :: Vector{TI}
  # projection onto S3
  v2_offset :: TI
  v3_offset :: TI
  v4_offset :: TI
  v5_offset :: TI
  v6_offset :: TI
  v11_offset :: TI
  v12_offset :: TI
  v13_offset :: TI
  proj_leafs_workspace :: Vector{TF}
  proj_nleafs_workspace :: Vector{TF}
end

struct MODEL_CP_IMPLICITL{TI, TF, TM} <: CUSTOM_SOLVER_MODEL
  solver_state :: GENERIC_SOLVER_STATE{TF, TI}
  solver_state_internal :: CP_IMPLICITL_STATE_INTERNAL{TI, TF, TM}
  problem_definition :: GENERIC_PROBLEM_DEFINITIONV2{TF, TI}
end

########
# SP model with implicit L
########

struct SP_IMPLICITL_STATE_INTERNAL{TI, TF, TM} <: SOLVER_STATE_INTERNAL
  mul_x_workspace :: Vector{TF}
  mul_x_workspace2 :: Vector{TF}
  mul_u_workspace :: Vector{TF}
  x_inds :: Vector{TI}
  u_inds :: Vector{TI}
  s_inds :: Vector{TI}
  tau_inds :: Vector{TI}
  y_inds :: Vector{TI}
  sqrtQ :: Vector{TM}
  sqrtR :: Vector{TM}
  sqrtQN :: Vector{TM}
  spock_mul_buffer_z :: Vector{TF}
  spock_mul_buffer_v :: Vector{TF}
  M :: Vector{TM} # TODO: Unnecessary?
  ls_matrix :: Vector{TM}
  ls_b :: Vector{TF}
  ls_b2 :: Vector{TF}
  # Ricatti
  P :: Vector{TM}
  K :: Vector{TM}
  R_chol :: Vector{LA.Cholesky{TF, TM}}
  ABK :: Vector{TM}
  ric_q :: Vector{TF}
  ric_d :: Vector{TF}
  sum_for_d :: Vector{TF}
  # prox_g*
  prox_v_workspace :: Vector{TF}
  v5_inds :: Vector{TI}
  v6_inds :: Vector{TI}
  v12_inds :: Vector{TI}
  v13_inds :: Vector{TI}
  # projection onto S3
  v2_offset :: TI
  v3_offset :: TI
  v4_offset :: TI
  v5_offset :: TI
  v6_offset :: TI
  v11_offset :: TI
  v12_offset :: TI
  v13_offset :: TI
  proj_leafs_workspace :: Vector{TF}
  proj_nleafs_workspace :: Vector{TF}  
  # For SuperMann
  workspace_rho_z :: Vector{TF}
  workspace_rho_v :: Vector{TF}
  dz :: Vector{TF}
  dv :: Vector{TF}
  w :: Vector{TF}
  u :: Vector{TF}
  wbar :: Vector{TF}
  ubar :: Vector{TF}
  rw :: Vector{TF}
  ru :: Vector{TF}
  w_workspace :: Vector{TF}
  workspace_Lz :: Vector{TF}
  workspace_Lv :: Vector{TF}
  # Restarted Broyden
  sz :: Vector{TF}
  sv :: Vector{TF}
  stildez :: Vector{TF}
  stildev :: Vector{TF}
  yz :: Vector{TF}
  yv :: Vector{TF}
  Psz :: Vector{TF}
  Psv :: Vector{TF}
  Sz_buf :: Vector{TF}
  Sv_buf :: Vector{TF}
  Stildez_buf :: Vector{TF}
  Stildev_buf :: Vector{TF}
  Psz_buf :: Vector{TF}
  Psv_buf :: Vector{TF}
  rz_old :: Vector{TF}
  rv_old :: Vector{TF}
  broyden_workspace_z :: Vector{TF}
  broyden_workspace_v :: Vector{TF}
  # Anderson
  MP :: Matrix{TF}
  MR :: Matrix{TF}
  delta_z_old :: Vector{TF}
  delta_v_old :: Vector{TF}
  delta_rz_old :: Vector{TF}
  delta_rv_old :: Vector{TF}
end

struct MODEL_SP_IMPLICITL{TI, TF, TM} <: CUSTOM_SOLVER_MODEL
  solver_state :: GENERIC_SOLVER_STATE{TF, TI}
  solver_state_internal :: SP_IMPLICITL_STATE_INTERNAL{TI, TF, TM}
  problem_definition :: GENERIC_PROBLEM_DEFINITIONV2{TF, TI}
end

"""
Verbose levels
"""
@enum VERBOSE_LEVEL begin
  SILENT = 1
  LOG = 2
end

##############
# Model union types
##############

# All models with their dynamics in L
const MODEL_DYNAMICSL = Union{MODEL_CP_DYNAMICSL, MODEL_SP_DYNAMICSL}

# All models with L implicitly constructed
const MODEL_IMPLICITL = Union{MODEL_CP_IMPLICITL, MODEL_SP_IMPLICITL}

# All models using the plain CP algorithm
const MODEL_CP = Union{MODEL_CP_DYNAMICSL, MODEL_CP_IMPLICITL}

# All models using the CP + SuperMann algorithm
const MODEL_SP = Union{MODEL_SP_DYNAMICSL, MODEL_SP_IMPLICITL}

#####################################################
# Exposed API funcions
#####################################################

function build_model(
  scen_tree :: ScenarioTree, 
  cost :: Cost, 
  dynamics :: Dynamics, 
  rms :: Union{Vector{RiskMeasureV1}, Vector{RiskMeasureV2}},
  solver_options :: SolverOptions = SolverOptions(DYNAMICSL, CP)
)
"""
Supported combinations of solver options:

- CP + Dynamics in L
- SP + Dynamics in L
- CP + Implicit L
"""

if solver_options.dynamics == DYNAMICSL
  if solver_options.algorithm == CP
    return build_model_cp_dynamicsl(scen_tree, cost, dynamics, rms)
  elseif solver_options.algorithm == SP
    return build_model_sp_dynamicsl(scen_tree, cost, dynamics, rms)
  end
elseif solver_options.dynamics == L_IMPLICIT
  if solver_options.algorithm == CP
    return build_model_cp_implicitl(scen_tree, cost, dynamics, rms)
  elseif solver_options.algorithm == SP
    return build_model_sp_implicitl(scen_tree, cost, dynamics, rms)
  end
end

error("This combination of solver options is not supported.")

end

function solve_model!(
  model :: CUSTOM_SOLVER_MODEL,
  x0 :: AbstractArray{TF, 1};
  tol :: TF = 1e-3,
  verbose :: VERBOSE_LEVEL = SILENT,
  z0 :: Union{Vector{Float64}, Nothing} = nothing, 
  v0 :: Union{Vector{Float64}, Nothing} = nothing,
  gamma :: Union{Float64, Nothing} = nothing,
  sigma :: Union{Float64, Nothing} = nothing
) where {TF <: Real}
  error("Solving this model is not supported.")
end