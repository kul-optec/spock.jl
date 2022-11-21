module spock

  import MathOptInterface as MOI
  import MathOptSetDistances as MOD
  import LinearAlgebra as LA

  using JuMP, SparseArrays, Test, DelimitedFiles

  include("cost.jl")
  include("dynamics.jl")
  include("scenario_tree.jl")
  include("risk_measures.jl")
  include("constraints.jl")
  include("model.jl")

  ## The supported algorithms
  include("model_algorithms/cp.jl")
  include("model_algorithms/sp.jl")

  # Quasi-Newton directions
  include("model_algorithms/qnewton_directions/restarted_broyden.jl")
  include("model_algorithms/qnewton_directions/anderson.jl")

  ## The supported dynamics
  include("model_dynamics/dynamics_in_L.jl")
  include("model_dynamics/implicit_l.jl")

  ## The specific models
  include("models/cpock.jl")
  include("models/spock.jl")
  include("models/model_cp_implicit.jl")
  include("models/model_sp_implicit.jl")
  include("models/model_mosek.jl")

  # Precompile for the required arguments
  precompile(solve_model!, (MODEL_CP_IMPLICITL, Vector{Float64}))
  precompile(solve_model!, (MODEL_SP_IMPLICITL, Vector{Float64}))

end