###
# Risk constraints
###

# To support more cones, extend ConvexBaseCone with more MOI.AbstractVectorSet types
const ConvexBaseCone = Union{MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.Reals, MOI.SecondOrderCone}

# dual_set(cone :: T) where T <: MOI.AbstractVectorSet = 

# function project_onto_cone!(vec :: AbstractVector{TF}, cone :: AbstractVectorSet) where {TF <: Real}
#   error("Not implemented")
# end

function project_onto_cone!(vec :: AbstractVector{TF}, cone :: MOI.Zeros) where {TF <: Real}
  for i in eachindex(vec)
    vec[i] = 0.
  end
end

function project_onto_cone!(vec :: AbstractVector{TF}, cone :: MOI.Nonnegatives) where {TF <: Real}
  for i in eachindex(vec)
    if vec[i] < 0.
      vec[i] = 0.
    end
  end
end

function project_onto_cone!(vec :: AbstractVector{TF}, cone :: MOI.Nonpositives) where {TF <: Real}
  for i in eachindex(vec)
    if vec[i] > 0.
      vec[i] = 0.
    end
  end
end

function project_onto_cone!(vec :: AbstractVector{TF}, cone :: MOI.Reals) where {TF <: Real}
  # Nothing should be done
end

function project_onto_cone!(vec :: AbstractArray{TF}, cone :: MOI.SecondOrderCone) where {TF <: Real}
  # vec[1:length(vec)] = MOD.projection_on_set(
  #   MOD.DefaultDistance(), 
  #   vec, 
  #   cone
  # )

  # return

  t = vec[1]
  x_norm = 0.
  for i = 2:length(vec)
    x_norm += vec[i]^2
  end
  x_norm = sqrt(x_norm)

  # || x || <= t
  if x_norm <= t

  # || x || <= -t
  elseif x_norm <= -t
    for k = 1:length(vec)
      vec[k] = 0.
    end
  # [t, x] <- (t + x_norm) / (2 * x_norm) * [x_norm, x]
  else
    vec[1] = (t + x_norm) / 2.
    for k = 2:length(vec)
      vec[k] = vec[1] * vec[k] / x_norm
    end
  end
end

struct ConvexCone
  subcones:: Vector{ConvexBaseCone}
end

abstract type RiskMeasure end 

struct RiskMeasureV2 <: RiskMeasure
  E:: Matrix{Float64}
  F:: Matrix{Float64}
  b:: Vector{Float64}
  K:: ConvexCone
end

#####################################################
# Exposed API funcions
#####################################################

"""
Return a RiskMeasure object with a uniform risk mapping over all nodes
"""
function get_uniform_rms_v2(E, F, b, K, d, N)
  return [
        RiskMeasureV2(
            E,
            F,
            b,
            K,
        ) for _ in 1:(d^(N - 1) - 1) / (d - 1)
    ]
end

"""
Returns a RiskMeasure object for a constant branching factor d where all risk mappings are uniformly AV@R
E = [alpha * I_d; -I_d; 1_d']
F =  zeros(2d+1 x d ) (anything, d times d)
b = [p; 0_d; 1]
"""
function get_uniform_rms_avar_v2(p, alpha, d, N)
  return get_uniform_rms_v2(
    [alpha * LA.I(d); - LA.I(d); [1. for _ in 1:d]'],
    zeros(2*d+1, d),
    [p; [0. for _ in 1:d]; 1.],
    ConvexCone([MOI.Nonnegatives(2 * d), MOI.Zeros(1)]),
    d,
    N
  )
end

function rand_probvec2(d)
  v = rand(d)
  return v / sum(v)
end


function get_nonuniform_rms_avar_v2(d, N)
  return [
    RiskMeasureV2(
        [rand() * LA.I(d); - LA.I(d); [1. for _ in 1:d]'],
        zeros(2*d+1, d),
        [rand_probvec2(d); [0. for _ in 1:d]; 1.],
        ConvexCone([MOI.Nonnegatives(2 * d), MOI.Zeros(1)]),
    ) for _ in 1:(d^(N - 1) - 1) / (d - 1)
  ]
end

function get_uniform_rms_tv_v2(p, r, d, N)
  return get_uniform_rms_v2(
    [0.5 * LA.I(d); -0.5 * LA.I(d); zeros(d, d)],
    [-LA.I(d); -LA.I(d); LA.I(d)],
    [0.5 * p; -0.5 * p; [r for _ in 1:d]],
    ConvexCone([MOI.Nonnegatives(3*d)]),
    d,
    N
  )
end