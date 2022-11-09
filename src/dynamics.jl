struct Dynamics{TF <: Real}
  A :: Vector{Matrix{TF}}
  B :: Vector{Matrix{TF}}
end