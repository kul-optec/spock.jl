abstract type Cost end

struct CostV1{TF <: Real} <: Cost
  Q :: Vector{Matrix{TF}}
  R :: Vector{Matrix{TF}}
end

struct CostV2{TF <: Real} <: Cost
  Q :: Vector{Matrix{TF}}
  R :: Vector{Matrix{TF}}
  QN :: Vector{Matrix{TF}}
end