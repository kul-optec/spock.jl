abstract type Cost end

struct CostV1{TF <: Real} <: Cost
  Q :: Vector{Matrix{TF}}
  R :: Vector{Matrix{TF}}
end

struct CostV2{TM} <: Cost
  Q :: Vector{TM}
  R :: Vector{TM}
  QN :: Vector{TM}
end