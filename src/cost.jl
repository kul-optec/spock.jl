abstract type Cost end

struct CostV2{TM} <: Cost
  Q :: Vector{TM}
  R :: Vector{TM}
  QN :: Vector{TM}
end