abstract type AbstractCost end

struct Cost{TM} <: AbstractCost
  Q :: Vector{TM}
  R :: Vector{TM}
  QN :: Vector{TM}
end