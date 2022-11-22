abstract type ScenarioTree end

###
# Scenario tree
###

"""
Struct that stores all relevant information for some node of the scenario tree.
    - x: Indices of state variables belonging to this node. In general, this is vector valued.
    - u: Indices of input variables belonging to this node. In general, this is vector valued.
    - w: Represents the uncertainty in dynamics when going from the parent of the given
        node to the given node. This integer is used as an index to retrieve these dynamics
        from the Dynamics struct.
    - s: Non-leaf nodes: Conditional risk measure index in this node, given the current node. 
         Leaf nodes: Index of the total cost of this scenario
        Always a scalar!
Not all of these variables are defined for all nodes of the scenario tree. In such case, nothing is returned.
The above variables are defined for:
    - x: all nodes
    - u: non-leaf nodes
    - w: non-root nodes
    - s: non-leaf nodes for risk measures values, leaf nodes for cost values of corresponding scenario
"""

struct ScenarioTreeNodeInfoV2{TI <: Integer}
  x :: Union{Vector{TI}, Nothing}
  u :: Union{Vector{TI}, Nothing}
  w :: Union{TI, Nothing}
  s :: Union{TI, Nothing}
  tau :: Union{TI, Nothing}
end

"""
Struct that represents a scenario tree.
    - child_mapping: Dictionary that maps node indices to a vector of child indices
    - anc_mapping: Dictionary that maps node indices to their parent node indices
    - node_info: All relevant information stored on a node, indexable by the node index
    - nx: Number of components of a state vector in a single node
    - nu: Number of components of an input vector in a single node
    - n: Total number of nodes in this scenario tree
    - min_index_per_timestep: array containing, for each time step, the index of the first node
"""

struct ScenarioTreeV2{TI <: Integer} <: ScenarioTree
  child_mapping :: Dict{TI, Vector{TI}}
  anc_mapping :: Dict{TI, TI}
  node_info :: Vector{ScenarioTreeNodeInfoV2{TI}}
  N :: TI
  n :: TI
  n_non_leaf_nodes :: TI
  n_leaf_nodes :: TI
  leaf_node_min_index :: TI
  leaf_node_max_index :: TI
  min_index_per_timestep :: Vector{TI}
end

##########################
### Exposed utility functions
##########################

function generate_scenario_tree_uniform_branching_factor_v2(N :: TI, d :: TI, nx :: TI, nu :: TI) where {TI <: Integer}
  if d <= 1
      error("Branching factor d must be larger than 1, but is $(d).")
  end
  
  # Total number of nodes in the scenario tree
  n_total = (d^N - 1) ÷ (d - 1)
  # Total number of leaf nodes
  n_leafs = d^(N - 1)
  # Total number of non-leaf nodes
  n_non_leafs = (d^(N - 1) - 1) ÷ (d - 1)
  
  node_info = [
      ScenarioTreeNodeInfoV2(
          collect((i - 1) * nx + 1 : i * nx),
          i <= n_non_leafs ? collect((i - 1) * nu + 1 : i * nu) : nothing, # todo: write slightly more genereal for nu > 1 (similar to the line above)
          i > 1 ? (i % d) + 1 : nothing,
          i,
          i > 1 ? i : nothing
      ) for i in collect(1:n_total)
  ]

  child_mapping = Dict{TI, Vector{TI}}()
  child_index = 2
  for i = 1:n_non_leafs
      child_mapping[i] = collect(child_index : child_index + d - 1)
      child_index += d
  end

  anc_mapping = Dict{TI, TI}()
  for (key, value) in child_mapping
      for v in value
          anc_mapping[v] = key
      end
  end

  return ScenarioTreeV2(
      child_mapping,
      anc_mapping,
      node_info,
      N,
      n_total,
      n_non_leafs,
      n_leafs,
      n_non_leafs + 1,
      n_total,
      vcat([1], [1 + (d^(i) - 1)÷(d-1) for i in collect(1:N-1)])
  )
end