import MathOptInterface as MOI
import MathOptSetDistances as MOD
import LinearAlgebra as LA

using JuMP, SparseArrays, Test

include("../src/cost.jl")
include("../src/dynamics.jl")
include("../src/scenario_tree.jl")
include("../src/risk_measures.jl")
include("../src/model.jl")

## The supported algorithms
include("../src/model_algorithms/cp.jl")
include("../src/model_algorithms/sp.jl")

## The supported dynamics
include("../src/model_dynamics/dynamics_in_L.jl")
include("../src/model_dynamics/implicit_l.jl")

## The specific models
include("../src/models/cpock.jl")
include("../src/models/spock.jl")
include("../src/models/model_cp_implicit.jl")
include("../src/models/model_mosek.jl")


###########################
# Load the actual tests
###########################

## The supported algorithms
include("model_algorithms/prox.jl")

## The supported dynamics
include("model_dynamics/implicit_L.jl")

##########
# Debugging
##########
# for N = 3

x = cp_model.solver_state.z[1 : 14]
u = cp_model.solver_state.z[15: 17]
s = cp_model.solver_state.z[18:24]
tau = cp_model.solver_state.z[25:30]
y = cp_model.solver_state.z[31:45]

println("x = $(x)")
println("u = $(u)")
println("s = $(s)")
println("tau = $(tau)")
println("y = $(y)")

## TODO: Refactor into nice tests:

### Check dynamics

# println("Verifying y >=_K 0...")
# @testset "y in K verification" begin
#   offset = 0
#   for i = 1:scen_tree.n_non_leaf_nodes
#     for cone in rms[i].K.subcones
#       dim = MOI.dimension(cone)
#       # Check that y \in K by checking if proj_K(y) = y
#       @test isapprox(
#         y[1 + offset : dim + offset], 
#         MOD.projection_on_set(MOD.DefaultDistance(), y[1 + offset : dim + offset], MOI.dual_set(cone))
#       )
#       offset += dim
#     end
#   end
# end

# println("Verifying y'b <= s...")
# @testset "y'b <= s verification" begin
#   offset = 0
#   for i = 1:scen_tree.n_non_leaf_nodes
#     b = rms[i].b; ny = length(b)
#     @test b' * y[offset + 1: offset + ny] <= s[i]
#     offset += ny
#   end
# end

# println("Verifying ell(x, u, w) <= tau")
# @testset "ell(x, u, w) <= tau" begin
#   for i = 1:scen_tree.n_non_leaf_nodes
#     children_of_i = scen_tree.child_mapping[i]
#     for j in children_of_i
#       stage_cost = x[(i - 1) * nx + 1 : i * nx]' * cost.Q[j-1] * x[(i - 1) * nx + 1 : i * nx] +
#         u[(i - 1) * nu + 1 : i * nu]' * cost.R[j-1] * u[(i-1) * nu + 1 : i * nu]
#       @test stage_cost <= tau[j - 1]
#     end
#   end
# end

# println("Verifying ell_N(x) <= s")
# @testset "ell_N(x) <= s" begin
#   for (ind, i) in enumerate(collect(scen_tree.leaf_node_min_index:scen_tree.leaf_node_max_index))
#     stage_cost = x[(i - 1) * nx + 1 : i * nx]' * cost.QN[ind] * x[(i - 1) * nx + 1 : i * nx]
#     @test stage_cost <= s[i]
#   end
# end

##################
# Dual variables
##################

# v1 = cp_model.solver_state.vbar[1:15]
# v2 = cp_model.solver_state.vbar[16:18]
# v3 = cp_model.solver_state.vbar[19:30]
# v4 = cp_model.solver_state.vbar[31:36]
# v5 = cp_model.solver_state.vbar[17:42]
# v6 = cp_model.solver_state.vbar[43:48]
# v11 = cp_model.solver_state.vbar[49:56]
# v12 = cp_model.solver_state.vbar[57:60]
# v13 = cp_model.solver_state.vbar[61:64]


#######################
# Firmly nonexpansiveness of proximal operators
#######################

# @testset "Projection onto dynamics is firmly nonexpansive" begin
#   for _ = 1:20
#     nz = cp_model.solver_state.nz
#     cp_model.problem_definition.x0[1:2] = [0.1, 0.1]
#     proj1 = rand(nz); z1 = copy(proj1)
#     proj2 = rand(nz); z2 = copy(proj2)
#     # proj1 = 50. *ones(nz); proj1[1:2] = [0.1, 0.1]; z1 = copy(proj1)
#     # proj2 = 50. * ones(nz); proj2[1:2] = [0.1, 0.1]; z2 = copy(proj2)
#     gamma = 0.1

#     projection_S1!(cp_model, proj1, gamma)
#     projection_S1!(cp_model, proj2, gamma)

#     len = cp_model.solver_state_internal.s_inds[1] - 1
#     z1 = z1[1:len]; z2 = z2[1:len]
#     proj1 = proj1[1:len]; proj2 = proj2[1:len]

#     # println(proj1 - proj2)
#     # println(z1 - z2)
#     # println((proj1 - proj2) .* (z1 - z2))
#     # println("-------------------------")

#     @test (LA.dot(proj1 - proj2, z1 - z2) >= LA.norm(proj1 - proj2)^2)
#   end
# end

# @testset "Kernel projection is firmly nonexpansive" begin
#   for _ = 1:10
#     nz = cp_model.solver_state.nz
#     proj1 = rand(nz); z1 = copy(proj1)
#     proj2 = rand(nz); z2 = copy(proj2)
#     gamma = 0.1

#     projection_S2!(cp_model, proj1, gamma)
#     projection_S2!(cp_model, proj2, gamma)

#     start = cp_model.solver_state_internal.s_inds[1] + 1
#     stop = cp_model.solver_state.nz
#     z1 = z1[start : stop]; z2 = z2[start:stop]
#     proj1 = proj1[start : stop]; proj2 = proj2[start:stop]

#     @test (LA.dot(proj1 - proj2, z1 - z2) >= LA.norm(proj1 - proj2)^2)
#   end
# end