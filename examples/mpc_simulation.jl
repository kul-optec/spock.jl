import LinearAlgebra as LA

using spock, JuMP, Plots, Random

pgfplotsx()

# Random.seed!(2)

#######################################
### Problem definition
#######################################

# Prediction horizon and branching factor
N = 7; d = 2

# Dimensions of state and input vectors
nx = 30; nu = nx ÷ 2

# Scenario tree definition
scen_tree = spock.generate_scenario_tree_uniform_branching_factor_v2(N, d, nx, nu)

# Cost definition (Quadratic, positive definite)
# cost = spock.CostV2(
#   # Q matrices
#   collect([
#     LA.Matrix([(2.2) 0; 0 (3.7)]) / 3.7 for i in 1:scen_tree.n - 1
#   ]),
#   # R matrices
#   collect([
#     reshape([(3.2)], 1, 1) / 3.7 for i in 1:scen_tree.n - 1
#   ]),
#   # QN matrices
#   collect([
#     LA.Matrix([(2.2) 0; 0 (3.7)]) / 3.7 for i in 1:scen_tree.n_leaf_nodes
#   ])
# )
Qs = collect([
  rand(nx, nx) for i in 1:scen_tree.n - 1
])
Rs = collect([
  rand(nu, nu) for i in 1:scen_tree.n - 1
])
QNs = collect([
  rand(nx, nx) for i in 1:scen_tree.n_leaf_nodes
])

cost = spock.CostV2(
  # Q matrices
  map(x -> x' * x / LA.opnorm(x' * x), Qs),
  # R matrices
  map(x -> x' * x / LA.opnorm(x' * x), Rs),
  # QN matrices
  map(x -> x' * x / LA.opnorm(x' * x), QNs)
)

# Dynamics (based on a discretised car model)
T_s = 0.01
A = [LA.diagm([j <= nx ? 1. - (i - 1) * T_s : 1. for j in 1:nx]) for i in 1:d]
B = [zeros(nx, nu) for _ in 1:d]

# for i = 1:nx
#   for j = i+1:nx
#     A[1][i, j] = T_s
#     A[2][i, j] = T_s
#   end
# end
# B[1][1, :] .= 0.; B[2][1, :] .= 0.
for i = 1:nu
  B[1][i, i] = T_s
  B[2][i, i] = T_s
end
dynamics = spock.Dynamics(A, B)

# A = [[[1.,0.] [T_s, 1.0 - (i - 1) / d * T_s]] for i in 1:d]
# B = [reshape([0., T_s], :, 1) for _ in 1:d]
# dynamics = spock.Dynamics(A, B)

# Risk measures: AV@R
p_ref = [0.3, 0.7]; alpha=0.95
rms = spock.get_uniform_rms_avar_v2(p_ref, alpha, d, N);
# rms = spock.get_nonuniform_rms_avar_v2(d, N);

model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))
cp_model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.CP))

model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms)
spock.solve_model(model_mosek, [0.1 for _ in 1:nx]) # already run to avoid false timings due to cold start

model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms)
spock.solve_model(model_gurobi, [0.1 for _ in 1:nx]) # already run to avoid false timings due to cold start

model_ipopt = spock.build_model_ipopt(scen_tree, cost, dynamics, rms)
spock.solve_model(model_ipopt, [0.1 for _ in 1:nx]) # already run to avoid false timings due to cold start


##########################################
###  MPC simulation
##########################################

MPC_N = 30

x0 = [.1 for _ in 1:nx]

model_timings = zeros(MPC_N)
cp_model_timings = zeros(MPC_N)
mosek_timings = zeros(MPC_N)
gurobi_timings = zeros(MPC_N)
ipopt_timings = zeros(MPC_N)

x0_cp = copy(x0)
x0_mosek = copy(x0)
x0_gurobi = copy(x0)
x0_ipopt = copy(x0)

for t = 1:MPC_N
  factor = 1e0
  model_timings[t] = @elapsed spock.solve_model!(model, x0 / factor, tol=1e-4 / factor)
  # cp_model_timings[t] = @elapsed spock.solve_model!(cp_model, x0_cp / factor, tol=1e-3 / factor)

  global model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms)
  mosek_timings[t] = @elapsed spock.solve_model(model_mosek, x0_mosek)

  global model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms)
  gurobi_timings[t] = @elapsed spock.solve_model(model_gurobi, x0_gurobi)

  global model_ipopt = spock.build_model_ipopt(scen_tree, cost, dynamics, rms)
  ipopt_timings[t] = @elapsed spock.solve_model(model_ipopt, x0_ipopt)

  u = model.solver_state.z[model.solver_state_internal.u_inds[1:nu]] * factor
  u_cp = cp_model.solver_state.z[cp_model.solver_state_internal.u_inds[1:nu]] * factor
  u_mosek = value.(model_mosek[:u][1:nu])
  u_gurobi = value.(model_gurobi[:u][1:nu])
  u_ipopt = value.(model_ipopt[:u][1:nu])
  # println("$(u),    $(u_mosek),     $(u_gurobi)")

  println(x0[1:8])
  println(x0_mosek[1:8])
  println("----")

  global x0 = dynamics.A[1] * x0 + dynamics.B[1] * u
  global x0_cp = dynamics.A[1] * x0_cp + dynamics.B[1] * u_cp
  global x0_mosek = dynamics.A[1] * x0_mosek + dynamics.B[1] * u_mosek
  global x0_gurobi = dynamics.A[1] * x0_gurobi + dynamics.B[1] * u_gurobi
  global x0_ipopt = dynamics.A[1] * x0_ipopt + dynamics.B[1] * u_ipopt

  x0 = reshape(x0, nx)
  x0_cp = reshape(x0_cp, nx)
  x0_mosek = reshape(x0_mosek, nx)
  x0_gurobi = reshape(x0_gurobi, nx)
  x0_ipopt = reshape(x0_ipopt, nx)
end

###########################################
###  Plot results
###########################################

fig = plot(
  xlabel = "MPC time step",
  ylabel = "Solver run time [s]",
  fmt = :pdf,
  legend = true
)

# plot!(1:MPC_N, cp_model_timings[1:MPC_N], color=:orange, labels=["CPOCK"])
plot!(1:MPC_N, model_timings[1:MPC_N], color=:red, labels=["SPOCK"])
plot!(1:MPC_N, mosek_timings[1:MPC_N], color=:blue, labels=["MOSEK"])
plot!(1:MPC_N, gurobi_timings[1:MPC_N], color=:green, yaxis=:log, labels=["GUROBI"])
plot!(1:MPC_N, ipopt_timings[1:MPC_N], color=:purple, yaxis=:log, labels=["IPOPT"])

savefig("examples/output/mpc_simulation.pdf")

savefig("examples/output/mpc_simulation.tikz")