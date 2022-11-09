import LinearAlgebra as LA

using spock, JuMP, Plots, Random

pgfplotsx()

# Random.seed!(1)

# TODO: Compute cold starts on exactly the same problem.

N_max = 9
N_ref_max = N_max

model_timings = zeros(N_max - 2)
mosek_timings = zeros(N_ref_max - 2)
gurobi_timings = zeros(N_ref_max - 2)
model_warmstart_timings = zeros(N_ref_max - 2)

for N = 3:N_ref_max

  #######################################
  ### Problem definition
  #######################################

  # Prediction horizon and branching factor
  d = 2

  # Dimensions of state and input vectors
  nx = 2; nu = 1

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
  T_s = 0.1
  A = [[[1.,0.] [T_s, 1.0 - (i - 1) / d * T_s]] for i in 1:d]
  B = [reshape([0., T_s], :, 1) for _ in 1:d]
  dynamics = spock.Dynamics(A, B)

  # Risk measures: AV@R
  p_ref = [0.3, 0.7]; alpha=0.95
  rms = spock.get_uniform_rms_avar_v2(p_ref, alpha, d, N);
  rms = spock.get_nonuniform_rms_avar_v2(d, N);

  model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))

  # model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms)
  # spock.solve_model(model_mosek, [0.1, 0.1]) # already run to avoid false timings due to cold start

  # model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms)
  # spock.solve_model(model_gurobi, [0.1, 0.1]) # already run to avoid false timings due to cold start


  ##########################################
  ###  MPC simulation
  ##########################################

  MPC_N = 30

  x0 = [.1, .1]

  model_timings_mpc = zeros(MPC_N)
  mosek_timings_mpc = zeros(MPC_N)
  gurobi_timings_mpc = zeros(MPC_N)

  x0_mosek = copy(x0)
  x0_gurobi = copy(x0)

  for t = 1:MPC_N
    factor = 1e3
    model_timings_mpc[t] = @elapsed spock.solve_model!(model, x0 / factor, tol=1e-4 / factor)

    global model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms)
    mosek_timings_mpc[t] = @elapsed spock.solve_model(model_mosek, x0_mosek)

    global model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms)
    gurobi_timings_mpc[t] = @elapsed spock.solve_model(model_gurobi, x0_gurobi)

    u = model.solver_state.z[model.solver_state_internal.u_inds[1]] * factor
    u_mosek = value.(model_mosek[:u][1])
    u_gurobi = value.(model_gurobi[:u][1])

    x0 = dynamics.A[1] * x0 + dynamics.B[1] * u
    x0_mosek = dynamics.A[1] * x0_mosek + dynamics.B[1] * u_mosek
    x0_gurobi = dynamics.A[1] * x0_gurobi + dynamics.B[1] * u_gurobi

    x0 = reshape(x0, 2)
    x0_mosek = reshape(x0_mosek, 2)
    x0_gurobi = reshape(x0_gurobi, 2)
  end

  model_timings[N-2] = model_timings_mpc[1]
  mosek_timings[N-2] = sum(mosek_timings_mpc) / length(mosek_timings_mpc)
  gurobi_timings[N-2] = sum(gurobi_timings_mpc) / length(gurobi_timings_mpc)
  model_warmstart_timings[N-2] = sum(model_timings_mpc[21:30]) / 10.

end

###########################################
###  Plot results
###########################################

fig = plot(
  xlabel = "Horizon N",
  ylabel = "Solver run time [s]",
  fmt = :pdf,
  legend = true
)

plot!(3:N_ref_max, model_timings, color=:red, yaxis=:log, labels=["SPOCK"])
plot!(3:N_ref_max, mosek_timings, color=:blue, yaxis=:log, labels=["MOSEK"])
plot!(3:N_ref_max, gurobi_timings, color=:green, yaxis=:log, labels=["GUROBI"])
plot!(3:N_ref_max, model_warmstart_timings, color=:purple, yaxis=:log, labels=["SPOCK (warm-started)"])

savefig("examples/output/scaling_extended.pdf")