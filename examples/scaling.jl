import LinearAlgebra as LA

using spock, JuMP, Plots

pgfplotsx()

M = 1
N_max = 12
N_ref_max = 12
# Dimensions of state and input vectors
nx = 25; nu = nx ÷ 2
x0 = [i <= 2 ? .1 : .1 for i = 1:nx]; factor = 1.#1e3

model_timings = zeros(N_max - 2)
mosek_timings = zeros(N_ref_max - 2)
gurobi_timings = zeros(N_ref_max - 2)
ipopt_timings = zeros(N_ref_max - 2)
sdpt3_timings = zeros(N_ref_max - 2)
sedumi_timings = zeros(N_ref_max - 2)

for m = 1:M
  for N = 3:N_ref_max

    # Branching factor
    d = 2

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
    T_s = 0.05
    A = [LA.diagm([1. + (i - 1) * T_s for _ in 1:nx]) for i in 1:d]
    B = [rand(nx, nu) for _ in 1:d]
    
    for i = 1:nx
      for j = i+1:nx
        A[1][i, j] = T_s
        A[2][i, j] = T_s
      end
    end
    B[1][1, :] .= 0.; B[2][1, :] .= 0.
    dynamics = spock.Dynamics(A, B)

    # T_s = 0.1
    # A = [[[1.,0.] [T_s, 1.0 - (i - 1) / d * T_s]] for i in 1:d]
    # B = [reshape([0., T_s], :, 1) for _ in 1:d]
    # dynamics = spock.Dynamics(A, B)

    # Risk measures: AV@R
    p_ref = [0.3, 0.7]; alpha=0.95
    rms = spock.get_uniform_rms_avar_v2(p_ref, alpha, d, N);
    # rms = spock.get_nonuniform_rms_avar_v2(d, N);

    if N <= N_max
      model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))
    end

    model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms)
    # spock.solve_model(model_mosek, [0.1, 0.1]) # already run to avoid false timings due to cold start

    model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms)
    # spock.solve_model(model_gurobi, [0.1, 0.1]) # already run to avoid false timings due to cold start

    model_ipopt = spock.build_model_ipopt(scen_tree, cost, dynamics, rms)
    # spock.solve_model(model_ipopt, [0.1, 0.1]) # already run to avoid false timings due to cold start

    model_sdpt3 = spock.build_model_sdpt3(scen_tree, cost, dynamics, rms)

    model_sedumi = spock.build_model_sedumi(scen_tree, cost, dynamics, rms)

    ##########################################
    ###  Solution
    ##########################################

    if N <= N_max
      model_timings[N - 2] += @elapsed spock.solve_model!(model, x0 / factor, tol=1e-3 / factor)
    end
    mosek_timings[N - 2] += @elapsed spock.solve_model(model_mosek, x0)
    gurobi_timings[N - 2] += @elapsed spock.solve_model(model_gurobi, x0)
    # ipopt_timings[N - 2] += @elapsed spock.solve_model(model_ipopt, x0)
    # sdpt3_timings[N - 2] += @elapsed spock.solve_model(model_sdpt3, x0)
    sedumi_timings[N - 2] += @elapsed spock.solve_model(model_sedumi, x0)

  end

  model_timings ./= M
  mosek_timings ./= M
  gurobi_timings ./= M
  sdpt3_timings ./= M
  sedumi_timings ./= M
  ipopt_timings ./= M
end

fig = plot(
  xlabel = "Horizon N",
  ylabel = "Solver run time [s]",
  fmt = :pdf,
  legend = true
)

plot!(3:N_max, model_timings, color=:red, yaxis=:log, labels=["SPOCK"])
plot!(3:N_ref_max, mosek_timings, color=:blue, yaxis=:log, labels=["MOSEK"])
plot!(3:N_ref_max, gurobi_timings, color=:green, yaxis=:log, labels=["GUROBI"])
# plot!(3:N_ref_max, ipopt_timings, color=:purple, yaxis=:log, labels=["IPOPT"])
# plot!(3:N_ref_max, sdpt3_timings, color=:orange, yaxis=:log, labels=["SDPT3"])
plot!(3:N_ref_max, sedumi_timings, color=:orange, yaxis=:log, labels=["SEDUMI"])

savefig("examples/output/scaling.pdf")

savefig("examples/output/scaling.tikz")