import LinearAlgebra as LA

using spock, JuMP, Plots

include("server_heat.jl")

pgfplotsx()

M = 1
N = 5
# Dimensions of state and input vectors
nx = 4
x0 = [i <= 2 ? .1 : .1 for i = 1:nx]
d = 2
TOL = 1e-3

alphas = 0.05:0.1:0.95

model_timings = zeros(length(alphas))
mosek_timings = zeros(length(alphas))
gurobi_timings = zeros(length(alphas))
ipopt_timings = zeros(length(alphas))
sedumi_timings = zeros(length(alphas))
cosmo_timings = zeros(length(alphas))

for m = 1:M
  for (alpha_i, alpha) in enumerate(alphas)

    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d, alpha)

    model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))

    model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms)
    set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_TOL_REL_GAP", TOL)
    set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", TOL)
    set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", TOL)

    model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms)
    set_optimizer_attribute(model_gurobi, "FeasibilityTol", TOL)
    set_optimizer_attribute(model_gurobi, "OptimalityTol", TOL)

    model_ipopt = spock.build_model_ipopt(scen_tree, cost, dynamics, rms)
    set_optimizer_attribute(model_ipopt, "tol", TOL)

    model_sedumi = spock.build_model_sedumi(scen_tree, cost, dynamics, rms)

    model_cosmo = spock.build_model_cosmo(scen_tree, cost, dynamics, rms)
    set_optimizer_attribute(model_cosmo, "eps_abs", TOL)
    set_optimizer_attribute(model_cosmo, "eps_rel", TOL)

    ##########################################
    ###  Solution
    ##########################################

    model_timings[alpha_i] += @elapsed spock.solve_model!(model, x0, tol=TOL)
    mosek_timings[alpha_i] += @elapsed spock.solve_model(model_mosek, x0)
    gurobi_timings[alpha_i] += @elapsed spock.solve_model(model_gurobi, x0)
    ipopt_timings[alpha_i] += @elapsed spock.solve_model(model_ipopt, x0)
    cosmo_timings[alpha_i] += @elapsed spock.solve_model(model_cosmo, x0)
    sedumi_timings[alpha_i] += @elapsed spock.solve_model(model_sedumi, x0)

    println("SPOCK: $(model.solver_state.z[model.solver_state_internal.s_inds[1]]), MOSEK: $(value(model_mosek[:s][1]))")

  end

  model_timings ./= M
  mosek_timings ./= M
  gurobi_timings ./= M
  sedumi_timings ./= M
  ipopt_timings ./= M
  cosmo_timings ./= M
end

fig = plot(
  xlabel = "α",
  ylabel = "CPU time [s]",
  fmt = :pdf,
  legend = true
)

plot!(alphas, model_timings, color=:red, labels=["SPOCK"])
plot!(alphas, mosek_timings, color=:blue, labels=["MOSEK"])
plot!(alphas, gurobi_timings, color=:green, labels=["GUROBI"])
plot!(alphas, ipopt_timings, color=:purple, labels=["IPOPT"])
plot!(alphas, sedumi_timings, color=:orange, labels=["SEDUMI"])
plot!(alphas, cosmo_timings, color=:black, labels=["COSMO"])

savefig("examples/server_heat/output/scaling_alpha.pdf")

savefig("examples/server_heat/output/scaling_alpha.tikz")