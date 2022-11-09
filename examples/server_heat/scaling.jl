import LinearAlgebra as LA

using spock, JuMP, Plots

include("server_heat.jl")

pgfplotsx()

M = 1
N_max = 12
N_ref_max = 11
N_low = 7
# Dimensions of state and input vectors
nx = 50
x0 = [i <= 2 ? .1 : .1 for i = 1:nx]
d = 2
TOL = 1e-3

model_timings = zeros(N_max - 2)
mosek_timings = zeros(N_max - 2)
gurobi_timings = zeros(N_max - 2)
ipopt_timings = zeros(N_ref_max - 2)
sedumi_timings = zeros(N_low - 2)
cosmo_timings = zeros(N_ref_max - 2)

for m = 1:M
  for N = 3:max(N_ref_max, N_max)

    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d)

    if N <= N_max
      model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))
    end

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

    if N <= N_max
      model_timings[N - 2] += @elapsed spock.solve_model!(model, x0, tol=TOL)
      mosek_timings[N - 2] += @elapsed spock.solve_model(model_mosek, x0)
      gurobi_timings[N - 2] += @elapsed spock.solve_model(model_gurobi, x0)
    end
    if N <= N_ref_max
      # gurobi_timings[N - 2] += @elapsed spock.solve_model(model_gurobi, x0)
      ipopt_timings[N - 2] += @elapsed spock.solve_model(model_ipopt, x0)
      cosmo_timings[N - 2] += @elapsed spock.solve_model(model_cosmo, x0)
    end
    if N <= N_low
      sedumi_timings[N - 2] += @elapsed spock.solve_model(model_sedumi, x0)
    end

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
  xlabel = "Horizon N",
  ylabel = "Solver run time [s]",
  fmt = :pdf,
  legend = true
)

plot!(3:N_max, model_timings, color=:red, yaxis=:log, labels=["SPOCK"])
plot!(3:N_max, mosek_timings, color=:blue, yaxis=:log, labels=["MOSEK"])
plot!(3:N_max, gurobi_timings, color=:green, yaxis=:log, labels=["GUROBI"])
plot!(3:N_ref_max, ipopt_timings, color=:purple, yaxis=:log, labels=["IPOPT"])
plot!(3:N_low, sedumi_timings, color=:orange, yaxis=:log, labels=["SEDUMI"])
plot!(3:N_ref_max, cosmo_timings, color=:black, yaxis=:log, labels=["COSMO"])

savefig("examples/server_heat/output/scaling.pdf")

savefig("examples/server_heat/output/scaling.tikz")