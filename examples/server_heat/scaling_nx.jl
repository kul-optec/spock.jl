import LinearAlgebra as LA

using spock, JuMP, Plots

include("server_heat.jl")

pgfplotsx()

M = 1
N_max = 70
N_ref_max = 50
N_low = 20
# Dimensions of state and input vectors
N = 10
d = 2
TOL = 1e-3

model_timings = zeros(N_max)
mosek_timings = zeros(N_max)
gurobi_timings = zeros(N_ref_max)
ipopt_timings = zeros(N_ref_max)
sdpt3_timings = zeros(N_low)
sedumi_timings = zeros(N_low)
cosmo_timings = zeros(N_ref_max)

for m = 1:M
  for nx = 5:5:max(N_ref_max, N_max)
    x0 = [i <= 2 ? .1 : .1 for i = 1:nx]

    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d)

    if nx <= N_max
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

    model_sdpt3 = spock.build_model_sdpt3(scen_tree, cost, dynamics, rms)

    model_sedumi = spock.build_model_sedumi(scen_tree, cost, dynamics, rms)

    model_cosmo = spock.build_model_cosmo(scen_tree, cost, dynamics, rms)

    ##########################################
    ###  Solution
    ##########################################

    if nx <= N_max
      model_timings[nx] += @elapsed spock.solve_model!(model, x0, tol=TOL)
      mosek_timings[nx] += @elapsed spock.solve_model(model_mosek, x0)
    end
    if nx <= N_ref_max
      gurobi_timings[nx] += @elapsed spock.solve_model(model_gurobi, x0)
      ipopt_timings[nx] += @elapsed spock.solve_model(model_ipopt, x0)
      cosmo_timings[nx] += @elapsed spock.solve_model(model_cosmo, x0)
    end
    if nx <= N_low
      # sedumi_timings[nx] += @elapsed spock.solve_model(model_sedumi, x0)
    end

    println("SPOCK: $(model.solver_state.z[model.solver_state_internal.s_inds[1]]), MOSEK: $(value(model_mosek[:s][1]))")

  end

  model_timings ./= M
  mosek_timings ./= M
  gurobi_timings ./= M
  sdpt3_timings ./= M
  sedumi_timings ./= M
  ipopt_timings ./= M
  cosmo_timings ./= M
end

fig = plot(
  xlabel = "nx",
  ylabel = "Solver run time [s]",
  fmt = :pdf,
  legend = true
)

plot!(5:5:N_max, model_timings[5:5:N_max], color=:red, yaxis=:log, labels=["SPOCK"])
plot!(5:5:N_max, mosek_timings[5:5:N_max], color=:blue, yaxis=:log, labels=["MOSEK"])
plot!(5:5:N_ref_max, gurobi_timings[5:5:N_ref_max], color=:green, yaxis=:log, labels=["GUROBI"])
plot!(5:5:N_ref_max, ipopt_timings[5:5:N_ref_max], color=:purple, yaxis=:log, labels=["IPOPT"])
# plot!(3:N_ref_max, sdpt3_timings, color=:orange, yaxis=:log, labels=["SDPT3"])
# plot!(5:5:N_low, sedumi_timings[5:5:N_low], color=:orange, yaxis=:log, labels=["SEDUMI"])
plot!(5:5:N_ref_max, cosmo_timings[5:5:N_ref_max], color=:black, yaxis=:log, labels=["COSMO"])

savefig("examples/server_heat/output/scaling_nx.pdf")

savefig("examples/server_heat/output/scaling_nx.tikz")