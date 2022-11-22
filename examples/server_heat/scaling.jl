import LinearAlgebra as LA

using spock, JuMP, Plots

include("server_heat.jl")

pgfplotsx()

M = 1
N_max = 15
# Dimensions of state and input vectors
nx = 50
x0 = [i <= 2 ? .1 : .1 for i = 1:nx]
d = 2
TOL = 1e-3

t_max = 150

model_timings = zeros(N_max - 2)
mosek_timings = zeros(N_max - 2)
gurobi_timings = zeros(N_max - 2)
ipopt_timings = zeros(N_max - 2)
sedumi_timings = zeros(N_max - 2)
cosmo_timings = zeros(N_max - 2)

for m = 1:M
  for N = 3:N_max

    scen_tree, cost, dynamics, rms, constraints = get_server_heat_specs(N, nx, d)

    model = spock.build_model(scen_tree, cost, dynamics, rms, constraints, spock.SolverOptions(spock.SP, nothing))

    model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms, constraints)
    set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_TOL_REL_GAP", TOL)
    set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", TOL)
    set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", TOL)

    model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms, constraints)
    set_optimizer_attribute(model_gurobi, "FeasibilityTol", TOL)
    set_optimizer_attribute(model_gurobi, "OptimalityTol", TOL)

    model_ipopt = spock.build_model_ipopt(scen_tree, cost, dynamics, rms, constraints)
    set_optimizer_attribute(model_ipopt, "tol", TOL)

    model_sedumi = spock.build_model_sedumi(scen_tree, cost, dynamics, rms, constraints)

    model_cosmo = spock.build_model_cosmo(scen_tree, cost, dynamics, rms, constraints)
    set_optimizer_attribute(model_cosmo, "eps_abs", TOL)
    set_optimizer_attribute(model_cosmo, "eps_rel", TOL)

    ##########################################
    ###  Solution
    ##########################################

    println("solving...")

    if maximum(model_timings) <= t_max
      model_timings[N - 2] += @elapsed spock.solve_model!(model, x0, tol=TOL)
    end
    if maximum(mosek_timings) <= t_max
      mosek_timings[N - 2] += @elapsed spock.solve_model(model_mosek, x0)
    end
    if maximum(gurobi_timings) <= t_max
      gurobi_timings[N - 2] += @elapsed spock.solve_model(model_gurobi, x0)
    end
    if maximum(ipopt_timings) <= t_max && N <= 12
      ipopt_timings[N - 2] += @elapsed spock.solve_model(model_ipopt, x0)
    end
    if maximum(cosmo_timings) <= t_max
      cosmo_timings[N - 2] += @elapsed spock.solve_model(model_cosmo, x0)
    end
    if maximum(sedumi_timings) <= t_max
      sedumi_timings[N - 2] += @elapsed spock.solve_model(model_sedumi, x0)
    end

    # println("SPOCK: $(model.state.z[model.solver_state_internal.s_inds[1]]), MOSEK: $(value(model_mosek[:s][1]))")

  end
end

model_timings ./= M
mosek_timings ./= M
gurobi_timings ./= M
sedumi_timings ./= M
ipopt_timings ./= M
cosmo_timings ./= M

fig = plot(
  xlabel = "Horizon N",
  ylabel = "Solver run time [s]",
  fmt = :pdf,
  legend = true
)

model_timings = filter(>(0.), model_timings)
mosek_timings = filter(>(0.), mosek_timings)
gurobi_timings = filter(>(0.), gurobi_timings)
ipopt_timings = filter(>(0.), ipopt_timings)
sedumi_timings = filter(>(0.), sedumi_timings)
cosmo_timings = filter(>(0.), cosmo_timings)

plot!(3:length(model_timings) + 2, model_timings, color=:red, yaxis=:log, labels=["SPOCK"])
plot!(3:length(mosek_timings) + 2, mosek_timings, color=:blue, yaxis=:log, labels=["MOSEK"])
plot!(3:length(gurobi_timings) + 2, gurobi_timings, color=:green, yaxis=:log, labels=["GUROBI"])
plot!(3:length(ipopt_timings) + 2, ipopt_timings, color=:purple, yaxis=:log, labels=["IPOPT"])
plot!(3:length(sedumi_timings) + 2, sedumi_timings, color=:orange, yaxis=:log, labels=["SEDUMI"])
plot!(3:length(cosmo_timings) + 2, cosmo_timings, color=:black, yaxis=:log, labels=["COSMO"])

savefig("examples/server_heat/output/scaling.pdf")
savefig("examples/server_heat/output/scaling.tikz")