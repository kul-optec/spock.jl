import LinearAlgebra as LA

using spock, JuMP, Plots

include("server_heat.jl")

pgfplotsx()

M = 1
d = 2
TOL = 1e-3

alphas = 0.05:0.1:0.95

model_timings = zeros(length(alphas))
model2_timings = zeros(length(alphas))
model3_timings = zeros(length(alphas))
model4_timings = zeros(length(alphas))
model5_timings = zeros(length(alphas))
model6_timings = zeros(length(alphas))

for m = 1:M
  for (alpha_i, alpha) in enumerate(alphas)

    N = 5; nx = 2; x0 = [.1 for i = 1:nx]
    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d, alpha)
    model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))

    N = 7; nx = 2; x02 = [.1 for i = 1:nx]
    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d, alpha)
    model2 = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))
    
    N = 5; nx = 8; x03 = [.1 for i = 1:nx]
    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d, alpha)
    model3 = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))

    N = 7; nx = 8; x04 = [.1 for i = 1:nx]
    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d, alpha)
    model4 = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))

    N = 5; nx = 32; x05 = [.1 for i = 1:nx]
    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d, alpha)
    model5 = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))

    N = 7; nx = 32; x06 = [.1 for i = 1:nx]
    scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d, alpha)
    model6 = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))

    ##########################################
    ###  Solution
    ##########################################

    model_timings[alpha_i] += @elapsed spock.solve_model!(model, x0, tol=TOL)
    model2_timings[alpha_i] += @elapsed spock.solve_model!(model2, x02, tol=TOL)
    model3_timings[alpha_i] += @elapsed spock.solve_model!(model3, x03, tol=TOL)
    model4_timings[alpha_i] += @elapsed spock.solve_model!(model4, x04, tol=TOL)
    model5_timings[alpha_i] += @elapsed spock.solve_model!(model5, x05, tol=TOL)
    model6_timings[alpha_i] += @elapsed spock.solve_model!(model6, x06, tol=TOL)

  end

  model_timings ./= M
  model2_timings ./= M
  model3_timings ./= M
  model4_timings ./= M
  model5_timings ./= M
  model6_timings ./= M
end

fig = plot(
  xlabel = "α",
  ylabel = "CPU time [s]",
  fmt = :pdf,
  legend = true
)

plot!(alphas, model_timings, color=:red, labels=["N = 5, nx = 2"])
plot!(alphas, model2_timings, color=:blue, labels=["N = 7, nx = 2"])
plot!(alphas, model3_timings, color=:green, labels=["N = 5, nx = 8"])
plot!(alphas, model4_timings, color=:black, labels=["N = 7, nx = 8"])
plot!(alphas, model5_timings, color=:purple, labels=["N = 5, nx = 32"])
plot!(alphas, model6_timings, color=:orange, labels=["N = 7, nx = 32"])

savefig("examples/server_heat/output/alpha.pdf")

savefig("examples/server_heat/output/alpha.tikz")