import LinearAlgebra as LA

using spock, JuMP, Plots, Random, Statistics

include("server_heat.jl")

pgfplotsx()

Random.seed!(1)

M = 15

#######################################
### Problem definition
#######################################

# Prediction horizon and branching factor
N = 10; d = 2

# Dimensions of state and input vectors
nx = 20; nu = nx

scen_tree, cost, dynamics, rms = get_server_heat_specs(N, nx, d)

##########################################
###  MPC simulation
##########################################

MPC_N = 20
TOL = 1e-3

model_timings = zeros(MPC_N, M)
mosek_timings = zeros(MPC_N, M)
gurobi_timings = zeros(MPC_N, M)
ipopt_timings = zeros(MPC_N, M)
cosmo_timings = zeros(MPC_N, M)

for m = 1:M

model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))
x0 = [i <= 2 ? .1 : .1 for i = 1:nx]
x0_mosek = copy(x0)
x0_gurobi = copy(x0)
x0_ipopt = copy(x0)
x0_cosmo = copy(x0)

for t = 1:MPC_N
  model_timings[t, m] += @elapsed spock.solve_model!(model, x0, tol=1e-3)

  x = model.solver_state.z[model.solver_state_internal.x_inds]
  u = model.solver_state.z[model.solver_state_internal.u_inds]
  s = model.solver_state.z[model.solver_state_internal.s_inds]
  tau = model.solver_state.z[model.solver_state_internal.tau_inds]
  y = model.solver_state.z[model.solver_state_internal.y_inds]

  model_mosek = spock.build_model_mosek(scen_tree, cost, dynamics, rms)
  # if t > 1
  #   for i in eachindex(x_mosek)
  #     set_start_value(model_mosek[:x][i], x_mosek[i])
  #   end
  #   for i in eachindex(u_mosek)
  #     set_start_value(model_mosek[:u][i], u_mosek[i])
  #   end
  #   for i in eachindex(s_mosek)
  #     set_start_value(model_mosek[:s][i], s_mosek[i])
  #   end
  #   for i in eachindex(tau_mosek)
  #     set_start_value(model_mosek[:tau][i], tau_mosek[i])
  #   end
  #   for i in eachindex(y_mosek)
  #     set_start_value(model_mosek[:y][i], y_mosek[i])
  #   end
  # end
  set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_TOL_REL_GAP", TOL)
  set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", TOL)
  set_optimizer_attribute(model_mosek, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", TOL)
  mosek_timings[t, m] += @elapsed spock.solve_model(model_mosek, x0_mosek)
  global x_mosek = value.(model_mosek[:x])
  global u_mosek = value.(model_mosek[:u])
  global s_mosek = value.(model_mosek[:s])
  global tau_mosek = value.(model_mosek[:tau])
  global y_mosek = value.(model_mosek[:y])

  model_gurobi = spock.build_model_gurobi(scen_tree, cost, dynamics, rms)
  # if t > 1
  #   for i in eachindex(x_gurobi)
  #     set_start_value(model_gurobi[:x][i], x_gurobi[i])
  #   end
  #   for i in eachindex(u_gurobi)
  #     set_start_value(model_gurobi[:u][i], u_gurobi[i])
  #   end
  #   for i in eachindex(s_gurobi)
  #     set_start_value(model_gurobi[:s][i], s_gurobi[i])
  #   end
  #   for i in eachindex(tau_gurobi)
  #     set_start_value(model_gurobi[:tau][i], tau_gurobi[i])
  #   end
  #   for i in eachindex(y_gurobi)
  #     set_start_value(model_gurobi[:y][i], y_gurobi[i])
  #   end
  # end
  set_optimizer_attribute(model_gurobi, "FeasibilityTol", TOL)
  set_optimizer_attribute(model_gurobi, "OptimalityTol", TOL)
  gurobi_timings[t, m] += @elapsed spock.solve_model(model_gurobi, x0_gurobi)
  global x_gurobi = value.(model_gurobi[:x])
  global u_gurobi = value.(model_gurobi[:u])
  global s_gurobi = value.(model_gurobi[:s])
  global tau_gurobi = value.(model_gurobi[:tau])
  global y_gurobi = value.(model_gurobi[:y])

  model_ipopt = spock.build_model_ipopt(scen_tree, cost, dynamics, rms)
  # if t > 1
  #   for i in eachindex(x_ipopt)
  #     set_start_value(model_ipopt[:x][i], x_ipopt[i])
  #   end
  #   for i in eachindex(u_ipopt)
  #     set_start_value(model_ipopt[:u][i], u_ipopt[i])
  #   end
  #   for i in eachindex(s_ipopt)
  #     set_start_value(model_ipopt[:s][i], s_ipopt[i])
  #   end
  #   for i in eachindex(tau_ipopt)
  #     set_start_value(model_ipopt[:tau][i], tau_ipopt[i])
  #   end
  #   for i in eachindex(y_ipopt)
  #     set_start_value(model_ipopt[:y][i], y_ipopt[i])
  #   end
  # end
  set_optimizer_attribute(model_ipopt, "tol", TOL)
  ipopt_timings[t, m] += @elapsed spock.solve_model(model_ipopt, x0_ipopt)  
  global x_ipopt = value.(model_ipopt[:x])
  global u_ipopt = value.(model_ipopt[:u])
  global s_ipopt = value.(model_ipopt[:s])
  global tau_ipopt = value.(model_ipopt[:tau])
  global y_ipopt = value.(model_ipopt[:y])

  model_cosmo = spock.build_model_cosmo(scen_tree, cost, dynamics, rms)
  if t > 1
    for i in eachindex(x_cosmo)
      set_start_value(model_cosmo[:x][i], x_cosmo[i])
    end
    for i in eachindex(u_cosmo)
      set_start_value(model_cosmo[:u][i], u_cosmo[i])
    end
    for i in eachindex(s_cosmo)
      set_start_value(model_cosmo[:s][i], s_cosmo[i])
    end
    for i in eachindex(tau_cosmo)
      set_start_value(model_cosmo[:tau][i], tau_cosmo[i])
    end
    for i in eachindex(y_cosmo)
      set_start_value(model_cosmo[:y][i], y_cosmo[i])
    end
  end
  set_optimizer_attribute(model_cosmo, "eps_abs", TOL)
  set_optimizer_attribute(model_cosmo, "eps_rel", TOL)
  cosmo_timings[t, m] += @elapsed spock.solve_model(model_cosmo, x0_cosmo)  
  global x_cosmo = value.(model_cosmo[:x])
  global u_cosmo = value.(model_cosmo[:u])
  global s_cosmo = value.(model_cosmo[:s])
  global tau_cosmo = value.(model_cosmo[:tau])
  global y_cosmo = value.(model_cosmo[:y])

  u = model.solver_state.z[model.solver_state_internal.u_inds[1:nu]]
  u_mosek = value.(model_mosek[:u][1:nu])
  u_gurobi = value.(model_gurobi[:u][1:nu])
  u_ipopt = value.(model_ipopt[:u][1:nu])
  u_cosmo = value.(model_cosmo[:u][1:nu])

  w = rand(1:d)
  x0 = dynamics.A[w] * x0 + dynamics.B[w] * u
  x0_mosek = dynamics.A[w] * x0_mosek + dynamics.B[w] * u_mosek
  x0_gurobi = dynamics.A[w] * x0_gurobi + dynamics.B[w] * u_gurobi
  x0_ipopt = dynamics.A[w] * x0_ipopt + dynamics.B[w] * u_ipopt
  x0_cosmo = dynamics.A[w] * x0_cosmo + dynamics.B[w] * u_cosmo

  x0 = reshape(x0, nx)
  x0_mosek = reshape(x0_mosek, nx)
  x0_gurobi = reshape(x0_gurobi, nx)
  x0_ipopt = reshape(x0_ipopt, nx)
  x0_cosmo = reshape(x0_cosmo, nx)
end
end

model_timings_mean = mean(model_timings, dims=2)
mosek_timings_mean = mean(mosek_timings, dims=2)
gurobi_timings_mean = mean(gurobi_timings, dims=2)
cosmo_timings_mean = mean(cosmo_timings, dims=2)
ipopt_timings_mean = mean(ipopt_timings, dims=2)

model_timings_std = std(model_timings, dims=2)
mosek_timings_std = std(mosek_timings, dims=2)
gurobi_timings_std = std(gurobi_timings, dims=2)
cosmo_timings_std = std(cosmo_timings, dims=2)
ipopt_timings_std = std(ipopt_timings, dims=2)

###########################################
###  Plot results
###########################################

fig = plot(
  xlabel = "MPC time step",
  ylabel = "Solver run time [s]",
  fmt = :pdf,
  legend = true
)

plot!(1:MPC_N, model_timings_mean, color=:red, labels=["SPOCK"])
plot!(1:MPC_N, mosek_timings_mean, color=:blue, labels=["MOSEK"])
plot!(1:MPC_N, gurobi_timings_mean, color=:green, labels=["GUROBI"])
plot!(1:MPC_N, ipopt_timings_mean, color=:purple, labels=["IPOPT"])
plot!(1:MPC_N, cosmo_timings_mean, color=:black, labels=["COSMO"])

savefig("examples/server_heat/output/mpc_simulation.pdf")
savefig("examples/server_heat/output/mpc_simulation.tikz")

fig = plot(
  xlabel = "MPC time step",
  ylabel = "Solver run time [s]",
  fmt = :pdf,
  legend = true
)

plot!(1:MPC_N, model_timings_mean, color=:red, labels=["SPOCK"], ribbon=model_timings_std, fillalpha=0.1)
plot!(1:MPC_N, mosek_timings_mean, color=:blue, labels=["MOSEK"], ribbon=mosek_timings_std, fillalpha=0.1)
plot!(1:MPC_N, gurobi_timings_mean, color=:green, labels=["GUROBI"], ribbon=gurobi_timings_std, fillalpha=0.1)
plot!(1:MPC_N, ipopt_timings_mean, color=:purple, labels=["IPOPT"], ribbon=ipopt_timings_std, fillalpha=0.1)
plot!(1:MPC_N, cosmo_timings_mean, color=:black, labels=["COSMO"], ribbon=cosmo_timings_std, fillalpha=0.1)

savefig("examples/server_heat/output/mpc_simulation_ribbon.pdf")
savefig("examples/server_heat/output/mpc_simulation_ribbon.tikz")