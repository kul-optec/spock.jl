import LinearAlgebra as LA

using spock, Random, Plots

include("server_heat.jl")

pgfplotsx()

# Prediction horizon and branching factor
N = 7; d = 2

# Dimensions of state and input vectors
nx = 10

TOL = 1e-5

x0 = [0.1 for i = 1:nx]

scen_tree, cost, dynamics, rms, constraints = get_server_heat_specs(N, nx, d)

spock.writedlm("examples/output/xi_sp.dat", "")
spock.writedlm("examples/output/xi1_sp.dat", "")
spock.writedlm("examples/output/xi2_sp.dat", "")
spock.writedlm("examples/output/xi_backtrack_count.dat", "")
sp_model = spock.build_model(scen_tree, cost, dynamics, rms, constraints, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))
spock.solve_model!(sp_model, x0, tol=TOL, verbose=spock.LOG)

spock.writedlm("examples/output/xi_cp.dat", "")
spock.writedlm("examples/output/xi1_cp.dat", "")
spock.writedlm("examples/output/xi2_cp.dat", "")
cp_model = spock.build_model(scen_tree, cost, dynamics, rms, constraints, spock.SolverOptions(spock.L_IMPLICIT, spock.CP))
spock.solve_model!(cp_model, x0, tol=TOL, verbose=spock.LOG)

fig = plot(
  xlabel = "# calls of L",
  ylabel = "Residual value ξ",
  fmt = :pdf,
  legend = true
)

xi_sp = spock.readdlm("examples/output/xi_sp.dat")
xi1_sp = spock.readdlm("examples/output/xi1_sp.dat")
xi2_sp = spock.readdlm("examples/output/xi2_sp.dat")
backtrack_count = spock.readdlm("examples/output/xi_backtrack_count.dat")
calls = backtrack_count .+ 2
xi = [0. for _ in 1:sum(calls) ]
s = 0
for (i, count) in enumerate(calls)
  for j = 1:count
    xi[s + Int64(j)] = xi_sp[i]
  end
  global s += Int64(count)
end

xi_cp = spock.readdlm("examples/output/xi_cp.dat")
xi1_cp = spock.readdlm("examples/output/xi1_cp.dat")
xi2_cp = spock.readdlm("examples/output/xi2_cp.dat")

plot!(xi_cp, color=:red, yaxis=:log, labels=["CP",])
plot!(xi, color=:blue, yaxis=:log, labels=["SPOCK",])

savefig("examples/server_heat/output/residuals.pdf")
savefig("examples/server_heat/output/residuals.tikz")