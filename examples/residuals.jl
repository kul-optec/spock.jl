import LinearAlgebra as LA

using spock, Random, Plots

pgfplotsx()

Random.seed!(99)

# Prediction horizon and branching factor
N = 10; d = 2

# Dimensions of state and input vectors
nx = 2; nu = 1

# Scenario tree definition
scen_tree = spock.generate_scenario_tree_uniform_branching_factor_v2(N, d, nx, nu)

# Cost definition (Quadratic, positive definite)
cost = spock.CostV2(
  # Q matrices
  collect([
    LA.Matrix([(2.2) 0; 0 (3.7)]) / 3.7 for i in 1:scen_tree.n - 1
  ]),
  # R matrices
  collect([
    reshape([(3.2)], 1, 1) / 3.7 for i in 1:scen_tree.n - 1
  ]),
  # QN matrices
  collect([
    LA.Matrix([(2.2) 0; 0 (3.7)]) / 3.7 for i in 1:scen_tree.n_leaf_nodes
  ])
)

  # Cost definition (Quadratic, positive definite)
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
    map(x -> x' * x, Qs),
    # R matrices
    map(x -> x' * x, Rs),
    # QN matrices
    map(x -> x' * x, QNs)
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

factor = 1e0

spock.writedlm("examples/output/xi_sp.dat", "")
spock.writedlm("examples/output/xi1_sp.dat", "")
spock.writedlm("examples/output/xi2_sp.dat", "")
spock.writedlm("examples/output/xi_backtrack_count.dat", "")
sp_model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.SP))
spock.solve_model!(sp_model, [0.1, .1] / factor, tol=1e-3 / factor, verbose=spock.LOG)

spock.writedlm("examples/output/xi_cp.dat", "")
spock.writedlm("examples/output/xi1_cp.dat", "")
spock.writedlm("examples/output/xi2_cp.dat", "")
cp_model = spock.build_model(scen_tree, cost, dynamics, rms, spock.SolverOptions(spock.L_IMPLICIT, spock.CP))
spock.solve_model!(cp_model, [0.1, .1] / factor, tol=1e-3 / factor, verbose=spock.LOG)

fig = plot(
  xlabel = "# calls of L",
  ylabel = "Residual value ξ", # TODO: Latexstrings
  fmt = :pdf,
  legend = true
)

xi_sp = spock.readdlm("examples/output/xi_sp.dat")
xi1_sp = spock.readdlm("examples/output/xi1_sp.dat")
xi2_sp = spock.readdlm("examples/output/xi2_sp.dat")
backtrack_count = spock.readdlm("examples/output/xi_backtrack_count.dat")
backtrack_count = backtrack_count .+ 1
xi = [0. for _ in 1:sum(backtrack_count) ]
s = 0
for (i, count) in enumerate(backtrack_count)
  for j = 1:count
    xi[s + Int64(j)] = xi_sp[i]
  end
  global s += Int64(count)
end

xi_cp = spock.readdlm("examples/output/xi_cp.dat")
xi1_cp = spock.readdlm("examples/output/xi1_cp.dat")
xi2_cp = spock.readdlm("examples/output/xi2_cp.dat")

plot!(xi_cp, color=:red, yaxis=:log, labels=["cp",])
plot!(xi, color=:blue, yaxis=:log, labels=["cp + sp",])

savefig("examples/output/residuals.pdf")
savefig("examples/output/residuals.tikz")