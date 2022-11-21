# spock.jl

`spock.jl` is an efficient Julia implementation of the Spock algorithm for multistage risk-averse optimal control problems. It largely benefits from warm-starts and is amenable to massive parallelization.

This solver handles risk-averse optimal control problems with:

  - Linear dynamics
  - Quadratic stage and terminal costs
  - Conic risk measures
  - Convex input-state constraints

## Installation

Make sure to install the dependencies in the `Project.toml` and in your script import

```
include("src/spock.jl")
```


## Examples

The `examples/server_heat` folder contains example code for a test system, modeling the heat of servers in a data center. The figures in the IFAC 2023 submission can be reproduced by running the scripts `residuals.jl`, `scaling.jl` and `mpc_simulation.jl`.
