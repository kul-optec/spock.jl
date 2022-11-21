abstract type ConvexConstraints end

"""
Impose a uniform constraint on all nodes

- x_min
- x_max
- u_min 
- u_max
"""
struct UniformRectangle{TF <: Real, TI <: Integer} <: ConvexConstraints
  x_min :: TF
  x_max :: TF
  u_min :: TF
  u_max :: TF
  nΓ_leaf :: TI
  nΓ_nonleaf :: TI
  nx :: TI
  nu :: TI
  n_leafs :: TI
  n_nonleafs :: TI
end

function project_onto_leaf_constraints!(rec :: UniformRectangle{TF}, x :: AbstractArray{TF, 1}) where {TF <: Real}
  for i = 1:rec.n_leafs
    for j = 1:rec.nx
      k = (i - 1) * rec.nx + j
      x[k] = x[k] > rec.x_max ? rec.x_max : (
          x[k] < rec.x_min ? rec.x_min : x[k]
        )
    end
  end
end

"""
  This function assumes the vector xu to be defined as ((x_i), (u_i))_i
"""
function project_onto_nonleaf_constraints!(rec :: UniformRectangle{TF}, xu :: AbstractArray{TF, 1}) where {TF <: Real}
  n_xu = rec.nx + rec.nu
  for i = 1:rec.n_nonleafs
    for j = 1:rec.nx
      k = (i - 1) * n_xu + j
      xu[k] = xu[k] > rec.x_max ? rec.x_max : (
          xu[k] < rec.x_min ? rec.x_min : xu[k]
      )
    end
    for j = 1:rec.nu
      k = (i - 1) * n_xu + rec.nx + j
      xu[k] = xu[k] > rec.u_max ? rec.u_max : (
        xu[k] < rec.u_min ? rec.u_min : xu[k]
      )
    end
  end
end

"""
  This function assumes v = ((x_i), (u_i))_i
  and sets x equal to the x_i components of v
"""
function update_nonleaf_constraints_primal_x!(
  rec :: UniformRectangle{TF},
  x :: AbstractArray{TF, 1},
  v :: AbstractArray{TF, 1}
) where {TF <: Real}
  n_xu = rec.nx + rec.nu
  for i = 1:rec.n_nonleafs
    for j = 1:rec.nx
      k = (i - 1) * n_xu + j
      x[(i - 1) * rec.nx + j] = v[k]
    end
  end
end

"""
  This function assumes v = ((x_i), (u_i))_i
  and sets u equal to the u_i components of v
"""
function update_nonleaf_constraints_primal_u!(
  rec :: UniformRectangle{TF},
  u :: AbstractArray{TF, 1},
  v :: AbstractArray{TF, 1}
) where {TF <: Real}
  n_xu = rec.nx + rec.nu
  for i = 1:rec.n_nonleafs
    for j = 1:rec.nu
      k = (i - 1) * n_xu + rec.nx + j
      u[(i - 1) * rec.nu + j] = v[k]
    end
  end
end

"""
  This function assumes v = (x_i)_i
  and sets x equal to the x_i components
"""
function update_leaf_constraints_primal!(
  rec :: UniformRectangle{TF},
  x :: AbstractArray{TF, 1},
  v :: AbstractArray{TF, 1}
) where {TF <: Real}
  copyto!(x, v)
end

function update_leaf_constraints_dual!(rec :: UniformRectangle{TF}, v :: AbstractArray{TF, 1}, x :: AbstractArray{TF, 1}) where {TF <: Real}
  copyto!(v, x)
end

"""
  Define v = ((x_i), (u_i))_i
"""
function update_nonleaf_constraints_dual!(
  rec :: UniformRectangle{TF}, 
  v :: AbstractArray{TF, 1}, 
  x :: AbstractArray{TF, 1}, 
  u :: AbstractArray{TF, 1}
) where {TF <: Real}
  n_xu = rec.nx + rec.nu
  for i = 1:rec.n_nonleafs
    for j = 1:rec.nx
      k = (i - 1) * n_xu + j
      v[k] = x[(i - 1) * rec.nx + j]
    end
    for j = 1:rec.nu
      k = (i - 1) * n_xu + rec.nx + j
      v[k] = u[(i - 1) * rec.nu + j]
    end
  end
end