module FastDifferentiationExt

using FastDifferentiation: FastDifferentiation as FD

function make_variables(::FastDifferentiationBackend, name::Symbol, dimension::Int)
  FD.make_variables(name, dimension)
end

function build_function(
  f_symbolic::AbstractArray{T},
  args_symbolic...;
  in_place,
  backend_options=(;),
) where {T<:FD.Node}
  FD.make_function(f_symbolic, args_symbolic...; in_place, backend_options...)
end

function gradient(f_symbolic::T, x_symbolic::Vector{T}) where {T<:FD.Node}
  # FD does not have a gradient utility so we just flatten the jacobian here
  vec(FD.jacobian([f_symbolic], x_symbolic))
end

function jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:FD.Node}
  FD.jacobian([f_symbolic], x_symbolic)
end

function sparse_jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:FD.Node}
  FD.sparse_jacobian(f_symbolic, x_symbolic)
end

end
