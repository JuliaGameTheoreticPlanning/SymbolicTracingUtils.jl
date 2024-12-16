"""
    gradient(f_symbolic, x_symbolic)

Computes the symbolic gradient of `f_symbolic` with respect to `x_symbolic`.
"""
function gradient end

function gradient(f_symbolic::T, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
    Symbolics.gradient(f_symbolic, x_symbolic)
end

function gradient(f_symbolic::T, x_symbolic::Vector{T}) where {T<:FD.Node}
    # FD does not have a gradient utility so we just flatten the jacobian here
    vec(FD.jacobian([f_symbolic], x_symbolic))
end

"""
    jacobian(f_symbolic, x_symbolic)

Computes the symbolic Jacobian of `f_symbolic` with respect to `x_symbolic`.
"""
function jacobian end

function jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
    Symbolics.jacobian(f_symbolic, x_symbolic)
end

function jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:FD.Node}
    FD.jacobian(f_symbolic, x_symbolic)
end

"""
    sparse_jacobian(f_symbolic, x_symbolic)

Computes the symbolic Jacobian of `f_symbolic` with respect to `x_symbolic` in a sparse format.
"""
function sparse_jacobian end

function sparse_jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
    Symbolics.sparsejacobian(f_symbolic, x_symbolic)
end

function sparse_jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:FD.Node}
    FD.sparse_jacobian(f_symbolic, x_symbolic)
end
