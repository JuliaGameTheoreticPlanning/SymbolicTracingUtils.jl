"""
    make_variables(backend, name, dimension)

Creates a vector of `dimension` where each element is a scalar symbolic variable from `backend` with the given `name`.
"""
function make_variables end

function make_variables(::SymbolicsBackend, name::Symbol, dimension::Int)
    vars = Symbolics.@variables($name[1:dimension]) |> only |> Symbolics.scalarize

    if isempty(vars)
        vars = Symbolics.Num[]
    end

    vars
end

function make_variables(::FastDifferentiationBackend, name::Symbol, dimension::Int)
    FD.make_variables(name, dimension)
end

"""
    build_function(f_symbolic, args_symbolic...; in_place, options)

Builds a callable function from a symbolic expression `f_symbolic` with the given `args_symbolic` as arguments.

Depending on the `in_place` flag, the function will be built as in-place `f!(result, args...)` or out-of-place variant `restult = f(args...)`.

`backend_options` will be forwarded to the backend specific function and differ between backends.
"""
function build_function end

# scalar fallback
function build_function(f_symbolic::T, args_symbolic...; backend_options = (;)) where {T<:FD.Node}
    f_callable! =
        build_function([f_symbolic], args_symbolic...; in_place = true, backend_options...)

    let output = [0.0]
        function (x)
            f_callable!(output, x)
            only(output)
        end
    end
end

function build_function(
    f_symbolic::AbstractArray{T},
    args_symbolic...;
    in_place,
    backend_options = (;),
) where {T<:Symbolics.Num}
    f_callable, f_callable! = Symbolics.build_function(
        f_symbolic,
        args_symbolic...;
        expression = Val{false},
        # slightly saner defaults...
        (; parallel = Symbolics.ShardedForm(), backend_options...)...,
    )

    in_place ? f_callable! : f_callable
end

function build_function(
    f_symbolic::AbstractArray{T},
    args_symbolic...;
    in_place,
    backend_options = (;),
) where {T<:FD.Node}
    f = FD.make_function(f_symbolic, args_symbolic...; in_place, backend_options...)

    if in_place
        function (result, args...)
            f(result, reduce(vcat, args))
        end
    else
        function (args...)
            f(reduce(vcat, args))
        end
    end
end

"""
Build a linear SciMLOperators.FunctionOperator from a matrix-valued function `A(p)`
to represent the matrix-vector product `A(p) * u` in matrix-free form.
"""
function build_linear_operator(A_of_p::AbstractMatrix{<:SymbolicNumber}, p; in_place)
    u = make_variables(infer_backend(A_of_p), gensym(), size(A_of_p)[end])
    A_of_p_times_u = build_function(A_of_p * u, p, u; in_place)
    # TODO: also analyze symmetry and other matrix properties to forward to the operator
    input_prototype = zeros(size(u))
    p_prototype = zeros(size(p))

    if in_place
        FunctionOperator(input_prototype; p = p_prototype, islinear = true) do result, u, p, _t
            A_of_p_times_u(result, p, u)
        end
    else
        FunctionOperator(input_prototype; p = p_prototype, islinear = true) do u, p, _t
            A_of_p_times_u(p, u)
        end
    end
end
