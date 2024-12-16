using SymbolicTracingUtils
using Test: @test, @testset, @test_broken
using LinearAlgebra: Diagonal
using SparseArrays: spzeros, findnz, nnz, rowvals

function dummy_function(x)
    (x .+ 1) .^ 2
end

function dummy_function_gradient(x)
    2 .* (x .+ 1)
end

@testset "SymbolicTracingUtils" begin
    for backend in [SymbolicsBackend(), FastDifferentiationBackend()]
        @testset "$backend" begin
            global x = make_variables(backend, :x, 10)
            global fx = dummy_function(x)
            global x_value = [1:10;]
            @testset "non-ad-tracing" begin
                f = build_function(fx, x; in_place = false)
                f! = build_function(fx, x; in_place = true)
                y_true = dummy_function(x_value)
                y_out_of_place = f(x_value)
                y_in_place = zeros(10)
                f!(y_in_place, x_value)
                @test y_out_of_place ≈ y_true
                @test y_in_place ≈ y_true
            end

            @testset "gradient" begin
                gx = gradient(sum(fx), x)
                g = build_function(gx, x; in_place = false)
                g! = build_function(gx, x; in_place = true)
                y_true = dummy_function_gradient(x_value)
                y_out_of_place = g(x_value)
                y_in_place = zeros(10)
                g!(y_in_place, x_value)
                @test y_out_of_place ≈ y_true
                @test y_in_place ≈ y_true
            end

            @testset "jacobian" begin
                Jx = jacobian(fx, x)
                J = build_function(Jx, x; in_place = false)
                J! = build_function(Jx, x; in_place = true)
                J_true = Diagonal(dummy_function_gradient(x_value))
                J_out_of_place = J(x_value)
                J_in_place = zeros(10, 10)
                J!(J_in_place, x_value)
                if backend isa SymbolicsBackend
                    # see: https://github.com/JuliaSymbolics/Symbolics.jl/issues/1380
                    @test_broken J_out_of_place ≈ J_true
                else
                    @test J_out_of_place ≈ J_true
                end
                @test J_in_place ≈ J_true
            end

            @testset "sparse_jacobian" begin
                Jx = sparse_jacobian(fx, x)
                J = build_function(Jx, x; in_place = false)
                J! = build_function(Jx, x; in_place = true)
                J_true = Diagonal(dummy_function_gradient(x_value))
                J_out_of_place = J(x_value)
                J_in_place = copy(J_out_of_place)
                J!(J_in_place, x_value)
                @test J_out_of_place ≈ J_true
                @test J_in_place ≈ J_true

                @testset "sparsity.jl" begin
                    rows, cols, _ = findnz(Jx)
                    constant_entries = get_constant_entries(Jx, x)
                    @test isempty(constant_entries) # all structural non-zeros are non-constant in this example
                    J_sparse = SparseFunction(J, rows, cols, size(Jx), constant_entries)
                    result = J_sparse(x_value)
                    @test result ≈ J_true
                    @test nnz(J_sparse) == nnz(Jx) # same structure as symbolic version
                    @test rowvals(J_sparse) == rows

                    J_sparse! = SparseFunction(J!, rows, cols, size(Jx), constant_entries)
                    result = get_result_buffer(rows, cols, size(Jx))
                    J_sparse!(result, x_value)
                    @test result ≈ J_true
                    @test nnz(J_sparse!) == nnz(Jx) # same structure as symbolic version
                    @test rowvals(J_sparse!) == rows
                end
            end
        end
    end
end
