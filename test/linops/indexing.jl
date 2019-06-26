using Random, LinearAlgebra
using Stheno: GPC

@testset "indexing" begin
    @testset "no noise" begin
        f, x = GP(eq(), GPC()), randn(17)
        fx = f(x)
        @test mean(fx) == ew(mean(f), x)
        @test cov(fx) == pairwise(kernel(f), x)
    end
    @testset "(Symmetric) Matrix-valued noise" begin
        rng, N = MersenneTwister(123456), 13
        f, x, A = GP(eq(), GPC()), randn(rng, N), randn(rng, N, N)
        C = Symmetric(A * A' + I)
        fx = f(x, C)
        @test mean(fx) == ew(mean(f), x)
        @test cov(fx) == pairwise(kernel(f), x) + C
    end
    @testset "Vector-valued noise" begin
        rng, N = MersenneTwister(123456), 11
        f, x, a = GP(eq(), GPC()), randn(rng, N), exp.(randn(rng, N))
        fx = f(x, a)
        @test mean(fx) == ew(mean(f), x)
        @test cov(fx) == pairwise(kernel(f), x) + Diagonal(a)
    end
    @testset "Real-valued noise" begin
        rng, N = MersenneTwister(123456), 13
        f, x, σ² = GP(eq(), GPC()), randn(rng, N), exp(randn(rng))
        fx = f(x, σ²)
        @test mean(fx) == ew(mean(f), x)
        @test cov(fx) == pairwise(kernel(f), x) + Diagonal(fill(σ², N))
    end
end
