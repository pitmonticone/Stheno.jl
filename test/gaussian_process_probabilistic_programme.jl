@timedtestset "gaussian_process_probabilistic_programme" begin

    @timedtestset "split" begin
        x = BlockData(randn(5), randn(4))
        @testset "Vector" begin
            x1, x2 = split(x, randn(9))
            @test length(x1) == 5
            @test length(x2) == 4
        end
        @testset "Matrix" begin
            x1, x2 = split(x, randn(9, 3))
            @test size(x1) == (5, 3)
            @test size(x2) == (4, 3)
        end
    end

    # Build a toy collection of processes.
    gpc = GPC()
    f1 = wrap(GP(sin, SEKernel()), gpc)
    f2 = wrap(GP(cos, Matern52Kernel()), gpc)
    f3 = f1 + 3 * f2

    # Use them to build a programme.
    f = Stheno.GPPP((f1 = f1, f2 = f2, f3 = f3), gpc)

    # The same answers should be obtained manually or via the GPPP.
    @timedtestset "External Consistency" begin

        x0 = GPPPInput(:f1, randn(4))
        x1 = GPPPInput(:f3, randn(3))

        @test mean(f1, x0.x) == mean(f, x0)
        @test mean(f3, x1.x) == mean(f, x1)

        @test cov(f1, x0.x) == cov(f, x0)
        @test cov(f3, x1.x) == cov(f, x1)

        @test cov(f1, f3, x0.x, x1.x) == cov(f, x0, x1)
        @test var(f3, f1, x1.x, x0.x) == var(f, x1, x0)

        y = rand(f(x1))
        @test cov(posterior(f3(x1.x), y)(x1.x)) == cov(posterior(f(x1), y)(x1))
    end

    # The GPPP must be self-consistent like any other AbstractGP.
    # This should hold for all of the various permutations of applicable input types.
    @testset "Internal Conistency ($(typeof(x0)), $(typeof(x1))" for (x0, x1) in [
        (
            GPPPInput(:f1, randn(4)),
            GPPPInput(:f3, randn(3)),
        ),
        (
            GPPPInput(:f1, randn(4)),
            BlockData([GPPPInput(:f2, randn(3)), GPPPInput(:f3, randn(2))]),
        ),
        (
            BlockData([GPPPInput(:f2, randn(3)), GPPPInput(:f3, randn(2))]),
            GPPPInput(:f1, randn(4)),
        ),
        (
            BlockData([GPPPInput(:f2, randn(3)), GPPPInput(:f3, randn(2))]),
            BlockData([GPPPInput(:f1, randn(6))]),
        ),
        (
            collect(GPPPInput(:f1, randn(4))),
            collect(GPPPInput(:f3, randn(3))),
        ),
        (
            GPPPInput(:f1, randn(4)),
            collect(GPPPInput(:f3, randn(3))),
        ),
        (
            collect(BlockData([GPPPInput(:f2, randn(3)), GPPPInput(:f3, randn(2))])),
            collect(GPPPInput(:f1, randn(4))),
        ),
        (
            collect(BlockData([GPPPInput(:f2, randn(3)), GPPPInput(:f3, randn(2))])),
            GPPPInput(:f1, randn(4)),
        ),
        (
            BlockData([collect(GPPPInput(:f2, randn(3))), GPPPInput(:f3, randn(2))]),
            GPPPInput(:f1, randn(4)),
        ),
    ]
        test_internal_abstractgps_interface(MersenneTwister(123456), f, x0, x1)
    end

    @timedtestset "gppp macro" begin

        # Declare a GPPP using the helper functionality.
        f = @gppp let
            f1 = GP(SEKernel())
            f2 = GP(Matern52Kernel())
            f3 = f1 + f2
        end
    end

    # No custom rules to worry about, just need to make sure that nothing errors.
    @timedtestset "Zygote" begin
        x = GPPPInput(:f3, randn(5))
        s = 0.1
        y = rand(f(x, s))
        Zygote.gradient((x, y, f, s) -> logpdf(f(x, s), y), x, y, f, s)
    end

    # Check that we can use one GPPP inside another.
    @timedtestset "nested gppp" begin

        gpc_outer = GPC()
        f1_outer = Stheno.wrap(f, gpc_outer)
        f2_outer = 5 * f1_outer
        f_outer = Stheno.GPPP((f1=f1_outer, f2=f2_outer), gpc_outer)

        x0 = GPPPInput(:f1, randn(5))
        x1 = GPPPInput(:f2, randn(4))
        x0_outer = GPPPInput(:f1, x0)
        x1_outer = GPPPInput(:f2, x1)
        rng = MersenneTwister(123456)
        test_internal_abstractgps_interface(rng, f_outer, x0_outer, x1_outer)
    end
end
