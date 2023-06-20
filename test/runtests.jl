using StochasticFrontiers
using Test, Distributions


@testset verbose=true "StochasticFrontiers.jl Tests" begin
    @testset verbose=true "Example 1.1: Half Normal, vanilla" begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/CrossData.csv"), 
                model     = :Cross, 
                type      = :Prod, 
                dist      = Half(σᵤ² =(:age, :school, :yr, :_cons)),
                σᵥ²       = :_cons, 
                depvar    = :yvar, 
                frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
            ),
            options = sfopt(verbose = false)
        )

        @test sfmaximizer(res)[1:2]                                ≈ [0.29488, 0.23191]   atol=1e-5
        @test sfmarginal(res, verbose=false)[1][!, :marg_age][1:2] ≈ [-0.00147, -0.00135] atol=1e-5

        std_ci, bsdata = sfmarginal(
            res,
            bootstrap   = true,
            R           = 20,
            seed        = 123,
            getBootData = true,
            verbose     = false
        );
        @test std_ci[1:3, 1] ≈ [0.00263, 0.01180, 0.01377]   atol = 1e-5
        @test bsdata[1, 1:3] ≈ [-0.00015, 0.00451, -0.03662] atol = 1e-5
    end


    @testset verbose=true "Example 1.2: Truncated Normal, vanilla" begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/CrossData.csv"), 
                model     = :Cross, 
                type      = :Prod, 
                dist      = Trun(μ=:_cons, σᵤ²=:_cons), 
                σᵥ²       = :_cons, 
                depvar    = :yvar, 
                frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
            ),
            options = sfopt(verbose  = false),
            init    = sfinit(log_σᵥ² = -0.1 )
        )

        @test sfmaximizer(res)[1:2] ≈ [0.29315, 0.23998] atol = 1e-5
    end


    @testset verbose=true "Example 1.3: Truncated Normal, (BC1995, no age)" begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/CrossData.csv"), 
                model     = :Cross, 
                type      = :Prod, 
                dist      = Trun(μ=(:school, :yr, :_cons), σᵤ²=:_cons), 
                σᵥ²       = :_cons, 
                depvar    = :yvar, 
                frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
            ),
            options = sfopt(verbose = false),
            init    = sfinit(μ = fill(0.1, 3))
        )

        # weird, commented in 00test_pkg.jl: [0.30298, 0.24794]
        @test sfmaximizer(res)[1:2] ≈ [0.30728, 0.24988] atol = 1e-5
    end


    @testset verbose=true "Example 1.4: Truncated Normal, BC1995" begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/CrossData.csv"), 
                model     = :Cross, 
                type      = :Prod, 
                dist      = Trun(μ = (:age, :school, :yr, :_cons), σᵤ² = :_cons), 
                σᵥ²       = :_cons, 
                depvar    = :yvar, 
                frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
            ),
            options = sfopt(verbose = false),
            init = sfinit(
                μ       = fill(0.1, 4), 
                log_σᵥ² = -0.1
            )
        )

        @test sfmaximizer(res)[1:2] ≈ [0.30298, 0.24794] atol = 1e-5 
    end


    @testset verbose=true "Example 2  : Truncated Normal, (Wang, 2002, no age)" begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/CrossData.csv"), 
                model     = :Cross, 
                type      = :Prod, 
                dist      = Trun(μ = (:school, :yr, :_cons), σᵤ² = (:school, :yr, :_cons)),
                σᵥ²       = :_cons, 
                depvar    = :yvar, 
                frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
            ),
            options = sfopt(verbose = false),
            init = sfinit(
                μ       = fill(0.1, 3), 
                log_σᵤ² = fill(-0.1, 3), 
                log_σᵥ² = -0.1
            )
        )

        @test sfmaximizer(res)[1:2] ≈ [0.30533, 0.22575] atol = 1e-5 
    end


    @testset verbose=true "Example 3  : Truncated Normal, Wang 2002" begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/CrossData.csv"), 
                model=:Cross, 
                type=:Prod, 
                dist=Trun(μ=(:age, :school, :yr, :_cons), σᵤ²=(:age, :school, :yr, :_cons)),
                σᵥ²      =:_cons, 
                depvar    = :yvar, 
                frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
            ),
            options = sfopt(
                warmstart_maxIT = 400, 
                verbose         = false
            ),
            init = sfinit(
                μ       = fill(0.1, 4), 
                log_σᵤ² = fill(-0.1, 4), 
                log_σᵥ² = -0.1
            )
        )

        marginal, marginal_mean = sfmarginal(res)
        std_ci, bsdata = sfmarginal(
            res,
            bootstrap   = true,
            R           = 100,
            seed        = 1232,
            iter        = 100,
            getBootData = true,
            verbose     = false
        )
        ci = sfCI(
            bootdata  = bsdata, 
            _observed = marginal_mean, 
            level     = 0.10, 
            verbose   = false
        )

        @test sfmaximizer(res)[1:2] ≈ [0.25821, 0.17173] atol = 1e-5
        @test ci[1] ≈ [-0.00687, 0.00073] atol = 1e-5
    end


    @testset verbose=true "Example 4  : Exponential, (Stata: chapter3.do, Model 10)"  begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/dairy.csv"), 
                model     = :Cross, 
                type      = :Prod, 
                dist      = Expo(λ=(:comp, :_cons)),
                σᵥ²       = :_cons, 
                depvar    = :ly, 
                frontiers = (:llabor, :lfeed, :lcattle, :lland, :_cons)
            ),
            options = sfopt(
                warmstart_solver = nothing, 
                main_maxIT       = 5000, 
                verbose          = false
            ),
            init = sfinit(
                log_λ²  = fill(-0.1, 2), 
                log_σᵥ² = -0.1
            )
        )

        marginal, marginal_mean = sfmarginal(res)

        @test sfmaximizer(res)[1:2]     ≈ [0.09655, 0.14869]   atol = 1e-5
        @test sf_inefficiency(res)[1:2] ≈ [0.04817, 0.10286]   atol = 1e-5
        @test sf_efficiency(res)[1:2]   ≈ [0.95375, 0.90431]   atol = 1e-5
        @test marginal.marg_comp[1:2]   ≈ [-0.00028, -0.00028] atol = 1e-5
    end


    @testset verbose=true "Example 6  : panel FE model of Wang and Ho, (Half Normal)" begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/WangHo2010_data.csv"),
                model     = :PFEWH, 
                type      = :Prod, 
                dist      = Half(σᵤ² = :_cons), 
                σᵥ²       = :_cons, 
                hscale    = (:z1, :z2),
                ivar      = :cnum, 
                depvar    = :y, 
                frontiers = (:x1, :x2, :x3)
            ),
            options = sfopt(
                warmstart_maxIT  = 100,
                warmstart_solver = :NewtonTrustRegion, 
                main_solver      = :NewtonTrustRegion,
                verbose          = false
            ), 
        )

        marginal, marginal_mean = sfmarginal(res)

        @test sfmaximizer(res)[1:2] ≈ [0.65495, 0.15002] atol = 1e-5
        @test marginal.marg_z1[1:2] ≈ [0.00261, 0.00252] atol = 1e-5  # failed 
    end


    @testset verbose=true "Example 7  : panel FE model of Wang and Ho (Truncated Normal)" begin
        res = sfmodel_fit(
            spec=sfspec(
                usedata("data/WH2010T.csv"),
                model     = :PFEWH, 
                type      = :Prod, 
                dist      = Trun(μ = :_cons, σᵤ² = :_cons), 
                σᵥ²       = :_cons, 
                hscale    = :zit, 
                ivar      = :id, 
                depvar    = :yit, 
                frontiers = (:xit,)
            ),
            options=sfopt(
                warmstart_maxIT  = 200,
                warmstart_solver = :NelderMead, 
                main_solver      = :Newton,
                verbose          = false
            ), 
        )

        marginal, marginal_mean = sfmarginal(res)
        std_ci = sfmarginal(
            res,
            bootstrap = true,
            R         = 10,
            seed      = 123,
            verbose   = false
        )

        @test sfmaximizer(res)[1:2]     ≈ [0.49727, 0.79545] atol = 1e-5
        @test sf_inefficiency(res)[1:2] ≈ [3.61393, 4.53487] atol = 1e-5
        @test sf_efficiency(res)[1:2]   ≈ [0.02811, 0.01146] atol = 1e-5
        @test marginal.marg_zit[1:2]    ≈ [1.28355, 1.61064] atol = 1e-5
        @test std_ci[1]                 ≈ 0.03323            atol = 1e-5 
        # @test std_ci[1]                 ≈ 0.03087            atol = 1e-5
    end


    @testset verbose=true "Example 12 : panel RE model with serial correlated error" begin
        res = sfmodel_fit(
            spec = sfspec(
                usedata("data/SNCreData.csv"),
                model      = :SNCre, 
                type       = :Prod, 
                dist       = Half(σᵤ²=:_cons), 
                σᵥ²        = :_cons, 
                ivar       = :i, 
                depvar     = :log_y, 
                frontiers  = (:log_x1, :log_x2, :t), 
                serialcorr = AR(1), 
                R          = 250, 
                σₑ²        = :_cons,
                verbose    = false
            ),
            options = sfopt(
                warmstart_solver = nothing, 
                main_solver      = :NewtonTrustRegion, 
                verbose          = false
            ), 
            init=sfinit([
                0.5227960098102571,        # coefficient of explanatory variables
                0.1868939866287993,        # coefficient of explanatory variables
                0.007442174221837823,      # coefficient of time effect
                -1.6397116052113527*2,     # log_σᵤ²
                -3.3244812689250423*2,     # log_σᵥ²
                0.3484365793340449,        # coefficient of fixed effect(mean of x1)
                -0.05768082007032795,      # coefficient of fixed effect2(mean of x2)
                -0.5943654485109733/14.5,  # costant term of fixed effect3(mean of x3)
                -0.8322378217931871*2,     # log_σₑ²
                0.,
            ])
        )

        @test sfmaximizer(res)[1:2]      ≈ [0.59928, 0.02075] atol = 1e-5
        @test sfmaximum(res)             ≈ 3531.74644         atol = 1e-5
        @test mean(sf_inefficiency(res)) ≈ 0.0447106          atol = 1e-5
        @test mean(sf_efficiency(res))   ≈ 0.9569270          atol = 1e-5
    end
end  # end of the test
