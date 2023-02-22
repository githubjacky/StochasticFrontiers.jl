using Test, Distributions, StochasticFrontiers


@testset "Cross-section(Half) of log-likelihood, jlms & bc index, marginal effect" begin
    res = sfmodel_fit(
        spec=sfspec(
            Cross, usedata("data/CrossData.csv"), 
            type=Prod, dist=Trun(μ=(:age, :school, :yr, :_cons), σᵤ²=(:age, :school, :yr, :_cons)),
            σᵥ²=:_cons, depvar=:yvar, frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
        ),
        options=sfopt(warmstart_maxIT=400, verbose=false),
        init=sfinit(log_σᵤ²=(-0.1, -0.1, -0.1, -0.1), log_σᵥ²=-0.1)
    )
    marginal, marginal_mean = sfmarginal(res)

    @test res.loglikelihood ≈ -82.02573 atol=1e-5
    @test mean(res.jlms) ≈ 0.33416 atol=1e-5
    @test mean(res.bc) ≈ 0.7462 atol=1e-5
    @test values(marginal_mean)[1] ≈ -0.0026449 atol=1e-5
end


@testset "SNCre of log-likelihood, jlms & bc index" begin
    res = sfmodel_fit(
        spec=sfspec(
            SNCre, usedata("data/SNCreData.csv"),
            type=Prod, dist=Half(σᵤ²=:_cons), σᵥ²=:_cons, ivar=:i, depvar=:log_y, 
            frontiers=(:log_x1, :log_x2, :t), serialcorr=AR(1), R=250, σₑ²=:_cons
        ),
        options=sfopt(warmstart_solver=nothing, main_solver=NewtonTrustRegion, verbose=false), 
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
    );

    @test res.loglikelihood ≈ 3531.74644 atol=1e-5
    @test mean(res.jlms) ≈ 0.0447106 atol=1e-5
    @test mean(res.bc) ≈ 0.9569270 atol=1e-5
end
