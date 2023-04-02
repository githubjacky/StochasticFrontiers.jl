using Revise
using StochasticFrontiers, Test


# estimate the flexible panel model with serial correlated error
res = sfmodel_fit(
    spec=sfspec(
        SNCre, usedata("examples/data/SNCreData.csv"),
        type=Prod, dist=Half(σᵤ²=:_cons), σᵥ²=:_cons, ivar=:i, depvar=:log_y, 
        frontiers=(:log_x1, :log_x2, :t), serialcorr=AR(1), R=250, σₑ²=:_cons
    ),
    options=sfopt(warmstart_solver=nothing, main_solver=:NewtonTrustRegion), 
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

@test sfmaximizer(res)[1:2] ≈ [0.59928, 0.02075] atol=1e-5
@test sfmaximum(res) ≈ 3531.74644 atol=1e-5


# efficiency and inefficiency index
plot_inefficieny(res)

@test mean(sf_inefficiency(res)) ≈ 0.0447106 atol=1e-5
@test mean(sf_efficiency(res)) ≈ 0.9569270 atol=1e-5