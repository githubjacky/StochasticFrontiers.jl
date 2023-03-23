using CSV, DataFrames, BenchmarkTools, Plots, Revise
using StochasticFrontiers


# estimate the flexible panel model with serial correlated error
res = sfmodel_fit(
    spec=sfspec(
        SNCre, usedata("test/data/SNCreData.csv"),
        type=Prod, dist=Half(σᵤ²=:_cons), σᵥ²=:_cons, ivar=:i, depvar=:log_y, 
        frontiers=(:log_x1, :log_x2, :t), serialcorr=AR(1), R=250, σₑ²=:_cons
    ),
    options=sfopt(warmstart_solver=nothing, main_solver=NewtonTrustRegion), 
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

# the results of matlab evaluation
# AR1ans = [0.5984016335288649,      # β₁
#           0.0255450270909429,      # β₂
#           -0.015445655766062863,   # slope
#           -2.841400375123482*2,    # Wᵤ
#           -3.3505623313661395*2,   # Wᵥ
#           0.2709332839924433,      # δ₁
#           0.08281672954904168,     # δ₂
#           2.156447004206457/14.5,  # δ₀
#           -1.6672813106398652*2,   # Wₑ
#           0.9732931813001965]      # ρ

# @show sum(abs.(AR1ans .- res.ξ))


# efficiency and inefficiency index
plot(
    histogram(res.jlms, xlabel="JLMS", bins=100, label=false),
    histogram(res.bc, xlabel="BC", bins=50, label=false),
    layout = (1,2), legend=false
)