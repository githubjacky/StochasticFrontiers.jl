using CSV, DataFrames, BenchmarkTools, Plots, Revise
using StochasticFrontiers


# estimate the basic stochastic frontiers model
res = sfmodel_fit(
    spec=sfspec(
        Cross, usedata("test/data/CrossData.csv"), 
        type=Prod, dist=Trun(μ=(:age, :school, :yr, :_cons), σᵤ²=(:age, :school, :yr, :_cons)),
        σᵥ²=:_cons, depvar=:yvar, frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
    ),
    options=sfopt(warmstart_maxIT=400),
    init=sfinit(log_σᵤ²=(-0.1, -0.1, -0.1, -0.1), log_σᵥ²=-0.1)
);


# efficiency and inefficiency index
plot(
    histogram(sf_inefficiency(res), xlabel="JLMS", bins=100, label=""),
    histogram(sf_efficiency(res), xlabel="BC", bins=50, label=""),
    layout = (1,2), legend=false
)


# marginal effect
marginal, marginal_mean = sfmarginal(res)
df = DataFrame(CSV.File("test/data/CrossData.csv"))
plot(
    df[:,:age],
    marginal[:, :marg_age],
    seriestype=:scatter,
    xlabel="age", 
    ylabel="marginal effect of age in E(u)",
    label=false
)
hline!([0.00], label = false)


# bootstrap marginal effect
err_ci, bsdata = sfmarginal(
    res,
    bootstrap=true,
    R=100,
    seed=1232,
    iter=100,
    getBootData=true
);