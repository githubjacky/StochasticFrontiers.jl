using Revise
using StochasticFrontiers, Test, CSV, DataFrames, Plots


################################################################################
# Example 1.2: Truncated Normal, vanilla
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/CrossData.csv"), 
        model=:Cross, type=:Prod, dist=Trun(μ=:_cons, σᵤ²=:_cons), σᵥ²=:_cons, 
        depvar=:yvar, frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
    ),
    init=sfinit(log_σᵥ²=-0.1)
);

@test sfmaximizer(res)[1:2] ≈ [0.29315, 0.23998] atol=1e-5
################################################################################


################################################################################
# Example 1.3: Truncated Normal, (BC1995, no age)
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/CrossData.csv"), 
        model=:Cross, type=:Prod, dist=Trun(μ=(:school, :yr, :_cons), σᵤ²=:_cons), σᵥ²=:_cons, 
        depvar=:yvar, frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
    ),
    init=sfinit(μ=ones(3)*0.1)
);

# weird: [0.30298, 0.24794]
@test sfmaximizer(res)[1:2] ≈ [0.30728, 0.24988] atol=1e-5
################################################################################


################################################################################
# Example 1.4: Truncated Normal, BC1995
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/CrossData.csv"), 
        model=:Cross, type=:Prod, dist=Trun(μ=(:age, :school, :yr, :_cons), σᵤ²=:_cons), σᵥ²=:_cons, 
        depvar=:yvar, frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
    ),
    init=sfinit(μ=ones(4)*0.1, log_σᵥ²=-0.1)
);

@test sfmaximizer(res)[1:2] ≈ [0.30298, 0.24794] atol=1e-5 
################################################################################


################################################################################
# Example 2: Truncated Normal, (Wang 2002, no age) 
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/CrossData.csv"), 
        model=:Cross, type=:Prod, dist=Trun(μ=(:school, :yr, :_cons), σᵤ²=(:school, :yr, :_cons)),
        σᵥ²=:_cons, depvar=:yvar, frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
    ),
    init=sfinit(μ=ones(3)*0.1, log_σᵤ²=ones(3)*(-0.1), log_σᵥ²=-0.1)
);

@test sfmaximizer(res)[1:2] ≈ [0.30533, 0.22575] atol=1e-5 
################################################################################


################################################################################
# Example 3: Truncated Normal, Wang 2002
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/CrossData.csv"), 
        model=:Cross, type=:Prod, dist=Trun(μ=(:age, :school, :yr, :_cons), σᵤ²=(:age, :school, :yr, :_cons)),
        σᵥ²=:_cons, depvar=:yvar, frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
    ),
    options=sfopt(warmstart_maxIT=400),
    init=sfinit(μ=ones(4)*0.1, log_σᵤ²=ones(4)*(-0.1), log_σᵥ²=-0.1)
);

@test sfmaximizer(res)[1:2] ≈ [0.25821, 0.17173] atol=1e-5

# efficiency and inefficiency index
plot_inefficieny(res)


# marginal effect
marginal, marginal_mean = sfmarginal(res)
df = DataFrame(CSV.File("examples/data/CrossData.csv"))
plot(
    df.age,
    marginal[:, :marg_age],
    seriestype=:scatter,
    xlabel="age", 
    ylabel="marginal effect of age in E(u)",
    label=false
)
hline!([0.00], label="")


# bootstrap marginal effect
std_ci, bsdata = sfmarginal(
    res,
    bootstrap=true,
    R=100,
    seed=1232,
    iter=100,
    getBootData=true
);

ci = sfCI(bootdata=bsdata, _observed=marginal_mean, level=0.10)
@test [i for i in ci[1]] ≈ [-0.00687, 0.00073] atol=1e-5


_, bsdata = sfmarginal(
    res,
    bootstrap=true,
    R=250,
    seed=123,
    getBootData=true
);


sfCI(bootdata=bsdata, _observed=marginal_mean, level=0.10)
sfCI(bootdata=bsdata, _observed=(-0.00264, -0.01197, -0.0265), level=0.10)
################################################################################