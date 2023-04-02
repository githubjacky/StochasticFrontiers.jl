using Revise
using StochasticFrontiers, Test


################################################################################
# Example 1: Half Normal, vanilla
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/CrossData.csv"), 
        model=:Cross, type=:Prod, dist=Half(σᵤ²=(:age, :school, :yr, :_cons)),
        σᵥ²=:_cons, depvar=:yvar, frontiers=(:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
    )
);

@test sfmaximizer(res)[1:2] ≈ [0.29488, 0.23191] atol=1e-5
@test sfmarginal(res)[1][!, :marg_age][1:2] ≈ [-0.00147, -0.00135] atol=1e-5

std_ci, bsdata = sfmarginal(
    res,
    bootstrap=true,
    R=20,
    seed=123,
    getBootData=true
);
@test std_ci[1:3, 1] ≈ [0.00263, 0.01180, 0.01377] atol=1e-5
@test bsdata[1, 1:3] ≈ [-0.00015, 0.00451, -0.03662] atol=1e-5
################################################################################


################################################################################
# Example 4: Exponential (Stata: chapter3.do, Model 10) 
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/dairy.csv"), 
        model=:Cross, type=:Prod, dist=Expo(λ=(:comp, :_cons)),
        σᵥ²=:_cons, depvar=:ly, frontiers=(:llabor, :lfeed, :lcattle, :lland, :_cons)
    ),
    options=sfopt(warmstart_solver=nothing, main_maxIT=5000),
    init=sfinit(log_λ²=[-0.1, -0.1], log_σᵥ²=-0.1)
);

@test sfmaximizer(res)[1:2] ≈ [0.09655, 0.14869] atol=1e-5
@test sf_inefficiency(res)[1:2] ≈ [0.04817, 0.10286] atol=1e-5
@test sf_efficiency(res)[1:2] ≈ [0.95375, 0.90431] atol=1e-5

marginal, marginal_mean = sfmarginal(res)

@test marginal.marg_comp[1:2] ≈ [-0.00028, -0.00028] atol=1e-5
################################################################################