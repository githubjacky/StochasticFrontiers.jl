using Revise
using StochasticFrontiers, Test

################################################################################
# Example 6: panel FE model of Wang and Ho, (Half Normal)
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/WangHo2010_data.csv"),
        model=:PFEWH, type=:Prod, dist=Half(σᵤ²=:_cons), σᵥ²=:_cons, hscale=(:z1, :z2),
        ivar=:cnum, depvar=:y, frontiers=(:x1, :x2, :x3)
    ),
    options=sfopt(
        warmstart_maxIT=100,
        warmstart_solver=:NewtonTrustRegion, 
        main_solver=:NewtonTrustRegion
    ), 
);

@test sfmaximizer(res)[1:2] ≈ [0.65495, 0.15002] atol=1e-5

marginal, marginal_mean = sfmarginal(res)
@test marginal.marg_z1[1:2] ≈ [0.00261,0.00252] atol=1e-5
################################################################################


################################################################################
# Example 7: panel FE model of Wang and Ho, (Truncated Normal)
res = sfmodel_fit(
    spec=sfspec(
        usedata("examples/data/WangHo2010_data.csv"),
        model=:PFEWH, type=:Prod, dist=Trun(μ=:_cons, σᵤ²=:_cons), σᵥ²=:_cons, hscale=(:z1, :z2),
        ivar=:cnum, depvar=:y, frontiers=(:x1, :x2, :x3)
    ),
    options=sfopt(
        warmstart_maxIT=200,
        warmstart_solver=:NelderMead, 
        main_solver=:Newton
    ), 
);

@test sfmaximizer(res)[1:2] ≈ [0.49727, 0.79545] atol=1e-5
@test sf_inefficiency(res)[1:2] ≈ [3.61393, 4.53487] atol=1e-5
@test sf_efficiency(res)[1:2] ≈ [0.02811, 0.01146] atol=1e-5

marginal, marginal_mean = sfmarginal(res)
@test marginal.marg_zit[1:2] ≈ [1.28355, 1.61064] atol=1e-5

std_ci = sfmarginal(
    res,
    bootstrap=true,
    R=10,
    seed=123,
);
@test bres[1] ≈ 0.03087 atol=1e-5
################################################################################