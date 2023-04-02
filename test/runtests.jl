using Test, Distributions, StochasticFrontiers


@testset verbose=true "Part 1 (cross-section) of log-likelihood functions, marginal effect, jlms & bc index" begin
    data= [-1.06966    1.53076     0.346522   1;
		   -1.55598   -1.54822     0.0296991  1;
		   -1.06309    0.0860199  -0.546911   1;
			0.396344  -1.59922    -1.62234    1;
		   -0.367106  -1.31003     1.67005    1]
    
    y = data[:, 1]
    X = μ = σᵤ² = data[:, 2:3]
    σᵥ² = data[:, 1]

    # Cross, half normal
    res = sfmodel_fit(
        spec=sfspec(
            Cross, type=Prod, 
            dist=Half(σᵤ²=σᵤ²), σᵥ²=σᵥ², 
            depvar=y, frontiers=X
        ),
        options=sfopt(verbose=false)
    )

    sfmaximum(res)
    sf_inefficiency(res)
    marginal, marginal_mean = sfmarginal(res)
    marginal[1, 1]
    marginal_mean[1]
    sfmaximizer(res)
end

