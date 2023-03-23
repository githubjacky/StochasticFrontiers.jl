function composite_error(coeff, model::PFEWH, data)
    econtype, fitted_dist, _σᵥ², depvar, frontiers = unpack(
        data, (:econtype, :fitted_dist, :σᵥ², :depvar, :frontiers)
    )
    ϵ = (econtype * (depvar- frontiers*coeff[1]))[:, 1]
    σᵥ² = exp.(_σᵥ² * coeff[3])
    dist_param = fitted_dist(coeff[2])

    return ϵ, σᵥ², dist_param
end


function LLT(ξ, model::PFEWH, data::Data)
    ϵ, σᵥ², dist_param = composite_error(
        slice(ξ, get_paramlength(model), mle=true), 
        model, 
        data
    )

    return -sum(loglikelihood(typeofdist(data), σᵥ², dist_param..., ϵ))
end