function composite_error(coeff, model::Cross, data)
    fitted_dist = distof(model)
    econtype, σᵥ², depvar, frontiers = unpack(
        data, (:econtype, :σᵥ², :depvar, :frontiers)
    )

    σᵥ² = exp.(σᵥ² * coeff[3])
    dist_param = fitted_dist(coeff[2])
    ϵ = (econtype * (depvar- frontiers*coeff[1]))[:, 1]
    
    return ϵ, σᵥ², dist_param
end


function LLT(ξ, model::Cross, data::Data)
    ϵ, σᵥ², dist_param = composite_error(
        slice(ξ, get_paramlength(model), mle=true), 
        model, 
        data
    )

    return -_loglikelihood(typeofdist(model), σᵥ², dist_param..., ϵ)
end