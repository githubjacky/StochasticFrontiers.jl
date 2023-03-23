function composite_error(coeff, model::PFEWH, data)
    Ỹ, X̃, hscale =  unpack(
        get_modeldata(model), (:Ỹ, :X̃, :hscale)
    )
    econtype, fitted_dist, σᵥ² = unpack(
        data, (:econtype, :fitted_dist, :σᵥ²)
    )

    σᵥ² = broadcast(exp, σᵥ²*coeff[3])
    dist_param = fitted_dist(coeff[2])

    h = broadcast(exp, hscale*coeff[4])
    h̃ = sf_demean(h)

    ϵ̃ = Panel(
        (econtype * (Ỹ - X̃*coeff[1]))[:, 1],
        get_rowidx(data)
    )

    return ϵ̃, σᵥ², dist_param, h̃
end


function LLT(ξ, model::PFEWH, data::Data)
    ϵ̃, σᵥ², _dist_param, h̃ = composite_error(
        slice(ξ, get_paramlength(model), mle=true), 
        model, 
        data
    )
    dist_type = typeofdist(model)

    return -loglikelihood(typeofdist(data), σᵥ², dist_param..., ϵ)
end