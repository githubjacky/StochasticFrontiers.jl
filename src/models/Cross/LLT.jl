function composite_error(ξ, struc::Cross, data::Data)
    β, dist_coeff, Wᵥ = slice(ξ, struc.ψ, mle=true)
    ϵ = (data.type() * (data.depvar - data.frontiers*β))[:, 1]
    σᵥ² = exp.(data.σᵥ² * Wᵥ)
    dist_param = data.dist(dist_coeff)

    return ϵ, σᵥ², dist_param
end


function LLT(ξ, struc::Cross, data::Data)
    ϵ, σᵥ², dist_param = composite_error(ξ, struc, data)

    return -sum(loglikelihood(typeof(data.dist), σᵥ², dist_param..., ϵ))
end