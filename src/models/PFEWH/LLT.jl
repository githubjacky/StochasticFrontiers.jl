function composite_error(coeff, model::PFEWH, data)
    fitted_dist, Ỹ, X̃, hscale =  unpack(
        model, (:fitted_dist, :Ỹ, :X̃, :hscale)
    )
    econtype, σᵥ² = unpack(data, (:econtype, :σᵥ²))

    σᵥ² = exp((σᵥ²*coeff[3])[1])  # σᵥ² is a scalar
    dist_param = [i[1] for i in fitted_dist(coeff[2])]  # all dist_param are scalars
    h = broadcast(exp, hscale*coeff[4])
    _ϵ̃ = sf_demean(econtype * (Ỹ - X̃*coeff[1]))
    ϵ̃ = Panel(_ϵ̃.data[:, 1], _ϵ̃.rowidx)

    return ϵ̃, σᵥ², dist_param, h
end


function loglikelihood(::Type{PFEWH{Trun{S, U}, V, W, X}}, 
                       σᵥ², μ, σᵤ², _ϵ̃, _h̃, γ, δ2) where{S, U, V, W, X}
    ϵ̃  = Panelized(_ϵ̃)
    h̃  = Panelized(_h̃) 
    T  = numberoft(ϵ̃)

    @floop for (ϵ̃ᵢ, h̃ᵢ, Tᵢ) in zip(ϵ̃, h̃, T)
        σₛₛ² = 1 / ( h̃ᵢ' * h̃ᵢ * (1/σᵥ²) + (1/σᵤ²) )
        μₛₛ  = ( μ/σᵤ² - (ϵ̃ᵢ' * h̃ᵢ * (1/σᵥ²))[1] ) * σₛₛ²
        
        es2  = -0.5 * sum(ϵ̃ᵢ.^2) * (1/σᵥ²)
        KK   = -0.5 * (Tᵢ-1) * log2π - 0.5 * (Tᵢ-1) * γ
        llhᵢ = KK + es2 +
               0.5 * ( μₛₛ^2/σₛₛ² - μ^2/σᵤ² ) +
               0.5 * log(σₛₛ²) + 
               normlogcdf( μₛₛ/sqrt(σₛₛ²) ) -
               0.5 * δ2 - 
               normlogcdf( μ/sqrt(σᵤ²) )

        @reduce llh = 0. + llhᵢ
    end

    return llh
end


function LLT(ξ, model::PFEWH{Trun{T, S}, U, V, W}, data::PanelData) where{T, S, U, V, W}
    coeff = slice(ξ, get_paramlength(model), mle=true)
    ϵ̃, σᵥ², dist_param, h = composite_error(
        coeff, 
        model, 
        data
    )
    h̃ = sf_demean(h)
    llh = loglikelihood(
        typeof(model), σᵥ², dist_param..., ϵ̃, h̃, coeff[3][1], coeff[2][2]
    )

    return -llh
end

function LLT(ξ, model::PFEWH{Half{T}, S, U, V}, data::PanelData) where{T, S, U, V}
    coeff = slice(ξ, get_paramlength(model), mle=true)
    ϵ̃, σᵥ², dist_param, h = composite_error(
        coeff, 
        model, 
        data
    )
    h̃ = sf_demean(h)

    llh = loglikelihood(
        PFEWH{Trun{T, T}, S, U, V}, σᵥ², 0., dist_param..., ϵ̃, h̃, coeff[3][1], coeff[2][1]
    )

    return -llh
end
