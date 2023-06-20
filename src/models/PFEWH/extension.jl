function _jlmsbc(::Type{PFEWH{Trun{S, U}, V, W, X}}, σᵥ², μ, σᵤ², ϵ̃, h) where{S, U, V, W, X}
    N    = sum(numberoft(ϵ̃))
    jlms = zeros(N)
    bc   = zeros(N)
    h̃    = sf_demean(h)

    @floop for (ϵ̃ᵢ, h̃ᵢ, hᵢ, idx) in zip(Panelized(ϵ̃), Panelized(h̃), Panelized(h), h.rowidx)
        σₛₛ² = 1.0 / ( h̃ᵢ' * h̃ᵢ * (1/σᵥ²) + 1/σᵤ² )
        σₛₛ  = sqrt(σₛₛ²)
        μₛₛ  = ( μ/σᵤ² - ϵ̃ᵢ' * h̃ᵢ * (1/σᵥ²) ) * σₛₛ²

        jlms[idx] = @. hᵢ * ( μₛₛ + normpdf(μₛₛ/σₛₛ) * σₛₛ / normcdf(μₛₛ/σₛₛ) )
        bc[idx]   = @. ( (normcdf(μₛₛ/σₛₛ-hᵢ*σₛₛ)) / normcdf(μₛₛ/σₛₛ) ) * 
                       exp( -hᵢ * μₛₛ + 0.5 * (hᵢ^2) * σₛₛ² )
    end

    return jlms, bc
end

function _jlmsbc(::Type{PFEWH{Half{S}, U, V, W}}, σᵥ², σᵤ², ϵ̃, h) where{S, U, V, W}
    return _jlmsbc(PFEWH{Trun{S, S}, U, V, W}, σᵥ², 0., σᵤ², ϵ̃, h)
end


function jlmsbc(ξ, model::PFEWH, data::PanelData)
    ϵ̃, σᵥ², dist_param, h = composite_error(
        slice(ξ, get_paramlength(model), mle=true),
        model, 
        data
    )
    jlms, bc = _jlmsbc(typeof(model), σᵥ², dist_param..., ϵ̃, h)

   return jlms, bc
end


marginal_data(model::PFEWH)                                       = _marg_data(model, :hscale)
marginal_coeff(::Type{PFEWH{T, S, U, V}}, ξ, ψ) where{T, S, U, V} = slice(ξ, ψ, mle=true)[[2, 4]]
function marginal_label(model::PFEWH, k)
    dist_var_num = sum(numberofvar.(unpack(distof(model))))
    beg_dist     = k + 1
    en_dist      = beg_dist + dist_var_num - 1

    _label = get_paramname(model)[:, 2]
    label  = _label[union( 
        beg_dist:en_dist, 
        (length(_label) - numberofvar(model.hscale) + 1):length(_label) 
    )]  

    return label
end

function unconditional_mean(::Type{PFEWH{Trun{S, U}, V, W, X}}, 
                            coeff, 
                            μ, 
                            log_σᵤ², 
                            log_h
                            ) where {S, U, V, W, X}

    Wμ = μ == zeros(1) ? zeros(1) : coeff[1][begin:begin+length(μ)-1]
    Wᵤ = coeff[1][end-length(log_σᵤ²)+1:end]

    log_h   = log_h' * coeff[2]

    μ  = exp(log_h) * (μ' * Wμ)
    σᵤ = exp(log_h + 0.5 * log_σᵤ²' * Wᵤ)
    Λ = μ / σᵤ
    uncondU = σᵤ * ( Λ + normpdf(Λ) / normcdf(Λ) )

    return uncondU
end

function unconditional_mean(::Type{PFEWH{Half{S}, U, V, W}}, coeff, log_σᵤ², log_h) where {S, U, V, W}
    return unconditional_mean(PFEWH{Trun{S, S}, U, V, W}, coeff, zeros(1), log_σᵤ², log_h)
end
