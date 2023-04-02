function _jlmsbc(::Type{PFEWH{Trun{S, U}, V, W, X}}, σᵥ², μ, σᵤ², ϵ̃, h) where{S, U, V, W, X}
    N = sum(numberoft(ϵ̃))
    jlms, bc = zeros(N), zeros(N)
    h̃ = sf_demean(h)
    @floop for i in eachindex(ϵ̃[:, 1], h, h̃)
        σₛₛ² = 1.0 / (h̃[i]^2*(1/σᵥ²) + 1/σᵤ²)  # h̃ is already de-meaned, so skip Π⁻ but mind 1/σᵥ² # 1.0/(h̃'*Π⁻*h̃ + 1/σᵤ²)  
        σₛₛ  = sqrt(σₛₛ²)
        μₛₛ = (μ/σᵤ² - ϵ̃[i]*h̃[i]*(1/σᵥ²)) * σₛₛ² # (μ/σᵤ² - ϵ̃[ind]'*Π⁻*h̃) * σₛₛ²

        jlms[i] = h[i] * (μₛₛ + normpdf(μₛₛ/σₛₛ)*σₛₛ/normcdf(μₛₛ/σₛₛ))
        bc[i] = ((normcdf(μₛₛ/σₛₛ-h[i]*σₛₛ)) / normcdf(μₛₛ/σₛₛ)) * exp(-h[i]*μₛₛ + 0.5*(h[i]^2)*σₛₛ²)
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

marginal_data(model::PFEWH) = _marg_data(model, :hscale)
marginal_coeff(::Type{PFEWH{T, S, U, V}}, ξ, ψ) where{T, S, U, V} = slice(ξ, ψ, mle=true)[[2, 4]]

function unconditional_mean(::Type{PFEWH{Trun{S, U}, V, W, X}}, coeff, μ, log_σᵤ², log_h) where {S, U, V, W, X}
    Wμ = μ == zeros(1) ? zeros(1) : coeff[1][begin:begin+length(μ)-1]
    Wᵤ, Wₕ = coeff[1][end-length(log_σᵤ²)+1:end], coeff[2]
    μ = μ' * Wμ

    log_h = log_h' * Wₕ
    σᵤ = exp(log_h + 0.5 * log_σᵤ²' * Wᵤ)
    μ = exp(log_h) * μ
    Λ = μ / σᵤ
    res = σᵤ * (Λ + normpdf(Λ) / normcdf(Λ))

    return res
end

function unconditional_mean(::Type{PFEWH{Half{S}, U, V, W}}, coeff, log_σᵤ², log_h) where {S, U, V, W}
    return unconditional_mean(PFEWH{Trun{S, S}, U, V, W}, coeff, zeros(1), log_σᵤ², log_h)
end