function halfpdf(σ, σ², ϵ, μₛ, σₛ²)
    res = (2 / σ) *
           normpdf(ϵ / sqrt(σ²)) *
           normcdf(μₛ / sqrt(σₛ²))

    return res
end

function trunpdf(σ, σ², μ, ϵ, σₛ², σᵤ, μₛ)
    res = (1/σ) *
           normpdf((μ+ϵ) / sqrt(σ²)) *
           normcdf(μₛ / sqrt(σₛ²)) / 
           normcdf(μ / σᵤ)

    return res
end

function expopdf(λ, ϵ, σᵥ, σᵥ², λ²)
    res = (1/λ) *
           normcdf(-(ϵ/σᵥ) - (σᵥ/λ)) *
           (ϵ/λ) *
           (σᵥ² / (2*λ²))

    return res
end

function halflogpdf(σ², ϵ, μₛ, σₛ²) 
    res = (-0.5 * log(σ²)) + 
           normlogpdf(ϵ / sqrt(σ²)) + 
           normlogcdf(μₛ / sqrt(σₛ²)) -
           normlogcdf(0)

    return res
end

function trunlogpdf(σ², μ, ϵ, σₛ², σᵤ, μₛ)
    res = (-0.5 * log(σ²)) +
           normlogpdf((μ+ϵ) / sqrt(σ²)) +
           normlogcdf(μₛ / sqrt(σₛ²)) - 
           normlogcdf(μ / σᵤ)

    return res
end

function expologpdf(λ, ϵ, σᵥ, σᵥ², λ²)
    res = -log(λ) +
           normlogcdf(-(ϵ/σᵥ) - (σᵥ/λ)) +
           (ϵ/λ) +
           (σᵥ² / (2*λ²))

    return res
end

"""
    _likelihood(::Type{<:Half}, σᵥ², σᵤ², ϵ)
    _likelihood(::Type{<:Trun}, σᵥ², μ, σᵤ², ϵ)
    _likelihood(::Type{<:Expo}, σᵥ², λ², ϵ)

The likelihood function of half normal, truncated normal and exponential distribution

# Arguments
- `σᵥ²::Vector{<:Real}`: variance of the random error
- `μ::Vector{<:Real}`  : parameter of the truncated normal distribution
- `σᵤ²::Vector{<:Real}`: parameter of the half or truncated normal distribution
- `λ²::Vector{<:Real}` : parameter of the exponential distribution
- `ϵ::Vector{<:Real}`  : conposite error term

"""
function _likelihood(::Type{<:Half}, σᵥ², σᵤ², ϵ) 
    res = similar(ϵ)
    @inbounds @views for i in eachindex(σᵥ², σᵤ², ϵ)
        σ² = σᵤ²[i] + σᵥ²[i]
        σ = sqrt(σ²)
        σₛ² = (σᵥ²[i] * σᵤ²[i]) / σ²
        μₛ = (-σᵤ²[i] * ϵ[i]) / σ²
        res[i] = halfpdf(σ, σ², ϵ[i], μₛ, σₛ²)
    end

    return Base.prod(res)
end

function _likelihood(::Type{<:Trun}, σᵥ², μ, σᵤ², ϵ) 
    res = similar(ϵ)
    @inbounds @views for i in eachindex(σᵥ², μ, σᵤ², ϵ)
        σ² = σᵤ²[i] + σᵥ²[i]
        σₛ² = (σᵥ²[i] * σᵤ²[i]) / σ²
        σᵤ = sqrt(σᵤ²[i])
        μₛ = (σᵥ²[i]*μ[i] - σᵤ²[i]*ϵ[i]) / σ²
        res[i] = trunpdf(sqrt(σ²), σ², μ[i], ϵ[i], σₛ², σᵤ, μₛ)
    end
    
    return Base.prod(res)
end

function _likelihood(::Type{<:Expo}, σᵥ², λ², ϵ) 
    res = similar(ϵ)
    @inbounds @views for i in eachindex(σᵥ², λ², ϵ)
        λ = sqrt(λ²[i])
        σᵥ = sqrt(σᵥ²[i])
        res[i] = expopdf(λ, ϵ[i], σᵥ, σᵥ²[i], λ²[i])
    end

    return Base.prod(res)

end

"""
    loglikelihood(::Type{<:Half}, σᵥ², σᵤ², ϵ)
    loglikelihood(::Type{<:Trun}, σᵥ², μ, σᵤ², ϵ)
    loglikelihood(::Type{<:Expo}, σᵥ², λ², ϵ)

The log likelihood function of half normal, truncated normal and exponential distribution

# Arguments
- `σᵥ²::Vector{<:Real}`: variance of the random error
- `μ::Vector{<:Real}`  : parameter of the truncated normal distribution
- `σᵤ²::Vector{<:Real}`: parameter of the half or truncated normal distribution
- `λ²::Vector{<:Real}` : parameter of the exponential distribution
- `ϵ::Vector{<:Real}` : conposite error term

"""
function _loglikelihood(::Type{<:Half}, σᵥ², σᵤ², ϵ) 
    @inbounds @views @floop for i in eachindex(σᵥ², σᵤ², ϵ)
        σ²  = σᵤ²[i] + σᵥ²[i]
        μₛ  = (-σᵤ²[i] * ϵ[i]) / σ²
        σₛ² = (σᵥ²[i] * σᵤ²[i]) / σ²
        llhᵢ = halflogpdf(σ², ϵ[i], μₛ, σₛ²)
        @reduce llh = 0 + llhᵢ
    end

    return llh
end

function _loglikelihood(::Type{<:Trun}, σᵥ², μ, σᵤ², ϵ) 
    @inbounds @views @floop for i in eachindex(σᵥ², μ, σᵤ², ϵ)
        σ²  = σᵤ²[i] + σᵥ²[i]
        σₛ² = (σᵥ²[i] * σᵤ²[i]) / σ²
        σᵤ = sqrt(σᵤ²[i])
        μₛ  = (σᵥ²[i] * μ[i] - σᵤ²[i] * ϵ[i]) / σ²
        llhᵢ = trunlogpdf(σ², μ[i], ϵ[i], σₛ², σᵤ, μₛ)
        @reduce llh = 0 + llhᵢ
    end
    
    return llh
end

function _loglikelihood(::Type{<:Expo}, σᵥ², λ², ϵ) 
     @inbounds @views @floop for i in eachindex(σᵥ², λ², ϵ)
        λ = sqrt(λ²[i])
        σᵥ = sqrt(σᵥ²[i])
        llhᵢ = expologpdf(λ, ϵ[i], σᵥ, σᵥ²[i], λ²[i])
        @reduce llh = 0 + llhᵢ
    end

    return llh
end


"""
    _jlmsbc(::Type{<:Half}, σᵥ², σᵤ², ϵ)
    _jlmsbc(::Type{<:Trun}, σᵥ², μ, σᵤ², ϵ)
    _jlmsbc(::Type{<:Expo}, σᵥ², λ², ϵ)

The close form of the jlms, bc index and the distribution assumption includes 
half normal, normal, exponential

# Arguments
- `σᵥ²::Vector{<:Real}`: variance of the random error
- `μ::Vector{<:Real}`  : parameter of the truncated normal distribution
- `σᵤ²::Vector{<:Real}`: parameter of the half or truncated normal distribution
- `λ²::Vector{<:Real}` : parameter of the exponential distribution
- `ϵ::Vector{<:Real}`  : conposite error term

"""
function _jlmsbc(::Type{<:Half}, σᵥ², σᵤ², ϵ) 
    jlms = similar(ϵ)
    bc = similar(ϵ)
    @inbounds @views @floop for i in eachindex(σᵥ², σᵤ², ϵ)
        σ²  = σᵤ²[i] + σᵥ²[i] 
        μₛ  = (-σᵤ²[i] * ϵ[i]) / σ²
        σₛ  = sqrt((σᵥ²[i]*σᵤ²[i]) / σ²)
        jlms[i] = (σₛ * normpdf(μₛ/σₛ)) / normcdf(μₛ/σₛ) + μₛ
        bc[i] = exp(-μₛ+0.5*σₛ^2) * (normcdf((μₛ/σₛ)-σₛ) / normcdf(μₛ/σₛ))
    end

    return jlms, bc
end

function _jlmsbc(::Type{<:Trun}, σᵥ², μ, σᵤ², ϵ) 
    jlms = similar(ϵ)
    bc = similar(ϵ)
    @inbounds @views @floop for i in eachindex(σᵥ², μ, σᵤ², ϵ)
        σ²  = σᵤ²[i] + σᵥ²[i] 
        μₛ  = (σᵥ²[i]*μ[i] - σᵤ²[i]*ϵ[i]) / σ²
        σₛ  = sqrt((σᵥ²[i]*σᵤ²[i]) / σ²)
        jlms[i] = (σₛ*normpdf(μₛ/σₛ)) / normcdf(μₛ/σₛ) + μₛ
        bc[i] = exp(-μₛ+0.5*σₛ^2) * (normcdf((μₛ/σₛ)-σₛ) / normcdf(μₛ/σₛ))
    end
    
    return jlms, bc
end

function _jlmsbc(::Type{<:Expo}, σᵥ², λ², ϵ)
    jlms = similar(ϵ)
    bc = similar(ϵ)
    @inbounds @views @floop for i in eachindex(σᵥ², λ², ϵ)
        σᵥ = sqrt(σᵥ²[i])
        λ = sqrt(λ²[i])
        μₛ  = (-ϵ[i]) - (σᵥ²[i]/λ)
        jlms[i] = (σᵥ*normpdf(μₛ/σᵥ)) / normcdf(μₛ/σᵥ) + μₛ
        bc[i] = exp(-μₛ+0.5*σᵥ²[i]) * (normcdf((μₛ/σᵥ)-σᵥ) / normcdf(μₛ/σᵥ))
    end
    
    return jlms, bc
end


"""
    _unconditional_mean(::Type{<:Half}, coeff, log_σᵤ²)
    _unconditional_mean(::Type{<:Trun}, coeff, μ, log_σᵤ²)
    _unconditional_mean(::Type{<:Expo}, coeff, log_λ²)

Calculate the unconditional mean of the composite error term
Notice that if it's calculated to generate the marginal effect return Real 
required by the forwardDiff

# Arguments
- `μ::Vector{<:Real}`: parameter of the truncated normal distribution

- `log_σᵤ²::Vector{<:Real}`

- `log_λ²::Vector{<:Real}`: parameter of the exponential distribution

- `dist_coeff::Vector{<:Real}`
"""
function _unconditional_mean(::Type{<:Half}, coeff, log_σᵤ²) 
    uncondU = sqrt((2/π) * exp(log_σᵤ²'*coeff))
    
    return uncondU
end

function _unconditional_mean(::Type{<:Trun}, coeff, _μ, log_σᵤ²) 
    Wμ, Wᵤ = slice(coeff, [length(_μ), length(log_σᵤ²)])
    μ = _μ' * Wμ
    σᵤ = exp(0.5*log_σᵤ²'*Wᵤ)
    Λ = μ / σᵤ
    uncondU = σᵤ * (Λ + normpdf(Λ)/normcdf(Λ))

    return uncondU
end

function _unconditional_mean(::Type{<:Expo}, coeff, log_λ²) 
    uncondU = sqrt(exp(log_λ²'*coeff))
    
    return uncondU
end
