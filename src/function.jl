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
    likelihood(::Type{Half{T}}, σᵥ², σᵤ², ϵ) where T
    likelihood(::Type{Trun{T, U}}, σᵥ², μ, σᵤ², ϵ) where {T, U}
    likelihood(::Type{Expo{T}}, σᵥ², λ², ϵ) where T

The likelihood function of half normal, truncated normal and exponential distribution

# Arguments
- `σᵥ²::Vector{<:Real}`: variance of the random error
- `μ::Vector{<:Real}`: parameter of the truncated normal distribution
- `σᵤ²::Vector{<:Real}`: parameter of the half or truncated normal distribution
- `λ²::Vector{<:Real}`: parameter of the exponential distribution
- `ϵ::Vector{<:Real}`: conposite error term
"""
function likelihood(::Type{Half{T}}, σᵥ², σᵤ², ϵ) where T
    res = similar(ϵ)
    for i in eachindex(σᵥ², σᵤ², ϵ)
        σ² = σᵤ²[i] + σᵥ²[i]
        σ = sqrt(σ²)
        σₛ² = (σᵥ²[i] * σᵤ²[i]) / σ²
        μₛ = (-σᵤ²[i] * ϵ[i]) / σ²
        res[i] = halfpdf(σ, σ², ϵ[i], μₛ, σₛ²)
    end

    return res
end

function likelihood(::Type{Trun{T, U}}, σᵥ², μ, σᵤ², ϵ) where {T, U}
    res = similar(ϵ)
    for i in eachindex(σᵥ², μ, σᵤ², ϵ)
        σ² = σᵤ²[i] + σᵥ²[i]
        σₛ² = (σᵥ²[i] * σᵤ²[i]) / σ²
        σᵤ = sqrt(σᵤ²[i])
        μₛ = (σᵥ²[i]*μ[i] - σᵤ²[i]*ϵ[i]) / σ²
        res[i] = trunpdf(σ, σ², μ[i], ϵ[i], σₛ², σᵤ, μₛ)
    end
    
    return res
end

function likelihood(::Type{Expo{T}}, σᵥ², λ², ϵ) where T
    res = similar(ϵ)
    for i in eachindex(σᵥ², λ², ϵ)
        λ = sqrt(λ²[i])
        σᵥ = sqrt(σᵥ²[i])
        res[i] = expopdf(λ, ϵ[i], σᵥ, σᵥ²[i], λ²[i])
    end

    return res

end

"""
    loglikelihood(::Type{Half{T}}, σᵥ², σᵤ², ϵ) where T
    loglikelihood(::Type{Trun{T, U}}, σᵥ², μ, σᵤ², ϵ) where {T, U}
    loglikelihood(::Type{Expo{T}}, σᵥ², λ², ϵ) where T

The log likelihood function of half normal, truncated normal and exponential distribution

# Arguments
- `σᵥ²::Vector{<:Real}`: variance of the random error
- `μ::Vector{<:Real}`: parameter of the truncated normal distribution
- `σᵤ²::Vector{<:Real}`: parameter of the half or truncated normal distribution
- `λ²::Vector{<:Real}`: parameter of the exponential distribution
- `ϵ::Vector{<:Real}`: conposite error term
"""
function loglikelihood(::Type{Half{T}}, σᵥ², σᵤ², ϵ) where T
    @floop for i in eachindex(σᵥ², σᵤ², ϵ)
        σ²  = σᵤ²[i] + σᵥ²[i]
        μₛ  = (-σᵤ²[i] * ϵ[i]) / σ²
        σₛ² = (σᵥ²[i] * σᵤ²[i]) / σ²
        llhᵢ = halflogpdf(σ², ϵ[i], μₛ, σₛ²)
        @reduce llh += llhᵢ
    end

    return res
end

function loglikelihood(::Type{<:Trun}, σᵥ², μ, σᵤ², ϵ)
    @floop for i in eachindex(σᵥ², μ, σᵤ², ϵ)
        σ²  = σᵤ²[i] + σᵥ²[i]
        σₛ² = (σᵥ²[i] * σᵤ²[i]) / σ²
        σᵤ = sqrt(σᵤ²[i])
        μₛ  = (σᵥ²[i] * μ[i] - σᵤ²[i] * ϵ[i]) / σ²
        llhᵢ = trunlogpdf(σ², μ[i], ϵ[i], σₛ², σᵤ, μₛ)
        @reduce llh += llhᵢ
    end
    
    return llh
end

function loglikelihood(::Type{Expo{T}}, σᵥ², λ², ϵ) where T
    @floop for i in eachindex(σᵥ², λ², ϵ)
        λ = sqrt(λ²[i])
        σᵥ = sqrt(σᵥ²[i])
        llhᵢ = expologpdf(λ, ϵ[i], σᵥ, σᵥ²[i], λ²[i])
        @reduce llh += llhᵢ
    end

    return llh
end


"""
    _jlmsbc(::Type{Half{T}}, σᵥ², σᵤ², ϵ) where T
    _jlmsbc(::Type{Trun{T, U}}, σᵥ², μ, σᵤ², ϵ) where {T, U}
    _jlmsbc(::Type{Trun{T, U}}, σᵥ², μ, σᵤ², ϵ) where {T, U}

The close form of the jlms, bc index and the distribution assumption includes 
half normal, normal, exponential

# Arguments
- `σᵥ²::Vector{<:Real}`: variance of the random error
- `μ::Vector{<:Real}`: parameter of the truncated normal distribution
- `σᵤ²::Vector{<:Real}`: parameter of the half or truncated normal distribution
- `λ²::Vector{<:Real}`: parameter of the exponential distribution
- `ϵ::Vector{<:Real}`: conposite error term
"""
function _jlmsbc(::Type{Half{T}}, σᵥ², σᵤ², ϵ) where T
    jlms = similar(ϵ)
    bc = similar(ϵ)
    @floop for i in eachindex(σᵥ², σᵤ², ϵ)
        σ²  = σᵤ²[i] + σᵥ²[i] 
        μₛ  = (-σᵤ²[i] * ϵ[i]) / σ²
        σₛ  = sqrt((σᵥ²[i]*σᵤ²[i]) / σ²)
        jlms[i] = (σₛ * normpdf(μₛ/σₛ)) / normcdf(μₛ/σₛ) + μₛ
        bc[i] = exp(-μₛ+0.5*σₛ^2) * (normcdf((μₛ/σₛ)-σₛ) / normcdf(μₛ/σₛ))
    end

    return jlms, bc
end

function _jlmsbc(::Type{Trun{T, U}}, σᵥ², μ, σᵤ², ϵ) where {T, U}
    jlms = similar(ϵ)
    bc = similar(ϵ)
    @floop for i in eachindex(σᵥ², μ, σᵤ², ϵ)
        σ²  = σᵤ²[i] + σᵥ²[i] 
        μₛ  = (σᵥ²[i]*μ[i] - σᵤ²[i]*ϵ[i]) / σ²
        σₛ  = sqrt((σᵥ²[i]*σᵤ²[i]) / σ²)
        jlms[i] = (σₛ*normpdf(μₛ/σₛ)) / normcdf(μₛ/σₛ) + μₛ
        bc[i] = exp(-μₛ+0.5*σₛ^2) * (normcdf((μₛ/σₛ)-σₛ) / normcdf(μₛ/σₛ))
    end
    
    return jlms, bc
end

function _jlmsbc(::Type{Expo{T}}, σᵥ², λ, ϵ) where T
    jlms = similar(ϵ)
    bc = similar(ϵ)
    @floop for i in eachindex(σᵥ², λ, ϵ)
        σᵥ = sqrt(σᵥ²[i])
        μₛ  = (-ϵ[i]) - (σᵥ²[i]/λ[i])
        jlms[i] = (σᵥ*normpdf(μₛ/σᵥ)) / normcdf(μₛ/σᵥ) + μₛ
        bc[i] = exp(-μₛ+0.5*σᵥ²[i]) .* (normcdf((μₛ/σᵥ)-σᵥ) / normcdf(μₛ/σᵥ))
    end
    
    return jlms, bc
end


"""
    uncondU(::Type{Half{T}}, σᵤ², dist_coeff) where T
    uncondU(::Type{Trun{T, U}}, μ, σᵤ², dist_coeff) where{T, U}
    uncondU(::Type{Expo{T}}, λ, dist_coeff) where T

Calculate the unconditional mean of the composite error term
Notice that if it's calculated to generate the marginal effect return Real 
required by the forwardDiff

# Arguments
- `μ::Vector{<:Real}`: parameter of the truncated normal distribution
- `σᵤ²::AbstractMatrix{<:Real}`
- `λ²::Vector{<:Real}`: parameter of the exponential distribution
- `dist_coeff::Vector{<:Real}`
"""
function uncondU(::Type{Half{T}}, σᵤ², coeff) where T
    res = broadcast(sqrt, broadcast(*, (2/π), broadcast(exp, (σᵤ² * coeff))))
    length(res) != 1 ? (return res) : (return res[1])
end

function uncondU(::Type{Trun{T, U}}, μ, σᵤ², coeff) where{T, U}
    n = numberofvar(μ)
    Wμ, Wᵤ = coeff[begin:n], coeff[n+1:end] 
    μ = μ * Wμ
    σᵤ = broadcast(exp, broadcast(*, 0.5, (σᵤ² * Wᵤ)))

    Λ = broadcast(/, μ, σᵤ)
    res = broadcast(*, σᵤ, (Λ + broadcast(/, broadcast(normpdf, Λ), broadcast(normcdf, Λ))))
    length(res) != 1 ? (return res) : (return res[1])
end

function uncondU(::Type{Expo{T}}, λ, dist_coeff) where T
    res = broadcast(exp, λ * coeff)
    length(res) != 1 ? (return res) : (return res[1])
end


"""
    clean_marginaleffect(m::Matrix{<:Any}, labels::Vector{Symbol})

To drop the constant and duplicated marginal effect
"""
function clean_marginaleffect(m, labels)
    unique_label = unique(labels)
    pos = Dict([(i, findall(x->x==i, labels)) for i in unique_label])
    id = Dict([(i, pos[i][1]) for i in unique_label])
    count = Dict([(i, length(pos[i])) for i in unique_label])
    drop = []
    for (i, label) in enumerate(labels)
        # task1: drop the constant columns
        if length(unique(m[:, i])) == 1
            append!(drop, i)
            count[label] -= 1
            if i == id[label] && count[label] != 0
                id[label] = pos[label][1+(length(pos[label])-count[label])]
            end
            continue
        end
        # task2: drop the columns with duplicated column names
        if i != id[label]
            tar = id[label]
            m[:, tar] = m[:, tar] + m[:, i]
            append!(drop, i)
            count[label] -= 1
        end
    end
    length(labels) == length(drop) && error("there is no marginal effect")

    return m[:, Not(drop)], unique_label
end


"""
    _marginaleffect(ξ::Vector{T}, struc::SFmodel, data::AbstractData, bootstrap::Bool) where T

*Warning: this method only create the base template, further completion should
be done.*
"""
function _marginaleffect(ξ, model, data, bootstrap)
    # prepare the distribution data
    dist_data = unpack(distof(model))
    var_nums = [numberofvar(i) for i in dist_data]
    var_num = sum(var_nums)

    
    dist_coef = slice(ξ, model.ψ, mle=true)[2]
    # the purpose of `let` is to avoid boxed variables required by the `@floop`
    dist_data = hcat(dist_data...)
    dist_type = typeofdist(model)
    mm = Matrix{Float64}(undef, numberofobs(dist_data), var_num)
    for i = axes(mm, 1)
        @inbounds mm[i, :] = gradient(
            marg -> uncondU(
                dist_type,
                [reshape(j, 1, length(j)) for j in slice(marg, var_nums)]...,
                dist_coef
            ),
            dist_data[i, :]
        )
    end

    beg_label = numberofvar(data.frontiers) + 1
    en_label = beg_label + sum(var_nums) - 1
    label = model.paramnames[beg_label:en_label, 2]  # use the varmat to get the column name of datafrae
    mm, label = clean_marginaleffect(mm, label)  # drop the duplicated and constant columns
    mean_marginal = mean(mm, dims=1)
    if bootstrap
        return mm, mean_marginal
    else
        label = [Symbol(:marg_, i) for i in label]
        return DataFrame(mm, label), NamedTuple{Tuple(label)}(mean_marginal)
    end
end