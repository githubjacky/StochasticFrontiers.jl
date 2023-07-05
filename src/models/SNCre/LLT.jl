#########################################################################################
# TODO: composite error term
# composite_error(coeff::Vector{Vector{T}}, model::AbstractModel, data) where T
#########################################################################################

"""
    dear(simulate_ϵ::T, rowidx::Vector{UnitRange{Int}}, p::Int, ϕ::Vector{<:Real}) where T

extract the pure composite error term from the all AR terms

"""
function dear(simulate_ϵ::T, rowidx, p, ϕ) where T
    base = lagdrop(simulate_ϵ, rowidx, p)

    ar_terms = T[ 
        lagshift(simulate_ϵ, rowidx, p; shift = lag)
        for lag in 1:p
    ]

    ar_factor = sum(ar_terms .* ϕ)

    return base - ar_factor
end


"""
    dema(rowidx, econtype, simulate_ϵ, Eη, q, θ)

Subtract the `simulate_ϵ` from the moving average terms

# Arguments
- `rowidx::Vector{UnitRange{Int}}`
- `econtype::AbstractEconType`
- `simulate_ϵ::Matrix{<:Real}`
- `Eη::Vector{<:Real}`        : unconditional mean of  `η` for each individual
- `q::Int`
- `θ::Vector{<:Real}`

"""
function dema(rowidx, simulate_ϵ::Matrix{T}, Eη, q, θ) where T
    G, R        = size(simulate_ϵ)
    simulate_ϵ_ = static_panelize(simulate_ϵ, rowidx)
    simulate_η  = Matrix{T}(undef, G-(length(rowidx)*q), R)  # some observations will be dropped
    rowidx_     = newrange(rowidx, q)

    @inbounds for i = eachindex(simulate_ϵ_)
        instance        = simulate_ϵ_[i]
        t               = numberofobs(instance)
        ηᵢ              = Matrix{T}(undef, t+q, R)
        ηᵢ[1:q, :]     .= view(Eη, i)

        for j = 1+q : t+q
            ηᵢ[j, :] = begin
                view(instance, j-q, :) - 
                sum( Vector{T}[view(ηᵢ, j-k, :) for k = eachindex(θ)] .* θ )
            end
        end

        simulate_η[rowidx_[i], :] = ηᵢ[(begin+2*q):end, :]
    end

    return simulate_η
end


"""
    eta(rowidx, econtype, serialcorr::AR, simulate_ϵ, ϕ, args...)
    eta(rowidx, econtype, serialcorr::MA, simulate_ϵ, θ, model_type, dist_coeff, dist_paaram)
    eta(rowidx, econtype, serialcorr::ARMA, simulate_ϵ, lagcoeff, model_type, dist_coeff, dist_param)

Purely calculate the total composite error term

# Arguments
- `econtype::AbstractEconType`    : economic interpretation, either Prodution or Cost.
- `serialcorr::AbstractSerialCorr`: serial correlation assumption, AR, MA or ARMA.
- `simulate_ϵ::Matrix{T} where T` : simulated total composite_error
- `ϕ::Vector{Int}`                :
    - `ϕ`
    - `lagcoeff`

- `model_type::Type{<:AbstractSFmodel}`
- `dist_coeff`::Vector{Int}
- `dist_param`::NTuple{N, Matrix{T}} where {N, T}`

"""
function eta(rowidx, econtype, serialcorr::AR, simulate_ϵ, ϕ, args...)

    # if lag period > 1, let all the element of ρ <  1 to ensure stationality
    # notice that the last coefficient is constant which will be decluded
    ϕ = begin
        serialcorr.p == 1 ? 
            ϕ : 
            coeffs(fromroots( ϕ ./ (abs.(ϕ).+1) ))[begin:end-1]
    end

    simulate_η = econtype * dear(simulate_ϵ, rowidx, serialcorr.p, ϕ)
    
    return simulate_η
end


function Eη_by_individual(rowidx, dist_param::NTuple{N, Vector{T}}, model_type, dist_coeff) where{N, T}
    dist_param_ = map(
        x -> static_panelize(x, rowidx),
        dist_param
    )

    Eη = Vector{T}(undef, length(rowidx))

    # mean of unconditional mean for individual i across different period t
    for (i, val) = enumerate(zip(dist_param_...))
        _Eη = map(
            x -> unconditional_mean(model_type, dist_coeff, x...),
            zip(eachrow.(val)...)
        )
        Eη[i] = mean(_Eη)
    end

    return Eη
end


function eta(rowidx, econtype, serialcorr::MA, simulate_ϵ, θ, model_type, dist_coeff, dist_param)

    Eη         = Eη_by_individual(rowidx, dist_param, model_type, dist_coeff)
    simulate_η = econtype * dema(rowidx, simulate_ϵ, Eη, lagparam(serialcorr) , θ)

    return simulate_η
end


function eta(rowidx, econtype, serialcorr::ARMA, simulate_ϵ, lag_coeff, model_type, dist_coeff, dist_param)
    p, q = unpack(serialcorr)
    ϕ, θ = lag_coeff[begin:p], lag_coeff[p+1:end]

    ϕ = begin
        serialcorr.p == 1 ? 
            ϕ : 
            coeffs(fromroots( ϕ ./ (abs.(ϕ).+1) ))[begin:end-1]
    end

    de_ar      = dear(simulate_ϵ, rowidx, p, ϕ)
    Eη         = Eη_by_individual(valid_range(rowidx, p), dist_param, model_type, dist_coeff)
    de_ma      = dema(newrange(rowidx, p), de_ar, Eη, q, θ)
    simulate_η = econtype * de_ma

    return simulate_η
end


function composite_error(coeff::Vector{Vector{T}}, model::SNCre, data::PanelData) where T
    σᵥ² = exp.(data.σᵥ² * coeff[3])

    dist_param = model.dist(coeff[2])

    Random.seed!(1234)
    # since it must be constant(Wₑ is a vector, σₑ² is a scalar)
    R   = model.R
    σₑ² = exp((model.σₑ² * coeff[5])[1])  
    e   = Matrix{T}(undef, data.nofobs, R)

    for i in data.rowidx
        e[i, 1:R] .= rand(Normal(0, sqrt(σₑ²)), 1, R)
    end

    simulate_ϵ = broadcast( 
        -,
        data.depvar - data.frontiers * coeff[1] -  model.xmean * coeff[4], 
        e
    )

    simulate_η = eta(
        data.rowidx, 
        data.econtype, 
        model.serialcorr, 
        simulate_ϵ, 
        coeff[6], 
        typeof(model), 
        coeff[2], 
        dist_param
    )

    return simulate_η, σᵥ², dist_param
end

#########################################################################################


#########################################################################################
# TODO: loglikelihood  function
# template functions are provided if needed
# notice that the type should be provide to utilize the multiple dispatch
# there is no need to reverse the sign because the minus symbol(-) will be added
# in structure/mle.jl
# LLT(ξ, model::AbstractModel, data::Data)
#########################################################################################

function LLT(ξ, model::SNCre, data::PanelData)
    coeff                       = slice(ξ, model.ψ, mle = true)
    simulate_η, σᵥ², dist_param = composite_error(coeff, model, data)

    rowidx  = data.rowidx
    lag     = lagparam(model)

    σᵥ²_ = drop_panelize(σᵥ², rowidx, lag)

    dist_type  = typeofdist(model)
    dist_param_ = map(
        x -> drop_panelize(x, rowidx, lag),
        dist_param 
    )

    # shift because we first lagdrop in `eta`
    simulate_η_ = static_panelize(simulate_η, newrange(rowidx, lag))


    @inbounds @floop for (i, val) in enumerate(zip(dist_param_...))

        simulate_llhᵢ = map(
            x -> _likelihood(dist_type, σᵥ²_[i], val..., x),
            eachcol(simulate_η_[i])
        )

        llhᵢ = log(mean(simulate_llhᵢ))
        llhᵢ = !isinf(llhᵢ) ? llhᵢ : -1e10

        @reduce llh = 0 + llhᵢ
    end

    return llh
end

#########################################################################################
