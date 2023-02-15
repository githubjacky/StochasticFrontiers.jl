"""
    error_simulation(σₑ²::Real, R::Int, inum::Int, tnum::Vector{Int})    

Simulate the error term of the individual effect, with `R` times

# Examples
```juliadoctest
julia> error_simulation(0.5, 250, 2, [3, 3])
6×250 Main.SFrontiers.Panel{Matrix{Float64}}:
 -0.56769    -0.600159  -0.361621  …  -0.128588  -0.512098   0.124228
 -0.56769    -0.600159  -0.361621     -0.128588  -0.512098   0.124228
 -0.56769    -0.600159  -0.361621     -0.128588  -0.512098   0.124228 
 -0.0315529   0.469108  -0.0538756     0.425247  -0.40964   -0.251104 
 -0.0315529   0.469108  -0.0538756     0.425247  -0.40964   -0.251104
 -0.0315529   0.469108  -0.0538756 …   0.425247  -0.40964   -0.251104
```
"""
function error_simulation(σₑ², R, inum, tnum)
    eᵢ = [
        rand(Normal(0, sqrt(σₑ²)), 1, R) 
        for _ in 1:inum
    ]
    e = reduce(
        vcat, 
        [
            repeat(i, inner=(t, 1))
            for (i,t) in zip(eᵢ, tnum)
        ]
    )
    return e
end


"""
    eta(porc, SCE::AR, simulate_ϵ, η_param)
    eta(porc, SCE::MA, simulate_ϵ, η_param)
    eta(porc, SCE::ARMA, simulate_ϵ, η_param)

Purely calculate the total composite error term

# Arguments
- `porc::Int`: economic interpretatino, if it's `Prod`, `porc`=1, while if it's the `Cost` case, `porc`=-1
- `SCE::AbstractSerialCorr`: serial correlation assumption
- `simulate_ϵ::Panel{T} where T`: simulated total composite_error
- `η_param::Tuple{Vector{<:Real}, AbstractDist, Vector{Union{Matrix{<:Real}, Panel{T} where T}}, Vector{<:Real}}`
    1. coefficient of serial correlation term(AR: ρ, MA: θ)
    2. type of distribution of inefficiency
    3. data of the distribution's parameters
    4. coefficient of the distribution's parameters
"""
function eta(porc, SCE::AR, simulate_ϵ, η_param)
    p = SCE()  # lag period

    # if lag period > 1, let all the element of ρ <  1 to ensure stationality
    # notice that the last coefficient is constant which will be decluded
    @inbounds rho = p == 1 ? η_param[begin] : (rho = coeffs(fromroots(η_param[begin]./(abs.(η_param[begin]).+1)))[begin:end-1])  
    
    base = lagdrop(simulate_ϵ, p)
    ar_terms = [ 
        lagshift(simulate_ϵ, i, totallag=p)
        for i in 1:p
    ]
    ar_factor = Panel(sum(ar_terms .* rho), base.rowidx)
    simulate_η =  Panelized(
        broadcast(*, porc, (base - ar_factor))
    )

    return simulate_η
end


"""
    dema(porc, simulate_ϵ, Eη, q, theta)

Subtract the `simulate_ϵ` from the moving average terms

# Arguments
- `proc::Int`
- `simulate_ϵ::PanelMatrix{T} where T`
- `Eη::Vector{<:Real}`: unconditional mean of  `η` for each individual
- `q::Int`
- `theta::Vector{<:Real}`
"""
function dema(porc, simulate_ϵ, Eη, q, theta)
    panelized_simulate_ϵ = Panelized(simulate_ϵ)
    simulate_η = Vector(undef, numberofi(simulate_ϵ))
    for i = eachindex(panelized_simulate_ϵ)
        T, R = nofobs(panelized_simulate_ϵ[i])
        ηᵢ = Matrix(undef, T+q, R)
        ηᵢ[begin:q, :] .= Eη[i]
        for j = q+1:T+q
            ηᵢ[j, :] = 
                panelized_simulate_ϵ[i][j-q, :] - 
                sum([
                    ηᵢ[j-k, :]*theta[k] 
                    for k = eachindex(theta)
                ])
        end
        simulate_η[i] = porc * ηᵢ[(begin+2*q):end, :]
    end
    simulate_η = Panelized(simulate_η, newrange(simulate_ϵ.rowidx, q))

    return simulate_η
end

function eta(porc, SCE::MA, simulate_ϵ, η_param)
    @inbounds Εη = mean.([
        uncondU(η_param[2], i..., η_param[4])
        for i in zip([Panelized(i) for i in η_param[3]]...)
    ])
    simulate_η = dema(porc, simulate_ϵ, Εη, SCE() , η_param[1])

    return simulate_η
end

function eta(porc, SCE::ARMA, simulate_ϵ, η_param)
    p, q = unpack(SCE)
    coeff, dist_type, dist_data, dist_coeff = η_param
    @inbounds rho, θ = η_param[1][begin:p], η_param[2][begin+p:end]
    p != 1 && (@inbounds rho = coeffs(fromroots(rho./(abs.(rho).+1)))[begin:end-1])

    base = lagdrop(simulate_ϵ, p)
    ar_terms = [ 
        lagshift(simulate_ϵ, lag, totallag=p)
        for lag in 1:p
    ]
    ar_factor = Panel(sum(ar_terms .* rho), base.rowidx)
    dear = base - ar_factor

    @inbounds Εη = mean.([
        uncondU(η_param[2], dist_dataᵢ..., η_param[4])
        for dist_dataᵢ in zip([Panelized(i) for i in η_param[3]]...)
    ])
    simulate_η = dema(porc, dear, Εη, q, θ)

    return simulate_η
end


function composite_error(ξ,  model::SNCre, data::PanelData)
    β, dist_coeff, Wᵥ, δ, Wₑ, ρ  = slice(ξ, model.ψ, mle=true)  # the order matters 

    σᵥ² = Panelized(
        lagdrop(broadcast(exp, data.σᵥ²*Wᵥ), model.SCE())
    )
    dist_param =[
        Panelized(lagdrop(i, model.SCE()))
        for i in data.dist(dist_coeff)
    ]

    Random.seed!(1234)
    σₑ² = exp((model.σₑ² * Wₑ)[1])  # since it must be constant(Wₑ is a vector, σₑ² is a scalar)
    simulate_e = error_simulation(σₑ², model.R, numberofi(data.depvar), numberoft(data.depvar))

    simulate_ϵ = broadcast(
        -, 
        (data.depvar - data.frontiers*β - model.xmean*δ), 
        simulate_e
    )
    η_param = (ρ, typeof(data.dist), unpack(data.dist), dist_coeff)
    simulate_η = eta(data.type(), model.SCE, simulate_ϵ, η_param)

    return simulate_η, σᵥ², dist_param
end


function LLT(ξ, model::SNCre, data::PanelData)
    simulate_η, σᵥ², _dist_param = composite_error(ξ, model, data)
    dist_param = [i for i in zip(_dist_param...)]

    dist_type = typeof(data.dist)
    @inbounds llike = [
        log(mean(
            [
                Base.prod(likelihood(
                    dist_type,
                    σᵥ²ᵢ, 
                    dist_paramᵢ..., 
                    simulate_ηᵢ[:, j]
                ))
                for j in axes(simulate_ηᵢ, 2)
            ]
        ))
        for (σᵥ²ᵢ, dist_paramᵢ, simulate_ηᵢ) in zip(σᵥ², dist_param, simulate_η)
    ]
    llike = broadcast(x -> isinf(x) ? -1e10 : x, llike)

    return -sum(llike)
end