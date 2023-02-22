# serial correlation
abstract type AbstractSerialCorr end
struct AR <: AbstractSerialCorr
    p::Integer
end

struct MA <: AbstractSerialCorr
    q::Integer
end

struct ARMA <: AbstractSerialCorr
    p::Integer
    q::Integer
end

lagparam(a::AR) = getproperty(a, :p)
lagparam(a::MA) = getproperty(a, :q)
lagparam(a::ARMA) = getproperty(a, :p) + getproperty(a, :q)
lagparam(a::PanelModel) = lagparam(get_serialcorr(a))


"""
   SNCreData <: AbstractModelData

Model Specific data which need to be check for the multicollinearity. It's not
necessay for each model.
"""
struct SNCreData <: AbstractModelData end  # initial a empty model specifc data type


"""
    SNCre(SCE, R, σₑ², xmean, ψ, paramnames, data)

# Arguments
- `SCE::AbstractSerialCorr`: serial correlation
- `R::Int`: number of simulation of error of controled random effect `e`
- `σₑ²::Real`: vaiance of the `e`
- `xmean::PanelMatrix`: explanatory variables of the controled random effect
- `ψ::Vector{Any}`: record the length of each parameter, `ψ[end]` is the arrgregate length of all parameters
- `paramnames::Matrix{Symbol}`: parameters' names used by the output estimation table
- `data::SNCreData`

"""
struct SNCre{T<:AbstractSerialCorr,
             S<:Real,
             U<:PanelMatrix,
            } <: PanelModel
    serialcorr::T
    R::Int
    σₑ²::S
    xmean::U
    ψ::Vector{Any}
    paramnames::Matrix{Symbol}
    data::SNCreData
end

get_R(a::SNCre) = getproperty(a, :R)


"""
    meanofx(frontiers::PanelMatrix{T} where T)

To estimate the correlated random effect

# Examples
```juliadoctest
julia> ivar = [1, 1, 1, 2, 2]; X=[3 5; 4 7; 5 9; 8 2; 3 1]
5×2 Matrix{Int64}:
 3  5
 4  7
 5  9
 8  2
 3  1

julia> tnum = [length(findall(x->x==i, ivar)) for i in unique(ivar)]
2-element Vector{Int64}:
 3
 2

julia> X = Panel(X, tnum=tnum)
5×2 Main.SFrontiers.Panel{Matrix{Int64}}:
 3  5
 4  7
 5  9
 8  2
 3  1

julia> cleanX, pivots = meanOfX(X); cleanX
* Find Multicollinearity

number 3 column in xmean is dropped
5×2 Matrix{Float64}:
4.0  7.0
4.0  7.0
4.0  7.0
5.5  1.5
5.5  1.5
```
"""
function meanofx(frontiers)
    mean_frontiers = mean.(Panelized(frontiers), dims=1)
    noft = numberoft(frontiers)
    _xmean = [repeat(i,t) for (i,t) in zip(mean_frontiers, noft)]
    xmean = reduce(vcat, _xmean)
    cons = ones(numberofobs(xmean))

    # if the panel is balanced, the average of variable t is constant
    # `real` ensure the element type is not any which will raise error in `rref_with_pivots`
    # `reduce` for fast `hcat` operation
    xmean, pivots = isMultiCollinearity(
        :xmean, real(reduce(hcat, (xmean, cons)))
    )
    return Panel(xmean, frontiers.rowidx), pivots
end


"""
    sfspec(::Type{SNCre}, <arguments>; type, dist, σᵥ², ivar, depvar, frontiers, SCE, R, σₑ²)

The model: 

# Arguments
- `data::Union{Tuple{DataFrame}, Tuple{}}`: frame or matrix data
- `type::Union{Type{Production}, Type{Cost}}`: type of economic interpretation
- `dist::Tuple{Union{Half, Trun, Expo}, Vararg{Union{Symbol, Matrix{T}}}} where{T<:Real}`: assumption of the inefficiency
- `σᵥ²::Union{Matrix{T}, Union{Symbol, NTuple{N, Symbol} where N}}`: 
- `ivar::Union{Vector{<:Real}, Sumbol}`: specific data of panel model
- `depvar::Union{AbstractVecOrMat{<:Real}, Symbol}`: dependent variable
- `frontiers::Union{Matrix{<:Real}, NTuple{N, Symbol} where N}`: explanatory variables
- `SCE::Union{AR, MA, ARMA}`: assumption of serial correlation
- `R::Int`: number of correlated random effect simulation
- `σₑ²::Union{Real, Symbol}`: variance of the random error of the correlated random effect
"""
function sfspec(::Type{SNCre}, data...; type, dist, σᵥ², ivar, depvar, frontiers, serialcorr, R, σₑ²)
    # get the variables and set up σₑ²
    paneldata, _col1, _col2 = getvar(data, ivar, type, dist, σᵥ², depvar, frontiers)
    @inbounds σₑ² = isa(σₑ², Symbol) ? Base.getindex(data[1], :, σₑ²)[1] : σₑ²[1] # since σₑ² will always be constant
    
    # generate the mean data of frontiers for the specification of correlated random effect
    xmean, pivots = meanofx(paneldata.frontiers)
    
    # construct remaind first column of output estimation table
    corrcol1 = isa(serialcorr, AR) ? (:ρ,) : (isa(serialcorr, MA) ? (:θ,) : (:ρ, :θ))
    col1 = complete_template(_col1, [:αᵢ, :log_σₑ², corrcol1...])

    # construct remaind second column of output estimation tabel
    if !isa(serialcorr, ARMA)
        var = corrcol1[1]
        @inbounds corrcol2 = [[Symbol(var, i) for i = 1:lagparam(serialcorr)]]
    else
        var1, var2 = corrcol1[1], corrcol1[2]
        @inbounds corrcol2 = [
            [Symbol(var1, i) for i = 1:serialcorr.p], 
            [Symbol(var2, i) for i = 1:serialcorr.q]
        ]
    end
    xmean_col2 = [Symbol(:mean_, i) for i in _col2[1]]
    push!(xmean_col2, :_cons)
    xmean_col2 = xmean_col2[pivots]
    col2 = complete_template(_col2, [xmean_col2, [:_cons], corrcol2...])

    # costruct the names of parameters of the output estimation table
    paramnames = paramname(col1, col2)

    # generate the remain rule for slicing parameter
    # generate the length of serially correlated error terms, σₑ² and correlated random effects
    ψ = complete_template(Ψ(paneldata), [numberofvar(xmean), 1, lagparam(serialcorr)])
    push!(ψ, sum(ψ))

    return SNCre(serialcorr, R, σₑ², xmean, ψ, paramnames, SNCreData()), paneldata
end


(t::AR)(::Symbol) = Symbol(typeof(t), t.p)
(t::MA)(::Symbol) = Symbol(typeof(t), t.q)
function (t::ARMA)(::Symbol)
    str = string(typeof(t))
    return Symbol(str[1:2], t.p, str[3:4], t.q)
end

function (t::AR)(::String)
    ϵᵢₜ = ""
    for i = 1:t.p
        ϵᵢₜ *= "ρ$i * ϵᵢₜ₋$i  + "
    end
    ϵᵢₜ *= "ηᵢₜ"
    return ϵᵢₜ
end
function (t::MA)(::String)
    ϵᵢₜ = ""
    for i = 1:t.q
        ϵᵢₜ *= "θ$i * ηᵢₜ₋$i  + "
    end
    ϵᵢₜ *= "ηᵢₜ"
    return ϵᵢₜ
end
function (t::ARMA)(::String)
    ϵᵢₜ = ""
    for i = 1:t.p
        ϵᵢₜ *= "ρ$i * ϵᵢₜ₋$i  + "
    end
    for i = 1:t.q
        ϵᵢₜ *= "θ$i * ηᵢₜ₋$i  + "
    end
    ϵᵢₜ *= "ηᵢₜ"
    return ϵᵢₜ
end

function modelinfo(model::SNCre)
    modelinfo1 = "flexible panel stochastic frontier model with serially correlated errors"
    
    SCE = model.serialcorr
    modelinfo2 =
"""
    Yᵢₜ = αᵢ + Xᵢₜ*β + T*π + ϵᵢₜ
        where αᵢ = δ₀ + X̄ᵢ'* δ₁ + eᵢ,

        and since the serial correlated assumptions is $(SCE(:type)),
            ϵᵢₜ = $(SCE("info"))
            ηᵢₜ = vᵢₜ - uᵢₜ

        further,     
            vᵢₜ ∼ N(0, σᵥ²),
            σᵥ²  = exp(log_σᵥ²)

            uᵢₜ ∼ N⁺(0, σᵤ²),
            σᵤ² = exp(log_σᵤ²)

            eᵢ ∼ N(0, σₑ²)
            σᵤ² = exp(log_σₑ²)

    In the case of type(cost), "- uᵢₜ" above should be changed to "+ uᵢₜ"
"""
    _modelinfo(modelinfo1, modelinfo2)
end


# other module
include("./LLT.jl")
include("extension.jl")