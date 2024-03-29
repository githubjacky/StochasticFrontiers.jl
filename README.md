# StochasticFrontiers
*the purpose of this repository is creating a bear bone structure for further development*


## Installation
Add StochasticFrontiers from the Pkg REPL, i.e., pkg> add https://github.com/githubjacky/StochasticFrontiers.jl


## Models
- `Cross`: basic stochastic frontier models
    - with half normal, truncated normal, exponential distribution assumption
- `PFEWH`: [Estimating fixed-effect panel stochastic frontier models by model transformation](https://www.sciencedirect.com/science/article/abs/pii/S0304407610000047)
- `SNCre`: [Flexible panel stochastic frontier model with serially correlated errors](https://www.sciencedirect.com/science/article/abs/pii/S0165176517304871)


## Table of Content
- [users](#users)
    - [fit the model](#fit_the_models)
        - [`sfspec`](#sfspec)
        - [`sfopt`](#sfopt)
        - [`sfinit`](#sfinit)
        - [`sfmodel_fit`](#sfmodel_fit)
    - [jlms and bc index](#jlms_bc)
    - [estimation results](#estimation_results)
    - [marginal effect](#marginal_effect)
- [developers](#developers)
    - [main.jl](#main)
        - [basic setup](#basic_setup)
        - [`spec`](#spec)
        - [`modelinfo`](#modelinfo)
    - [LLT.jl](#LLT)
        - [`composite_error`](#composite_error)
        - [`LLT`](#log_likelihood)
    - [extension.jl](#extension)
        - [`jlmsbc`](#jlmsbc)
        - [utility function for bootstrap marginal effect](#bootstrap)


## Users <a name="users"></a>
To see more usages, please check out the [examples/](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/examples) folser


### fit the model <a name = "fit_the_models"></a>


#### `sfspec`: wrapper for assigning the data and some model specific parameters <a name="sfspec"></a>
- examples in examples/Cross_Trun.ipynb
```julia
spec = sfspec(
    data      = "data/sampledata.csv", 
    model     = Cross(), 
    type      = Prod(), 
    dist      = Trun(μ = (:age, :school, :yr, :_cons), σᵤ² = (:age, :school, :yr, :_cons)),
    σᵥ²       = :_cons, 
    depvar    = :yvar, 
    frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
)
```
Generally, user should assign the source of data other than your input such as `depvar` and 
`frontiers` are already in matrix form. Some keyword arguments are necessary for all models:
1. data: optional, path of data
2. model: sort of model
3. type: economic assumption, can be either `Prod()` or `Cost()`
    - `Prod`: production, alias `prod`, `p`
    - `Cost`: cost, alias `cost`
4. dist: distribution assumption of inefficiency u, can be `Half()`, `Trun()` and `Exop()` 
    - `Half`: Half Normal, alias `half`
    - `Trun`: Truncated Normal, alias `trun`
    - `Expo`: Exponential, alias `expo`
5. σᵥ²: variance of noise
6. depvar: dependent variable
7. frontiwrs: explanantory varialbes
8. ivar: variable to indicate observations which are from same the individual but different 
periods, should be assigned in panel model

- another examples in examples/SNCre.ipynb
```julia
spec = sfspec(
    data       = "data/SNCreData.csv",
    model      = SNCre(), 
    type       = Prod(), 
    dist       = Half(σᵤ² = :_cons), 
    σᵥ²        = :_cons, 
    ivar       = :i, 
    depvar     = :log_y, 
    frontiers  = (:log_x1, :log_x2, :t), 
    serialcorr = AR(1), 
    R          = 250,     # for MSLE of correlated random effect
    σₑ²        = :_cons,  # noise for correlated random effect 
    verbose    = false
)
```

To see each model's specification, check out the `spec` in each model's main.jl located in
src/models/{ **model** }/main.jl and **model** is the argument you should specify.


#### `sfopt`: MLE optimization parameters <a name="sfopt"></a>
default options:
```julia
options = sfopt(
    warmstart_solver = Optim.NelderMead(),  # to forbidden warmup, set it to be `nothing`
    warmstart_maxIT  = 100,                 # maximum iterations for warmup optimiazation
    main_solver      = Optim.Newton(),      # main solver, default to `Optim.Newton()`
    main_maxIT,      = 2000,                # maximum iterations for main optimiazation
    tolerance        = 1e-8,                # converge if gradient < `tolerance`
    show_trace       = false,               # wheter to show the trace of the main optimiazation
    verbose          = true,                # wheter to print out messages or estimation results
    table_format     = :text                # format for output estimation results
)
```


#### `sfinit`: initial points <a name=sfinit></a>
The initial points can be assigned in two ways, assigning all or block by block.
- example in examples/SNCre.ipynb
```julia
init = sfinit([
    0.5227960098102571,        # coefficient of explanatory variables
    0.1868939866287993,        # coefficient of explanatory variables
    0.007442174221837823,      # coefficient of time effect
    -1.6397116052113527*2,     # log_σᵤ²
    -3.3244812689250423*2,     # log_σᵥ²
    0.3484365793340449,        # coefficient of fixed effect(mean of x1)
    -0.05768082007032795,      # coefficient of fixed effect2(mean of x2)
    -0.5943654485109733/14.5,  # costant term of fixed effect3(mean of x3)
    -0.8322378217931871*2,     # log_σₑ²
    0.,
])
```
- example in examples/Cross_Trun.ipynb
```julia
init = sfinit(
    μ       = fill(0.1, 4), 
    log_σᵤ² = fill(-0.1, 4), 
    log_σᵥ² = -0.1
)
```


#### `sfmodel_fit`: wrap the spec, options and init to avoid global variables <a name="sfmodel_fit"></a>
- examples in examples/Cross_Trun.ipynb
```julia
res = sfmodel_fit(
    spec = sfspec(
        data      = "data/sampledata.csv", 
        model     = Cross(), 
        type      = Prod(), 
        dist      = Trun(μ = (:age, :school, :yr, :_cons), σᵤ² = (:age, :school, :yr, :_cons)),
        σᵥ²       = :_cons, 
        depvar    = :yvar, 
        frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons)
    ),
    options = sfopt(warmstart_maxIT = 400),
    init = sfinit(
        μ       = fill(0.1, 4), 
        log_σᵤ² = fill(-0.1, 4), 
        log_σᵥ² = -0.1
    )
);
```


### jlms and bc index <a name="jlms_bc"></a>
Mean of the indices will be calculated automatically aftre fitting, and just remember to 
set the `verbose` to true to see full estimation results. To get the indeices, use the 
function `sf_inefficiency` and `sf_efficiency`.

### estimation results <a name="estimation_results"></a>
The type to store the estimation results: [`SFresult`](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/types.jl#L186).
Notice that there are two fields for `SFresult`. One is for the baisc storage and the other 
is model specific.

Some API to extract information from the `res::SFresult`
- `plot_inefficieny`: plot jlms, bc index
- `sfmaximizer`: maximum likelihood estimator
- `sfmodel::AbstractSFmodel`: model informations
- `sfdata::AbstractData`: general infromations 
- `sfstartpt`: initial points 
- `sfoptions`: estimation options
- `sf_inefficiency`: Jondrow et al. (1982) inefficiency index
- `sf_efficiency`: Battese and Coelli (1988) efficiency index
- `sfmaximum`: maximized log-likelihood
- `sfcheck_converge`: check the converge status provided by package [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
- `sftrace`: trace of the estimation procedure, and also provided by [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)

- `sfAIC`, `sfBIC`: aic and bic for model `SNCre`



### marginal effect <a name="marginal_effect"></a>
Examples can be found in exampeles/Cross_Trun.ipynb.
1. `sfmarginal`: marginal effect
```julia
marginal, marginal_mean = sfmarginal(res)
```

2. `sfmarginal_bootstrap`: bootstrap std of the mean marginal effect
```julia
std_ci, bsdata = sfmarginal_bootstrap(
    res, 
    R = 250, 
    seed = 123
);
``````
User can also set the MLE estimation options in `sfmarginal_bootstrap` as keyword arguments.
Below is the example to reset `warmstart_solver` and `tolerance`. For all options, refer to
[default options](#sfopt)
```julia
std_ci, bsdata = sfmarginal_bootstrap(
    res, 
    R = 250, 
    seed = 123, 
    warmstart_solver = nothing,
    tolerance = 1e-4
);
```

3. `sfCI`: confidence interval
```julia
ci = sfCI(bsdata, marginal_mean, level = 0.1)
```
`bsdata` is the second return of `sfmarginal_bootstrap` and the `marginal_mean` is the
second return of `sfmarginal`.

## developers
To develop a new model, check out the [src/models/template/](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/src/models/template)
folders. You can simply copy the **template** folder and rename it.

Basically, there are three files should be created: main.jl, LLT.jl and extension.jl. Some 
function is necessary to implement such as the `spec` and `modelinfo` in main.jl, 
`composite_error` and `LLT` in LLT.jl and `jlmsbc`, `marginal_data`, `marginal_coeff`,
`marginal_label` and `unconditional_mean` in extension.jl. For quick development,
there are some templates for these functions. Don't forget to `include` the main.jl of your
model in src/StochasticFrontiers.jl.

Let me introduce the whole structure taking model `Cross` and `SNCre` as the examples.


### main.jl <a name=main></a>
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/main.jl)
- [SNCre](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/SNCre/main.jl)

#### basic setup <a name="basic_setup"></a>
1. define the model 

`dist`, `ψ` and `paramnames` are three necessary fields. Each new model type is the subtype
of `AbstractSFmodel` or `AbstractPanelModel` and `AbstractPanelModel` <: `AbstractSFmodel`. 
```julia
# Cross
struct Cross{T<:AbstractDist} <: AbstractSFmodel
    dist::T
    ψ::Vector{Int64}
    paramnames::Matrix{Symbol}
end

# SNCre
struct SNCre{T, S} <: AbstractPanelModel
    dist::T
    ψ::Vector{Int64}
    paramnames::Matrix{Symbol}
    serialcorr::S
    R::Int64
    σₑ²::Float64
    xmean::Matrix{Float64}
end
```

2. define the "undefined" type to ensure type stability
```julia
# Cross
struct UndefCross <: AbstractUndefSFmodel end
Cross() = UndefCross()
(::UndefCross)(args...) = Cross(args...) 

# SNCre
struct UndefSNCre <: AbstractUndefSFmodel end
SNCre() = UndefSNCre()
(::UndefSNCre)(args...) = SNCre(args...) 
```

3. bootstrap re-construction rules

The definition of the `resample` function for reconstruct the `AbstractDist` during bootstrap
procedure liess here: [definition](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/types.jl#L95)
```julia
# Cross
function (a::Cross)(selected_row, ::Data)
    bootstrap_model = Cross(
        resample(a.dist, selected_row),
        a.ψ,
        a.paramnames
    )

    return bootstrap_model
end

# SNCre
function (a::SNCre)(selected_row, data::PanelData)
    return SNCre(
        resample(a.dist, selected_row),
        a.ψ,
        a.paramnames,
        a.serialcorr,
        a.R,
        a.σₑ²,
        meanofx(data.rowidx, data.frontiers; verbose = false)
    )

end
```

4. model specific result

Developers should define the rule for multiple dispatch on function `SFresult`
```julia
# Cross
struct Crossresult <: AbstractSFresult end

function SFresult(main_res::MainSFresult{T, S, U, V}) where{T<:Cross, S, U, V}
    return SFresult(main_res, Crossresult())
end


# SNCre
struct SNCreresult <: AbstractSFresult
    aic::Float64
    bic::Float64
end

function SFresult(main_res::MainSFresult{T, S, U, V}) where{T<:SNCre, S, U, V} 
    aic = -2 * main_res.loglikelihood + numberofparam(main_res.model)
    bic = -2 * main_res.loglikelihood + numberofparam(main_res.model) * log(numberofobs(main_res.data))
    return SFresult(main_res, SNCreresult(aic, bic))
end

sfAIC(a::SFresult) = round(a.model_res.aic, digits = 5)
sfBIC(a::SFresult) = round(a.model_res.bic, digits = 5)
```
`numberofparam` and `numberofobs` is the self-defined utility functions. All the utility
functions are defined in src/utils.jl

#### `spec` <a name=sfpec></a>
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/main.jl#L79)
- [SNCre](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/SNCre/main.jl#L166)

The interface will be:
```julia
function spec(model::AbstractUndefSFmodel, df; 
              type, dist, σᵥ², depvar, frontiers, 
              kwargs...
              verbose = true
             )
...
end
```
The `AbstractUndefSFmodel` should various across different model. For instance, the `Cross`
model is `UndefCross` while `SNCre` is `UndefSNCre`. Besides, `kwargs...` should be define
as well.

The purpose of `spec`:
1. extract the data from .csv file and ensure there is no multicollinearity
2. create names for variables in output estimation table
3. define the rule for slicing parameters
    - in optimization process, all parameters is clustered into a vector


#### `modelinfo` <a name="modelinfo"></a>

```julia
# Cross
function modelinfo(::Cross)
    modelinfo1 = "Base stochastic frontier model"
    
    modelinfo2 =
"""
    Yᵢ = Xᵢ*β + + ϵᵢ
        where ϵᵢ = vᵢ - uᵢ

        further,     
            vᵢ ∼ N(0, σᵥ²),
            σᵥ²  = exp(log_σᵥ²)

            uᵢ ∼ N⁺(0, σᵤ²),
            σᵤ² = exp(log_σᵤ²)

    In the case of type(cost), "- uᵢₜ" above should be changed to "+ uᵢₜ"
"""
    _modelinfo(modelinfo1, modelinfo2)
end


# SNCre
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
```

### LLT.jl <a name=LLT></a>
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/LLT.jl)
- [SNCre](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/SNCre/LLT.jl)

#### `composite_error` <a name=composite_error></a>
```julia
# Cross
function composite_error(coeff::Vector{Vector{T}}, model::Cross, data) where T
    σᵥ²        = exp.(data.σᵥ² * coeff[3])
    dist_param = model.dist(coeff[2])
    ϵ          = (data.econtype * (data.depvar- data.frontiers*coeff[1]))[:, 1]
    
    return ϵ, σᵥ², dist_param
end

# SNCre
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
```

#### `LLT` <a name=log_likelihood></a>
```julia
# Cross
function LLT(ξ, model::Cross, data::Data)
    coeff              = slice(ξ, model.ψ, mle=true)
    ϵ, σᵥ², dist_param = composite_error(coeff, model, data)

    return _loglikelihood(typeofdist(model), σᵥ², dist_param..., ϵ)
end

# SNCre
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

```
`slice`, `drop_panelize`, `static_panelize`, `newrange` are utility functions can be found
in src/utils.jl. `_likelihood` is the template function can be found in src/basic_equations



### extension.jl <a name=extension></a>
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/extension.jl)
- [SNCre](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/SNCre/extension.jl)

In this section, I will use `Cross` and `PFEWH` models to illustrate as both `Cross` and
`SNCre` take advantage of the template functions.

#### `jlmsbc` <a name=jlmsbc></a>
- template `jlmsbc`: [_jlmsbc](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/basic_equations.jl#L159)
```julia
# Cross
function jlmsbc(ξ, model::Cross, data::Data)
    coeff              = slice(ξ, model.ψ, mle=true)
    ϵ, σᵥ², dist_param = composite_error(coeff, model, data)

    jlms, bc = _jlmsbc(typeofdist(model), σᵥ², dist_param..., ϵ)::NTuple{2, Vector{Float64}}
    return jlms, bc
end

# PFEWH
function PFEWH_jlmsbc(rowidx, σᵥ², μ, σᵤ², ϵ, h)

    ϵ̃  = demean_panelize(ϵ, rowidx)
    h_ = static_panelize(h, rowidx)
    h̃  = demean_panelize(h, rowidx)

    N    = sum(length.(ϵ̃))
    jlms = zeros(N)
    bc   = zeros(N)

    @floop for (ϵ̃ᵢ, h̃ᵢ, hᵢ, idx) in zip(ϵ̃, h̃, h_, rowidx)
        σₛₛ² = 1.0 / ( h̃ᵢ' * h̃ᵢ * (1/σᵥ²) + 1/σᵤ² )
        σₛₛ  = sqrt(σₛₛ²)
        μₛₛ  = ( μ/σᵤ² - ϵ̃ᵢ' * h̃ᵢ * (1/σᵥ²) ) * σₛₛ²

        jlms[idx] = @. hᵢ * ( μₛₛ + normpdf(μₛₛ/σₛₛ) * σₛₛ / normcdf(μₛₛ/σₛₛ) )
        bc[idx]   = @. ( (normcdf(μₛₛ/σₛₛ-hᵢ*σₛₛ)) / normcdf(μₛₛ/σₛₛ) ) * 
                       exp( -hᵢ * μₛₛ + 0.5 * (hᵢ^2) * σₛₛ² )
    end

    return jlms, bc
end

function jlmsbc(ξ, model::PFEWH, data::PanelData)
    ϵ, σᵥ², dist_param, h = composite_error(
        slice(ξ, model.ψ, mle=true),
        model, 
        data
    )
    jlms, bc = begin
        model.dist isa Trun ?
        PFEWH_jlmsbc(data.rowidx, σᵥ², dist_param..., ϵ, h) :
        PFEWH_jlmsbc(data.rowidx, σᵥ², 0., dist_param..., ϵ, h)
    end

   return jlms, bc
end
```
Some models should define calculation of the jlms, bc index such as `PFEWH`. In this case,
the `jlmsbc` can't rely on the template function.


#### utility function for bootstrap marginal effect <a name="bootstrap"></a>
- template `marginal_data`: [_marg_data](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/structure/extension.jl#L1)
```julia
# Cross
marginal_data(model::Cross) = _marg_data(model)

# PFEWH
marginal_data(model::PFEWH) = _marg_data(model, :hscale)
```

- template `marginal_coeff`: [_marginal_coeff](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/structure/extension.jl#L25)
```julia
# Cross
marginal_coeff(::Type{<:Cross}, ξ, ψ) = _marginal_coeff(ξ, ψ)

# PFEWH
marginal_coeff(::Type{<:PFEWH}, ξ, ψ) = slice(ξ, ψ, mle=true)[[2, 4]]
```

- template `marginal_label`: [_marginal_label](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/structure/extension.jl#L35)
```julia
# Cross
marginal_label(model::Cross, k) = _marginal_label(model, k)

# PFEWH
marginal_label(model::PFEWH, k) = _marginal_label(model, k, :log_hscale)
```

- template `unconditional_mean`: [_unconditional_mean](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/basic_equations.jl#L218)
```julia
# Cross
unconditional_mean(::Type{Cross{T}}, coeff, args...) where T = _unconditional_mean(T, coeff, args...)

# PFEWH
function PFEWH_unconditional_mean(coeff, μ, log_σᵤ², _log_h)

    Wμ = μ == zeros(1) ? zeros(1) : coeff[1][begin:begin+length(μ)-1]
    Wᵤ = coeff[1][end-length(log_σᵤ²)+1:end]

    log_h = _log_h' * coeff[2]

    μ       = exp(log_h) * (μ' * Wμ)
    σᵤ      = exp(log_h + 0.5 * log_σᵤ²' * Wᵤ)
    Λ       = μ / σᵤ
    uncondU = σᵤ * ( Λ + normpdf(Λ) / normcdf(Λ) )

    return uncondU
end

function unconditional_mean(::Type{PFEWH{Half{T}}}, coeff, log_σᵤ², log_h) where T
    return PFEWH_unconditional_mean(coeff, [0.], log_σᵤ², log_h)
end

function unconditional_mean(::Type{PFEWH{Trun{T, S}}}, coeff, μ, log_σᵤ², log_h) where{T, S}
    return PFEWH_unconditional_mean(coeff, μ, log_σᵤ², log_h)
end
```

To have general idea about the structure, refer to [src/models/Cross](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/src/models/Cross)
