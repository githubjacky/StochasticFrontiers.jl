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
src/models/{model}/main.jl.


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
The type to store the fitted result: [`SFresult`](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/types.jl#L186).
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
Below is the example to reset `warmstart_solver` and `tolerance`
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
there are some templates for these functions.

Let me introduce the whole structure taking models `Cross` and `SNCre` as the example.

### main.jl <a name=main></a>
#### basic setup <a name="basic_setup"></a>
1. define the model and dist, ψ and paramnames are three necessary fields
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

2. define the "undefined" model to ensure type stability
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

#### `spec` <a name=sfpec></a>
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/main.jl#L79)
- [SNCre](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/SNCre/main.jl#L166)

#### `modelinfo` <a name="modelinfo"></a>
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/main.jl#L113)
- [SNCre](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/SNCre/main.jl#L256)


### LLT.jl <a name=LLT></a>
#### `composite_error` <a name=composite_error></a>
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/LLT.jl#L6)
- [SNCre](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/SNCre/LLT.jl#L151)

#### `LLT` <a name=log_likelihood></a>
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/LLT.jl#L26)
- [SNCre](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/SNCre/LLT.jl#L198)


### extension.jl <a name=extension></a>
In this section, I will use `Cross` and `PFEWH` models to illustrate as both `Cross` and
`SNCre` take advantage of the template functions.

#### `jlmsbc` <a name=jlmsbc></a>
- template `jlmsbc`: [_jlmsbc](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/basic_equations.jl#L159)
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/extension.jl#L7)
- [PFEWH](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/PFEWH/extension.jl#L31)

#### utility function for bootstrap marginal effect <a name="bootstrap"></a>
- template `marginal_data`: [_marg_data](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/structure/extension.jl#L1)
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/extension.jl#L26)
- [PFEWH](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/PFEWH/extension.jl#L58)

- template `marginal_coeff`: [_marginal_coeff](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/structure/extension.jl#L25)
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/extension.jl#L30)
- [PFEWH](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/PFEWH/extension.jl#L62)

- template `marginal_label`: [_marginal_label](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/structure/extension.jl#L35)
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/extension.jl#L34)
- [PFEWH](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/PFEWH/extension.jl#L66)

- template `marginal_label`: [_unconditional_mean](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/basic_equations.jl#L218)
- [Cross](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/Cross/extension.jl#L37)
- [PFEWH](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/models/PFEWH/extension.jl#L84)

To have general idea about the structure, refer to [src/models/Cross](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/src/models/Cross)
