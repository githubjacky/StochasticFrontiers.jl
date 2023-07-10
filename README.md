# StochasticFrontiers

*the purpose of this repository is creating a bear bone structure for further development*

## install
Add StochasticFrontiers from the Pkg REPL, i.e., pkg> add https://github.com/githubjacky/StochasticFrontiers.jl

## models
- `Cross`: basic stochastic frontier models
    - with half normal, truncated normal, exponential distribution assumption
- `PFEWH`: [Estimating fixed-effect panel stochastic frontier models by model transformation](https://www.sciencedirect.com/science/article/abs/pii/S0304407610000047)
- `SNCre`: [Flexible panel stochastic frontier model with serially correlated errors](https://www.sciencedirect.com/science/article/abs/pii/S0165176517304871)

## users
### fit the model 
#### `sfspec`: wrapper for assigning the data and some model specific parameters.
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
    R          = 250, 
    σₑ²        = :_cons,
    verbose    = false
)
```

To see each model's specification, check of the `spec` in each model's main.jl located in
src/models/{model}/main.jl.


#### `sfopt`: MLE optimization parameters
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


#### `sfinit`: initial points
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


#### `sfmodel_fit`: wrap the spec, options and init to avoid global variables
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


#### jlms and bc index
Mean of the indices will be calculated automatically aftre fitting, and just remember to 
set the `verbose` to true to see full estimation results. To get the indeices, use the 
function `sf_inefficiency` and `sf_efficiency`.

#### estimation results
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

To see more usages, please check out the [examples/](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/examples) folser


#### marginal effect
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
To develop a new model, check out the [src/models/template/](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/src/models/template) folders. You can simply copy the **template** folder and rename it.

Conceptually, there are three files should be create: main.jl, LLT.jl, extension.jl
- main.jl: basic specification such as the model type, bootstrap reconstruciton rules.
- LLT.jl: composite error term and log-likelihood
- extension.jl: jlms and bc index, marginal effect

To have general idea about the structure, refer to [src/models/Cross](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/src/models/Cross)
