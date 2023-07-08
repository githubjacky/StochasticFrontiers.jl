# StochasticFrontiers

*the purpose of this repository is creat a bear bone structure for further development*

## install
Add StochasticFrontiers from the Pkg REPL, i.e., pkg> add https://github.com/githubjacky/StochasticFrontiers.jl

## models
- basic stochastic frontier models with half normal, truncated normal, exponential distribution assumption
- `PFEWH`: [Estimating fixed-effect panel stochastic frontier models by model transformation](https://www.sciencedirect.com/science/article/abs/pii/S0304407610000047)
- `SNCre`: [Flexible panel stochastic frontier model with serially correlated errors](https://www.sciencedirect.com/science/article/abs/pii/S0165176517304871)

## users
- fit the model: `sfspec`, `sfopt`, `sfinit`, `sfmodel_fit`
- marginal effect, bootstrap marginal effect and construct the confidence interval for bootstraping:
    - `sfmarginal`, `sfCI`, `sfmarginal_bootstrap`
- the type to store the fitted result: [`SFresult`](https://github.com/githubjacky/StochasticFrontiers.jl/blob/main/src/types.jl#L173)
    - notice that there are two types one for the baisc storage the other is model specific
- api to extract information for the `result::SFresult`
    - `plot_inefficieny`, `sfmaximizer`, `sfmodel`, `sfdata`, `sfstartpt`, `sfoptions`, `sf_inefficiency`, `sf_efficiency`, `sfmaximum`, `sfcheck_converge`, `sftrace`
    - for `SNCre model`: `sfAIC`, `sfBIC`
- for the usage please check out the [examples/](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/examples) folser

## developers
to develop a new model, check out the [src/models/template/](https://github.com/githubjacky/StochasticFrontiers.jl/tree/main/src/models/template) folders
