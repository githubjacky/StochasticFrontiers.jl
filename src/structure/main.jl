usedata(data::String) = DataFrame(CSV.File(data))
usedata(data)         = data


"""
    sfspec(; data, model, kwargs...)

This is the highest api to wrap the specification of model. Multiple dispatch is implemented
after the `model` is evaluated which can be found in model specific function `spec`.

"""
sfspec(; data = nothing, model, kwargs...) = spec(model, usedata(data); kwargs...)


"""
    sfopt(;warmstart_solver = Optim.NelderMead(), 
           warmstart_maxIT  = 100, 
           main_solver      = Optim.Newton(), 
           main_maxIT,      = 2000,
           tolerance        = 1e-8, 
           show_trace       = false,
           verbose          = true, 
           table_format     = :text
          )

Define Hyperparmeters for the MLE estimatation. The `solver` is the optimiazation Algorithm,
and to see different algorithms, check out Optim.jl. The option of `table_format` is used
by PrettyTables.jl. Possible selections are: :text, :html, :latex. Moreover, Users can 
decide to store the trace in optimiazation procedure or see some provided information 
such as the Muliticollinearity by set `verbose` to `true`.

# Arguments
- `warmstart_solver::Optim.AbstractOptimizer`: to forbidden warmup, set it to be `nothing`
- `warmstart_maxIT::Int64`                   : maximum iterations for warmup optimiazation
- `main_solver::Optim.AbstractOptimizer`     : main solver, default to `Optim.Newton()`
- `main_maxIT::Int64`                        : maximum iterations for main optimiazation
- `torlerance::Float64`                      : converge if gradient < `tolerance`
- `show_trace::Bool`                         : wheter to show the trace of the main optimiazation
- `verbose::Bool`
- `table_format::Symbol`

"""
function sfopt(;warmstart_solver = Optim.NelderMead(),
                warmstart_maxIT  = 100,
                main_solver      = Optim.Newton(),
                main_maxIT       = 2000,
                tolerance        = 1e-8,
                show_trace       = false,
                verbose          = true,
                table_format     = :text
              )

    opt = (
        warmstart_solver = warmstart_solver,
        warmstart_maxIT  = warmstart_maxIT,
        main_solver      = main_solver,
        main_maxIT       = main_maxIT,
        tolerance        = tolerance,
        show_trace       = show_trace,
        verbose          = verbose,
        table_format     = table_format
    )

    return opt
end


"""
    reset_options(tp::NamedTuple; kwargs...)

It's used in `sfmarginal_bootstrap` to simply given the same kewyward arguments, options
can be modified in bootstrap.

# Examples
```juliadoctest
julia> opt = (
           warmstart_solver = Optim.NelderMead(),
           warmstart_maxIT  = 200,
           main_solver      = Optim.Newton(),
           main_maxIT       = 2000,
           tolerance        = 1e-8,
           verbose          = true,
           table_format     = :text
        );


julia> new_opt = reset_options(opt; warmstart_solver = nothing, main_maxIT = 30);

julia> new_opt.main_maxIT
30
```

"""
function reset_options(tp; kwargs...)
    key = keys(tp)
    val = collect(values(tp))

    idx = Int64[findfirst(x->x==i, key) for i in keys(kwargs)]
    val[idx] .= values(values(kwargs)) 

    return NamedTuple{key}(tuple(val...))
end


"""
    sfinit(init::Vector{<:Real})
    sfinit(; <keyword arguments>)

Two methods to provide initial condition:
1. given the **all** initial values of parameters
2. given only some of the initial values and it's necessary to assign the keyword
   arguments

# Examples
```juliadoctest
julia> sfinit([0.5,0.1, 0.3])
3-element Vector{Float64}:
 0.5
 0.1
 0.3

julia> sfinit(log_σᵤ²=(-0.1, -0.1, -0.1, -0.1), log_σᵥ²=-0.1)
(log_σᵤ² = (-0.1, -0.1, -0.1, -0.1), log_σᵥ² = -0.1)
```
"""
sfinit(init::AbstractVector) = init
sfinit(;kwargs...) = values(kwargs)


"""
    olsinfo(frontiers::AbstractMatrix{<:Real}, depvar::AbstractMatrix{<:Real})

Calculate coefficient, log likelihood, skewnewss of OLS
"""
function olsinfo(frontiers, depvar, nofx, nofobs, noffixed)
    β0    = frontiers \ depvar
    resid = depvar - frontiers*β0
    sse   = sum(resid .^ 2)  
    ssd   = sqrt( sse / ( nofobs - (nofx+noffixed) ) ) # sample standard deviation; σ² = (1/(N-K))* Σ ϵ^2
    ll    = sum(normlogpdf.(0, ssd, resid)) # ols log-likelihood
    sk    = sum(resid.^3 / (ssd^3*nofobs)) # skewnewss of ols residuals

    return β0, ll, sk
end


"""
    initial_condition(froniters, depvar, init::Vector, nofx, nofobs; kwargs...)
    initial_condition(froniters, depvar, init::NamedTuple, nofx, nofobs; kwargs...)
    initial_condition(froniters, depvar, init::Nothing, nofx, nofobs; kwargs...)

This function are used by the method `sfmodel_fit` to set the initial condition for 
coefficient of explanatory variables. 
The procedure is first calculating the ols estimator for `frontiers` through `olsinfo` and 
then set 0.1 to be the default initial condition for all the other parameters.

"""
function initial_condition(frontiers, depvar, init::AbstractVector, nofx, nofobs, noffixed; kargs...)
    _, llols, skols =  olsinfo(frontiers, depvar, nofx, nofobs, noffixed)

    return init, llols, skols
end

function initial_condition(frontiers, depvar, init::NamedTuple, nofx, nofobs, noffixed; kwargs...)
    β0, llols, skols =  olsinfo(frontiers, depvar, nofx, nofobs, noffixed)

    startpt          = fill(0.1, kwargs[:nofparam])
    startpt[1:nofx] .= β0
    template         = kwargs[:paramnames]

    push!(template, :end)  # to prevent can't getnext

    for i in keys(init)
        beg              = findfirst(x->x==i, template)
        en               = findnext(x->x!=Symbol(""), template, beg+1) - 1
        startpt[beg:en] .= Base.getproperty(init, i)
    end

    return startpt, llols, skols
end

function initial_condition(frontiers, depvar, ::Nothing, nofx, nofobs, noffixed; kwargs...)
    β0, llols, skols =  olsinfo(frontiers, depvar, nofx, nofobs, noffixed)

    startpt          = fill(0.1, kwargs[:nofparam])
    startpt[1:nofx] .= β0

    return startpt, llols, skols
end


"""
    sfmodel_fit(;spec, options, init)

# Arguments
- `spec::Tuple{<:AbstractSFmodel, <:AbstractData}`
- `options::OrderedDict{Symbol, Any}`
- `init::Union{Vector{<:Real}, NamedTuple{Symbol, <:Tuple{Vararg{Any}}}}`


There are three main process:
1. set the hyperparmeters with `options` for mle estimatation, set the initial initial condition with `init`
2. mle estimation
3. some extension(e.g. inefficiency index, marginal effect...)

"""
function sfmodel_fit(;spec,
                      options = nothing,
                      init    = nothing
                    )
    model, data = spec
    # set the mle hyperparmeters
    opt = options isa Nothing ? sfopt() : options

    # only TFE_WH2010, TFE_CSW2014 should be adjust to numberofi
    noffixed = 0

    # set the initial condition
    startpt, llols, skols = initial_condition(
        data.frontiers, 
        data.depvar, 
        init, 
        model.ψ[1], 
        numberofobs(data), 
        noffixed;
        nofparam   = numberofparam(model), 
        paramnames = model.paramnames[:, 1]
    )

    opt.verbose && modelinfo(model)
    func, ξ, warmup_iter, main_opt = mle(model, data, opt, startpt)
    
    # output the mle optimiazation results
    diagonal = post_estimation(func, ξ)

    # efficiency and inefficiency index
    jlms, bc = jlmsbc(ξ, model, data)
    loglikelihood = round(-1*Optim.minimum(main_opt); digits=5)

    if opt.verbose
        output_estimation(numberofobs(data), warmup_iter, main_opt, opt.tolerance)
        output_table(model.paramnames, ξ, diagonal, numberofobs(data), opt.table_format)
        println("Table format: $(opt.table_format). Use `sfopt(...)` to choose between `:text`, `:html`, and `:latex`.")

        printstyled("\n\n*********************************\n "; color=:cyan)
        printstyled("    Additional Information     \n"; color=:cyan); 
        printstyled("*********************************\n\n"; color=:cyan)
        println(" - OLS (frontier-only) log-likelihood:             $(round(llols, digits=5))")
        println(" - Skewness of OLS residuals:                      $(round(skols, digits=5))")
        println(" - The sample mean of the JLMS inefficiency index: $(round(mean(jlms), digits=5))")
        println(" - The sample mean of the BC efficiency index:     $(round(mean(bc), digits=5))")
        println("")
        println(" - Check out the availabel API in file: structure/api.jl")
        println("     - `res`                 : the return of sfmodel_fit, `res = sfmodel_fit(...)`")
        println("     - `sfmaximum(res)`      : the log-likelihood value of the model;")
        println("     - `sf_inefficiency(res)`: Jondrow et al. (1982) inefficiency index;")
        println("     - `sf_efficiency(res)`  : Battese and Coelli (1988) efficiency index;")
        println(" - Check out more examples in : examples/ \n")
        printstyled("*********************************\n\n"; color=:cyan)
    end

    # res = SFresult(ξ, model, data, opt, jlms, bc, loglikelihood, main_opt)
    res = SFresult(ξ, model, data, opt, jlms, bc, loglikelihood, main_opt)

    return res
end
