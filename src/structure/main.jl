usedata(data_path::String) = DataFrame(File(data_path))
usedata(df::DataFrame) = df
usedata() = nothing


"""
    setindex!(A::OrderedDict{T, U}, X::Union{Vector{T}, NTuple{N, T} where N}, ind::Union{Vector{U}, NTuple{M, U} where M}) where{T, U}

Oordered dictionary utilities to set multiple indexes in cleaner way

# Examples
```juliadoctest
julia> example = OrderedDict{Symbol, Int}(:a=>3, :b=>2)
OrderedCollections.OrderedDict{Symbol, Int64} with 2 entries:
  :a => 3
  :b => 2

julia> setindex!(example, (4, 4), (:a, :b)); example
OrderedDict{Symbol, Int64} with 2 entries:
  :a => 4
  :b => 4

See also: [`getindex`](@ref), [`sfopt`](@ref)
```
"""
function setindex!(A, X, ind)
    defaultKey = keys(A)
    for i in eachindex(X)
        j = ind[i]
        j in defaultKey || throw("misspecification of the keywords argmentus $j in OrderedDict") 
        Base.setindex!(A, X[i], j)
    end
end

"""
    sfopt(;kwargs...)

Defining Hyperparmeters with default for the mle estimatation.

See also: [`setindx!`](@ref)
"""
function sfopt(;kwargs...)
    default_opt = OrderedDict{Symbol, Any}(
        :warmstart_solver=>NelderMead, 
        :warmstart_maxIT=>100,
        :main_solver=>Newton,
        :main_maxIT=>2000,
        :tolerance=>1e-8,
        :verbose=>true,
        :table_format=>:text
    )
    # values(kwags)` get a named tuple
    # setindex!(A, X, inds...)
    length(kwargs) != 0 && setindex!(default_opt, values(values(kwargs)), keys(kwargs))
    default_opt[:warmstart_solver] !== nothing && (default_opt[:warmstart_solver] = default_opt[:warmstart_solver]())
    default_opt[:main_solver] = default_opt[:main_solver]()
    return default_opt

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

julia> sfinit(log_???????=(-0.1, -0.1, -0.1, -0.1), log_???????=-0.1)
(log_??????? = (-0.1, -0.1, -0.1, -0.1), log_??????? = -0.1)
```
"""
sfinit(init::Vector) = init
sfinit(;kwargs...) = values(kwargs)


"""
    olsinfo(frontiers::AbstractMatrix{<:Real}, depvar::AbstractMatrix{<:Real})

Calculate coefficient, log likelihood, skewnewss of OLS
"""
function olsinfo(frontiers, depvar, nofx, nofobs, noffixed)
    ??0 = frontiers \ depvar
    resid = depvar - frontiers*??0
    sse = sum((resid).^2)  
    ssd = sqrt(sse/(nofobs-(nofx+noffixed))) # sample standard deviation; ???? = (1/(N-K))* ?? ??^2
    ll = sum(normlogpdf.(0, ssd, resid)) # ols log-likelihood
    sk = sum(resid.^3 / (ssd^3*nofobs)) # skewnewss of ols residuals

    return ??0, ll, sk
end


"""
    initial_condition(froniters, depvar, init::Vector, nofx, nofobs; kwargs...)
    initial_condition(froniters, depvar, init::NamedTuple, nofx, nofobs; kwargs...)
    initial_condition(froniters, depvar, init::Nothing, nofx, nofobs; kwargs...)

Set the initial condition for coefficient of explanatory variables simply devide
dependent variable `depvar` by independent variables `frontiers`.
Then set 0.1 to be the default initial condition for all the other parameters.

This function are used by the method `sfmodel_fit()` and don't export
"""

function initial_condition(frontiers, depvar, init::Vector, nofx, nofobs, noffixed; kargs...)
    _, llols, skols =  olsinfo(frontiers, depvar, nofx, nofobs, noffixed)

    return init, llols, skols
end

function initial_condition(frontiers, depvar, init::NamedTuple, nofx, nofobs, noffixed; kwargs...)
    ??0, llols, skols =  olsinfo(frontiers, depvar, nofx, nofobs, noffixed)

    startpt = ones(AbstractFloat, kwargs[:nofparam]) .* 0.1
    startpt[1:nofx] = ??0
    template = kwargs[:paramnames]
    push!(template, :end)  # to prevent can't getnext
    for i in keys(init)
        beg = findfirst(x->x==i, template)
        en = findnext(x->x!=Symbol(""), template, beg+1) - 1
        startpt[beg:en] .= Base.getproperty(init, i)
    end

    return startpt, llols, skols
end

function initial_condition(frontiers, depvar, init::Nothing, nofx, nofobs, noffixed; kwargs...)
    ??0, llols, skols =  olsinfo(frontiers, depvar, nofx, nofobs, noffixed)

    startpt = ones(AbstractFloat, kwargs[:nofparam]) .* 0.1
    startpt[1:nofx] = ??0

    return startpt, llols, skols
end


"""
    sfmodel_fit(;spec, options init)

# Arguments
- `spec::Tuple{<:AbstractSFmodel, <:AbstractData}`
- `options::OrderedDict{Symbol, Any}`
- `init::Union{Vector{<:Real}, NamedTuple{Symbol, <:Tuple{Vararg{Any}}}}`


There are three main process:
1. set the hyperparmeters with `options` for mle estimatation, set the initial initial condition with `init`
2. mle estimation
3. some extension(e.g. inefficiency index, marginal effect...)

See also: [`sfopt`](@ref), [`sfinit`](@ref), [`startpt`](@ref)
"""
function sfmodel_fit(;spec,
                      options=nothing,
                      init=nothing
                    )
    model, data = spec
    # set the mle hyperparmeters
    options === nothing && (options = sfopt())

    # set the initial condition
    noffixed = !isa(data, PanelData) ? 0 : numberofi(data)
    startpt, llols, skols = initial_condition(
        frontier(data), dependentvar(data), init, get_paramlength(model)[1], numberofobs(data), noffixed;
        nofparam=numberofparam(model), paramnames=get_paramname(model)[:, 1]
    )

    # mle estimation
    if options[:verbose]
        modelinfo(model)  # print some model information
        if isa(options[:warmstart_solver], Tuple{})
            printstyled("\n * optimization \n\n", color=:yellow)
        else
            printstyled("\n * optimization(with warmstart) \n\n", color=:yellow)
        end
    end

    _Hessian, ??, warmup_opt, main_opt = mle(model, data, options, startpt)
    
    # output the mle optimiazation results
    diagonal = post_estimation(get_modeldata(model), data, _Hessian, ??)  # diagonostic

    # efficiency and inefficiency index
    jlms, bc = jlmsbc(??, model, data)
    loglikelihood = round(-1*opt.minimum(main_opt); digits=5)

    if options[:verbose]
        output_estimation(numberofobs(data), warmup_opt, main_opt)
        stderr, cilow, ciup = output_table(get_paramname(model), ??, diagonal, numberofobs(data), options[:table_format])
        
        
        println("Table format: $(options[:table_format]). Use `sfopt(...)` to choose between `:text`, `:html`, and `:latex`.")

        printstyled("\n\n*********************************\n "; color=:cyan)
        printstyled("    Additional Information:     \n"; color=:cyan); 
        printstyled("*********************************\n\n"; color=:cyan)
        println(" - OLS (frontier-only) log-likelihood: $llols")
        println(" - Skewness of OLS residuals: $skols")
        println(" - The sample mean of the JLMS inefficiency index: $(round(mean(jlms), digits=5))")
        println(" - The sample mean of the BC efficiency index: $(round(mean(bc), digits=5))")
        println(" - Use `name.list` to see saved results (keys and values) where `name` is the return specified in `name = sfmodel_fit(...)`")
        println("     - `name.loglikelihood`: the log-likelihood value of the model;")
        println("     - `name.jlms`: Jondrow et al. (1982) inefficiency index;")
        println("     - `name.bc`: Battese and Coelli (1988) efficiency index;")
        println(" - Use `keys(name)` to see available keys.\n")
        printstyled("*********************************\n\n"; color=:cyan)
    end  # end if options[:verbose]

    res = (
        ??=??, model=model, data=data, options=options, jlms=jlms, bc=bc, 
        loglikelihood=loglikelihood, 
    )

    return res
end