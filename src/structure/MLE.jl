"""
    Base.getindex(A::OrderedDict{T, <:Any}, ind::Union{Vector{T}, NTuple{N, T}}) where{N, T}

Ordered dictionary utilities to get multiple indexes in cleaner way

# Examples
```juliadoctest
julia> example = OrderedDict{Symbol, Int}(:a=>3, :b=>2);

julia> a, b = getindex(example, (:a, :b)); @show a, b
(a, b) = (3, 2)

See also: [`setindex!`](@ref), [`sfopt`](@ref)
```
"""
Base.getindex(A::OrderedDict, ind::Union{Tuple, Vector}) = [Base.getindex(A, i) for i in ind]


"""
    warmup(_Hessian, _init, warmstart_solver, tolerance, warmstart_maxIT, verbose)

Maximum likelihood estimation(warmup)

# Arguments
- `_Hessian::NLSolversBase.TwiceDifferentiable{T} where T`: required by the optimiazation
- `_init::Vector{<:Real}`: initial condition(starting point)
- `warmstart_solver::opt.AbstractOptimizer`: warmstart optimizer
- `tolerance::AbstractFloat`: criterion for convergence
- `warmstart_maxIT::Int`: maximum iterations
- `verbose::Bool`
"""
function warmup(_Hessian,
                _init,
                warmstart_solver,
                tolerance,
                warmstart_maxIT,
                verbose)
    warmup_opt = opt.optimize(
            _Hessian,
            _init,
            warmstart_solver,
            opt.Options(
                g_tol=tolerance,
                iterations=warmstart_maxIT,
                store_trace=verbose,
                show_trace=verbose
            )
    )
    init = opt.minimizer(warmup_opt)
    return warmup_opt, init
end



"""
    mle(model, data, optinos, _coevec)    

Maximum likelihood estimation

# Arguments
- `model::SFmodel`
- `data::AbstractData`: general data
- `options::OrderedDict{Symbol, <:Any}`: Hyperparmeters for the mle estimatatio
- `init::Vector{<:Real}`: initial condition
"""
function mle(model, data, options, init)
    warmstart = options[:warmstart_solver] === nothing ? false : true  # check whether warmup
    _Hessian = opt.TwiceDifferentiable(
        ξ -> LLT(ξ, model, data),
        ones(length(init));
        autodiff=:forward
    )
    warmup_opt = nothing

    # warmup optimiazation
    if warmstart
        warmup_opt = opt.optimize(
            _Hessian,
            init,
            options[:warmstart_solver],
            opt.Options(
                g_tol=options[:tolerance],
                iterations=options[:warmstart_maxIT],
                # store_trace=options[:verbose],
                # show_trace=options[:verbose]
            )
        )
        init = opt.minimizer(warmup_opt)
    end

    # main optimiazation
    main_opt = opt.optimize(
        _Hessian,
        init,
        options[:main_solver],
        opt.Options(
            g_tol=options[:tolerance],
            iterations=options[:main_maxIT],
            # store_trace=options[:verbose],
            # show_trace=options[:verbose]
        )
    )
    _coevec = opt.minimizer(main_opt)
    
    return _Hessian, _coevec, warmup_opt, main_opt
end


"""
    post_estimation(model_data, data, _Hessian, _coevec)    

The purpose of the post estimation procesis to check the more rigorous converge criterion

# Arguments
- `model_data::AbstractData`: required specifically based on the set up of the model
- `data::AbstractData`: general data
- `_Hessian::NLSolversBase.TwiceDifferentiable{T} where T`: required by the optimiazation
- `_coevec::Vector{<:Real}`: maximum likelihood estimator
"""
function post_estimation(_Hessinan, _coevec)
    numerical_hessian = hessian!(_Hessinan, _coevec)
    var_cov_matrix = try  # check if the matrix is invertible
        inv(numerical_hessian)
    catch
        throw("The Hessian matrix is not invertible, indicating the model does not converge properly. The estimation is abort.")
    end
    diagonal = diag(var_cov_matrix)
    if !all(diagonal .> 0)
        throw("Some of the diagonal elements of the var-cov matrix are non-positive, indicating problems in the convergence. The estimatIon is abort.")
    end
    return diagonal
end


"""
    output_estimation(nofobs::Int, warm_opt, main_opt)

Show some information after maximum likelihood estimation to better understand
converge situation.

# Arguments
- `nofobs::Int`: number of observations
- `warm_opt::Union{Tuple{}, Optim.MultivariateOptimizationResults{T} where T}`: results of the warmup optimiazation, not necessarily required
- `main_opt::Optim.MultivariateOptimizationResults{T} where T`: results of the main optimization
"""
function output_estimation(nofobs, warm_opt, main_opt)
    # get the total iteratons
    iter = warm_opt === nothing ? opt.iterations(main_opt) : (opt.iterations(main_opt) + opt.iterations(warm_opt))
    loglikelihood = round(-1*opt.minimum(main_opt); digits=5)

    printstyled("*********************************\n "; color=:cyan)
    printstyled("      Estimation Results:\n"; color=:cyan); 
    printstyled("*********************************\n\n"; color=:cyan)
    println(" Numberf Of Observations:    $nofobs")
    println(" Log-likelihood Value:       $loglikelihood")
    println(" Time Consuming:             $(opt.time_run(main_opt))")
    println("")
    println(" Converged:                  $(opt.converged(main_opt))")
    println(" Number Of Total Iterations: $iter")
    println(" Iteration Limit Reached:    $(opt.iteration_limit_reached(main_opt))")
    println("")

end


"""
    output_talble(paramnames, _coevec, diagonal, nofobs, format)

Show the estimation table.

# Arguments
- `paramnames::Matrix{Symbol}`: names of some variables
- `_coevec::Vector{<:Real}`: maximum likelihood estimator
- `diagonal::Vector{<:Real}`: diagonal of the variance covariance matrix(invers of the hessian)
- `nofobs::Int`: number of observations
- `format::Symbol`: backend of the `PrettyTables.pretty_table`
"""
function output_table(paramnames, _coevec, diagonal, nofobs, format)
    stderr = sqrt.(diagonal)
    t_stats = _coevec ./ stderr
    p_value = [pvalue(TDist(nofobs - length(_coevec)), i; tail=:both) for i in t_stats]
    tt = cquantile(Normal(0,1), 0.025)
    cilow = [round(_coevec[i]-tt*stderr[i], digits=5) for i = eachindex(_coevec)]
    ciup = [round(_coevec[i]+tt*stderr[i], digits=5) for i = eachindex(_coevec)]
    pretty_table(
        hcat(paramnames, _coevec, stderr, t_stats, p_value, cilow, ciup),
        header=["", "Var.", "Coef.", "Std. Err.", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
        formatters = ft_printf("%5.4f", 3:8),
        compact_printing = true,
        backend = Val(format)
    )
    return stderr, cilow, ciup
end