"""
    mle(model, data, optinos, init)    

Maximum likelihood estimation

# Arguments
- `model::AbstractSFmodel`
- `data::AbstractData`
- `options::NamedTuple` : Hyperparmeters for the mle estimatation
- `init::Vector{<:Real}`: initial condition

"""
function mle(model, data, options, init)
    warmstart   = options.warmstart_solver isa Nothing ? false : true
    warmup_iter = -1
    verbose     = options.verbose

    func = Optim.TwiceDifferentiable(
        ξ -> -LLT(ξ, model, data),
        init;
        autodiff = :forward
    )

    # warmup optimiazation
    if warmstart
        warmup_opt = Optim.optimize(
            func,
            init,
            options.warmstart_solver,
            Optim.Options(
                g_tol       = options.tolerance,
                iterations  = options.warmstart_maxIT,
                store_trace = verbose,
                show_trace  = false
            )
        )
        warmup_iter = Optim.iterations(warmup_opt)
        init        = Optim.minimizer(warmup_opt)
    end

    # main optimiazation
    main_opt = Optim.optimize(
        func,
        init,
        options.main_solver,
        Optim.Options(
            g_tol       = options.tolerance,
            iterations  = options.main_maxIT,
            store_trace = verbose,
            show_trace  = options.show_trace
        )
    )
    _coevec = Optim.minimizer(main_opt)
    
    return func, _coevec, warmup_iter, main_opt
end


"""
    post_estimation(func, _coevec)    

Check the variance covariance matrix

# Arguments
- `func::NLSolversBase.TwiceDifferentiable{T} where T`: Otpim object
- `_coevec::Vector{<:Real}`                           : maximum likelihood estimator

"""
function post_estimation(func, _coevec)
    numerical_hessian = hessian!(func, _coevec)
    var_cov_matrix = try  # check if the matrix is invertible
        inv(numerical_hessian)
    catch
        println("the maximizer: ")
        _coevec |> display
        throw("The Hessian matrix is not invertible, indicating the model does not converge properly. The estimation is abort.")
    end

    diagonal = diag(var_cov_matrix)
    if !all(diagonal .> 0)
        throw("Some of the diagonal elements of the var-cov matrix are non-positive, indicating problems in the convergence. The estimatIon is abort.")
    end

    return diagonal
end


"""
    output_estimation(nofobs, warm_opt, main_opt, gtol)

Show some information after maximum likelihood estimation to better understand
converge situation.

# Arguments
- `nofobs::Int`: number of observations

- `warm_opt::Union{Tuple{}, Optim.MultivariateOptimizationResults{T} where T}`: 
    results of the warmup optimiazation, not necessarily required

- `main_opt::Optim.MultivariateOptimizationResults{T} where T`: 
    results of the main optimization

"""
function output_estimation(nofobs, warmup_iter, main_opt, gtol)
    # get the total iteratons
    iter = warmup_iter == -1 ? Optim.iterations(main_opt) : (Optim.iterations(main_opt) + warmup_iter)
    loglikelihood = round(-1*Optim.minimum(main_opt); digits=5)

    converge = string(Optim.g_residual(main_opt) <= gtol)
    if Optim.iterations(main_opt) == 0 && converge == "true" 
        converge = "false (saddle point)"
    elseif Optim.iteration_limit_reached(main_opt)
        converge = "false (maximum number of iterations)"
    elseif converge == "false"
        f_abs_c = Optim.f_abschange(main_opt) == 0.
        x_abs_c = Optim.x_abschange(main_opt) == 0.
        f_rel_c = Optim.f_relchange(main_opt) == 0.
        x_rel_c = Optim.x_relchange(main_opt) == 0.

        if f_rel_c && f_abs_c && x_rel_c && x_abs_c
            converge = "false (|g| > gtol but f, x converge)"
        elseif f_rel_c && f_abs_c
            converge = "false (|g| > gtol but f converge)"
        elseif x_rel_c && x_abs_c
            converge = "false (|g| > gtol but x converge)"
        end
    end

    printstyled("*********************************\n "; color = :cyan)
    printstyled("       Estimation Results \n"; color = :cyan); 
    printstyled("*********************************\n\n"; color = :cyan)

    print(" Converge:                   ")
    if converge == "true"
        print("$(converge)\n")
    else
        printstyled("$(converge)\n"; color = :red)
    end

    println(" Log-likelihood Value:       $loglikelihood")
    println("")

    println(" Number Of Total Iterations: $iter")
    println(" Time Consuming:             $(round(Optim.time_run(main_opt), digits=5))")
    println(" Numberf Of Observations:    $nofobs")
    println("")
end


"""
    output_talble(paramnames, _coevec, diagonal, nofobs, format)

Show the estimation table.

# Arguments
- `paramnames::Matrix{Symbol}`: names of some variables
- `_coevec::Vector{<:Real}`   : maximum likelihood estimator
- `diagonal::Vector{<:Real}`  : diagonal of the variance covariance matrix(invers of the hessian)
- `nofobs::Int`               : number of observations
- `format::Symbol`            : backend of the `PrettyTables.pretty_table`

"""
function output_table(paramnames, _coevec, diagonal, nofobs, format)
    stderr  = sqrt.(diagonal)
    t_stats = _coevec ./ stderr
    p_value = Float64[pvalue(TDist(nofobs - length(_coevec)), i; tail = :both) for i in t_stats]
    tt      = cquantile(Normal(0,1), 0.025)
    cilow   = Float64[round(_coevec[i]-tt*stderr[i], digits=5) for i = eachindex(_coevec)]
    ciup    = Float64[round(_coevec[i]+tt*stderr[i], digits=5) for i = eachindex(_coevec)]

    pretty_table(
        hcat(paramnames, _coevec, stderr, t_stats, p_value, cilow, ciup),
        header           = ["", "Var.", "Coef.", "Std. Err.", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
        formatters       = ft_printf("%5.4f", 3:8),
        compact_printing = true,
        backend          = Val(format)
    )
end
