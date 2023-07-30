println("##############################################")

# cd("E:\\ajen\\myJulia\\devtest")
# push!(LOAD_PATH, "E:\\ajen\\myJulia\\SFrontiers")


using SFrontiers

using CSV, DataFrames
using Revise #, JLD2
# using BenchmarkTools
using Random
using Test
using StatFiles
using Distributions

Random.seed!(0);     # Fix random seed generator for reproducibility. Need Random package.

using SFrontiers


# * ------ MoM test ------------------

df2 = DataFrame(CSV.File("sampledata.csv")) 

#  #** append a column of 1 to be used as a constant
df2[!, :_cons] .=1;

res = sfmodel_MoMTest(sftype(prod), sfdist(half),
                     @depvar(yvar), @frontier( Lland, PIland, Llabor, Lbull, Lcost, _cons),
                     data = df2,
                     ω = (0.5, 1, 2)                  
                     );



df = DataFrame(CSV.File("demodata.csv"));
df[!, :_cons] .=1.0;

aa = sfmodel_MoMTest(sftype(prod), sfdist(half),
                     @depvar(y), @frontier( _cons, x1, x2, x3),
                     α=0.05, data = df,
                     ω = (0.5, 1, 2),
                     verbose=false,
                    );



# * -------- Example 11.5, Kumbhakar 1990 Model (Stata: ch10.do, Model 5) ----- 

# df = CSV.read("paneldata1.csv", DataFrame; header=1, delim=",")
df = DataFrame(CSV.File("paneldata1.csv")) 
df[!, :_cons] .=1.0;
df[!, :yearT2] = df.yearT.^2;

aa = sfmodel_spec( sfpanel(Kumbhakar1990), sftype(production), sfdist(trun),
              @timevar(yr), @idvar(code),
              @depvar(lny), 
              @frontier(lnk, lnl, yr, _cons), 
              @μ(iniStat, _cons),
              @gamma(yearT, yearT2),
              @σᵤ²(_cons),
              @σᵥ²(_cons))


sfmodel_init( μ(0,0), gamma(0,0), sigma_u_2(0), sigma_v_2(0)  )         

sfmodel_opt( warmstart_solver(NelderMead()),    
         warmstart_maxIT(200),
         main_solver(Newton()),      
         main_maxIT(2000), 
         tolerance(1e-4)
         )

res = ()

res = sfmodel_fit(useData(df));


std, boot1 = sfmodel_boot_marginal(result = res, data=df, 
    R=20, seed=123, every=10, iter=200, getBootData=true, level=0.05);

sfmodel_CI(bootdata = boot1, observed=res.marginal_mean, 
     level = 0.9)


# * -------- Example 11, Kumbhakar 1990 Model (Stata: ch10.do, Model 5) ----- 


# df = CSV.read("paneldata1.csv", DataFrame; header=1, delim=",")
df = DataFrame(CSV.File("paneldata1.csv")) 
df[!, :_cons] .=1.0;
df[!, :yr2] = df.yr.^2;


aa = sfmodel_spec( sfpanel(Kumbhakar1990), sftype(production), sfdist(trun),
                  @timevar(yr), @idvar(code),
                  @depvar(lny), 
                  @frontier(lnk, lnl, yr, _cons), 
                  @μ(iniStat, _cons),
                  @gamma(yr, yr2),
                  @σᵤ²(_cons),
                  @σᵥ²(_cons))


sfmodel_init( μ(0,0), gamma(0,0), sigma_u_2(0), sigma_v_2(0)  )         

sfmodel_opt( warmstart_solver(NelderMead()),    
         warmstart_maxIT(),
         main_solver(Newton()),      
         main_maxIT(2000), 
         tolerance(1e-4)
         )

res = ()

res = sfmodel_fit(useData(df));

@test res.coeff[1:2] ≈ [0.85481, 0.11195] atol=1e-5 
@test res.jlms[1:2] ≈ [2.91400, 2.85425] atol=1e-5 
@test res.bc[1:2] ≈ [0.05428, 0.05762] atol=1e-5 
@test res.marginal.marg_iniStat[1:2] ≈ [0.03605, 0.03531] atol=1e-5

res_kumb90 = res;


test1  = sfmodel_predict(@eq(frontier), df);
test_pre = test1[1:2];
@test test_pre ≈ [7.24726, 7.22944] atol=1e-5 


bres3 = sfmodel_boot_marginal(result = res_kumb90, data=df, 
    R=200, seed=123, every=10, iter=200)



#* ---- generated data --------------

#=
n=20; T=1; σₓₒ²=1; σzₒ²=1; σᵥₒ²=1; μₒ=0; σᵤₒ²=3;
γ1 = 0.5; 

X =  randn(n*T, 1)*sqrt(σₓₒ²) 
Z =  randn(n*T, 1)*sqrt(σzₒ²)
v =  randn(n*T, 1)*sqrt(σᵥₒ²)
_con = ones(n*T, 1)

xvar = hcat(X, _con)
uvar = hcat(Z, _con)

distTN = TruncatedNormal(μₒ, sqrt(σᵤₒ²), 0, Inf)
     u = exp.(0.5*Z*0.5).*rand(distTN, (n*T, 1)) # matrix
     y = 0.5 .+ X*0.5 .+ v .- u



=#


#* --- Example 10, Panel True Random Effect Model, half normal ------------------


# df = CSV.read("TRE_half.csv", DataFrame; header=1, delim=",")
df = DataFrame(CSV.File("TRE_half.csv")) 
df[!, :_cons] .= 1.0


tmp1 = sfmodel_spec( sfpanel(TRE), sftype(prod), sfdist(half),
              @timevar(time), @idvar(id),
              @depvar(y), 
              @frontier(x1, x2, _cons), 
              @σₐ²(_cons),
              @σᵤ²(_cons),
              @σᵥ²(_cons),
              message = true)

sfmodel_init(message=true);              

sfmodel_opt( warmstart_solver(NelderMead()),   
         warmstart_maxIT(200),
         main_solver(Newton()),      
         main_maxIT(2000), 
         tolerance(1e-8),
         table_format(text),
         message = true
         )

         
res = sfmodel_fit(useData(df));


res_treh = res;

@test res.coeff[1:2] ≈ [0.53269, 0.67494] atol=1e-5 
@test res.bc[1:2] ≈ [0.62551, 0.72209] atol=1e-5



#* -- Example 9, Panel FE, CSW 2014 (CSN) (see Schmidt 2014 public folder ) -

# df = CSV.read("utility.csv", DataFrame; header=1, delim=",")
df = DataFrame(CSV.File("utility.csv")) 
# df = CSV.read("utility_minus1.csv", DataFrame; header=1, delim=",")

df[!, :_cons] .= 1.0

sfmodel_spec( sfpanel(TFE_CSW2014), sftype(prod), sfdist(half),
              @timevar(year), @idvar(firm),
              @depvar(lny), 
              @frontier(llabor, lfuel, lk, llabor2, lfuel2, lk2, 
                        labfue2, labk2, fuek2, trend), 
              @σᵤ²(_cons),
              @σᵥ²(_cons))

sfmodel_init()              

sfmodel_opt( warmstart_solver(NelderMead()),   
         warmstart_maxIT(200),
         main_solver(Newton()),      
         main_maxIT(2000), 
         tolerance(1e-8)
         )


res = sfmodel_fit(useData(df));
res_csw = res;

@test res.coeff[1:2] ≈ [0.03180, 0.66562] atol=1e-5 
@test res.jlms[1:2] ≈ [0.13239, 0.10492] atol=1e-5



# * -------- Example 8.1 (alternative method), Time Decay Model (BC1992) (Stata: ch10.do, Model 6) ----- 


# df = CSV.read("paneldata1.csv", DataFrame; header=1, delim=",")
df = DataFrame(CSV.File("paneldata1.csv")) 
df[!, :_cons] .=1.0;

tv =  df[!, [:yr]];
iv =  df[!, [:code]];
yv =  df[!, [:lny]];
fv =  df[!, [:lnk, :lnl, :yr, :_cons]];
mv =  df[!, [:iniStat, :_cons]];
gv =  df[!, [:yearT]];
uv =  df[!, [:_cons]];
vv =  df[!, [:_cons]];



tv = convert(Array{Float64}, Matrix(df[!, [:yr]]));
iv = convert(Array{Float64}, Matrix(df[!, [:code]]));
yv = convert(Array{Float64}, Matrix(df[!, [:lny]]));
fv = convert(Array{Float64}, Matrix(df[!, [:lnk, :lnl, :yr, :_cons]]));
mv = convert(Array{Float64}, Matrix(df[!, [:iniStat, :_cons]]));
gv = convert(Array{Float64}, Matrix(df[!, [:yearT]]));
uv = convert(Array{Float64}, Matrix(df[!, [:_cons]]));
vv = convert(Array{Float64}, Matrix(df[!, [:_cons]]));


ha1 = sfmodel_spec( sfpanel(TimeDecay), sftype(prod), sfdist(trun),
              timevar(tv), idvar(iv),
              depvar(yv), 
              frontier(fv), 
              μ(mv),
              gamma(gv),
              σᵤ²(uv),
              σᵥ²(vv))


para0 = ones(9)*(-0.1);               


sfmodel_init( all_init(para0)  )         

#* This model is peculiar. If change sf_init to BigFloat, converge to 0.8532...
#* If change the main_solver to NewtonTrust, same thing. If change the main_solver 
#* to BFGS(), converge to very strange coefficients.
#* Need to check with Stata.

sfmodel_opt( warmstart_solver(NelderMead()),    #* BFGS does not work
         warmstart_maxIT(200),
         main_solver(Newton()),      #* BFGS does not work
         main_maxIT(2000), 
         tolerance(1e-8),
         verbose(true),
         banner(true),
         marginal(true))

res = ()

res = sfmodel_fit();


@test res.coeff[1:2] ≈ [0.58062, 0.01895] atol=1e-5 
@test res.jlms[1:2] ≈ [1.41460, 1.41279] atol=1e-5 
@test res.bc[1:2] ≈ [0.24309, 0.24353] atol=1e-5 

res_bc1992 = res;


test1  = sfmodel_predict(@eq(frontier));
test_pre = test1[1:2];
@test test_pre ≈ [5.78038, 5.81550] atol=1e-5 


test2  = sfmodel_predict(@eq(gamma));
@test test2[1:2] ≈ [1.03524, 1.03391] atol=1e-4 

test2a = sfmodel_predict(@eq(log_gamma));

test3  = sfmodel_predict(@eq(sigma_u_2));
@test test3[1:1] ≈ [0.56645] atol=1e-5 
test3a  = sfmodel_predict(@eq(σᵤ²));

test4 = sfmodel_predict(@eq(log_sigma_u_2));
test4a = sfmodel_predict(@eq(log_σᵤ²));

test5  = sfmodel_predict(@eq(sigma_v_2));
@test test5[1:1] ≈ [0.01524] atol=1e-5 
test5a  = sfmodel_predict(@eq(σᵥ²))

test6 = sfmodel_predict(@eq(log_sigma_v_2))
test6a = sfmodel_predict(@eq(log_σᵥ²))

test7  = sfmodel_predict(@eq(μ))
@test test7[1:1] ≈ [1.84177] atol=1e-5 
test7a = sfmodel_predict(@eq(mu))

bres2 = sfmodel_boot_marginal(result = res_bc1992, R=10, seed=123, every=1)



# * -------- Example 8, Time Decay Model (BC1992) (Stata: ch10.do, Model 6) ----- 


# df = CSV.read("paneldata1.csv", DataFrame; header=1, delim=",")
df = DataFrame(CSV.File("paneldata1.csv")) 
df[!, :_cons] .=1.0;


aa = sfmodel_spec( sfpanel(TimeDecay), sftype(production), sfdist(trun),
              @timevar(yr), @idvar(code),
              @depvar(lny), 
              @frontier(lnk, lnl, yr, _cons), 
              # @frontier([:lnk, :lnl, :yr, :_cons]),
              @μ(iniStat, _cons),
              @gamma(yearT),
              @σᵤ²(_cons),
              @σᵥ²(_cons))


para0 = ones(9)*(-0.1);               

sfmodel_init( all_init(para0)  )         

sfmodel_opt( warmstart_solver(NelderMead()),    #* BFGS does not work
         warmstart_maxIT(200),
         main_solver(Newton()),      #* BFGS does not work
         main_maxIT(2000), 
         tolerance(1e-8)
         # , verbose(false)
         # , nobanner(true)
         )

res = ()

res = sfmodel_fit(useData(df));


@test res.coeff[1:2] ≈ [0.58062, 0.01895] atol=1e-5 
@test res.jlms[1:2] ≈ [1.41460, 1.41279] atol=1e-5 
@test res.bc[1:2] ≈ [0.24309, 0.24353] atol=1e-5 
@test res.marginal.marg_iniStat[1:2] ≈ [-0.13404, -0.13387] atol=1e-5

res_bc1992 = res;


test1  = sfmodel_predict(@eq(frontier), df);
test_pre = test1[1:2];
@test test_pre ≈ [5.78038, 5.81550] atol=1e-5 

test2  = sfmodel_predict(@eq(gamma), df);
test_pre = test2[1:2];
@test test_pre ≈ [1.03524, 1.03391] atol=1e-4 
test2a = sfmodel_predict(@eq(log_gamma), df);

test3  = sfmodel_predict(@eq(sigma_u_2), df);
test_pre = test3[1:1];
@test test_pre ≈ [0.56645] atol=1e-5 
test3a  = sfmodel_predict(@eq(σᵤ²), df);

test4 = sfmodel_predict(@eq(log_sigma_u_2), df);
test4a = sfmodel_predict(@eq(log_σᵤ²), df);

test5  = sfmodel_predict(@eq(sigma_v_2), df);
test_pre = test5[1:1];
@test test_pre ≈ [0.01524] atol=1e-5 
test5a  = sfmodel_predict(@eq(σᵥ²), df)

test6 = sfmodel_predict(@eq(log_sigma_v_2), df)
test6a = sfmodel_predict(@eq(log_σᵥ²), df)

test7  = sfmodel_predict(@eq(μ), df)
test_pre = test7[1:1];
@test test_pre ≈ [1.84177] atol=1e-5 
test7a = sfmodel_predict(@eq(mu), df) 


#* ------ Example 7, panel FE model of Wang and Ho, truncated normal ----------


# df = CSV.read("WH2010T.csv", DataFrame; header=1, delim=",")
df = DataFrame(CSV.File("WH2010T.csv")) 

df[!, :_cons] .= 1.0


sfmodel_spec( sfpanel(TFE_WH2010), sftype(prod), sfdist(trun),
              @timevar(time), @idvar(id),
              @depvar(yit), 
              @frontier(xit), 
              @μ(_cons),
              @hscale(zit),
              @σᵤ²(_cons),
              @σᵥ²(_cons))


sfmodel_init()

#= 
sfmodel_init( # frontier(0.5),
              μ(0.5),
              σᵤ²(-0.1),
              hscale(-0.5),
              σᵥ²(-0.1) )   
=#


sfmodel_opt( warmstart_solver(NelderMead()),  # BFGS work,
             warmstart_maxIT(200),
             main_solver(Newton()), # BFGS ok; may try warmstart_delta=0.2
             main_maxIT(2000), 
             tolerance(1e-8))

res = ()

res = sfmodel_fit(useData(df));
res_wh2010T = res;
bres = sfmodel_boot_marginal(result=res_wh2010T, data=df, R=10, seed=123)

@test res.coeff[1:2] ≈ [0.49727, 0.79545] atol=1e-5
@test res.jlms[1:2] ≈ [3.61393, 4.53487] atol=1e-5
@test res.bc[1:2] ≈ [0.02811, 0.01146] atol=1e-5
@test res.marginal.marg_zit[1:2] ≈ [1.28355, 1.61064] atol=1e-5
@test bres[1] ≈ 0.03087 atol=1e-5


#* ------ Example 6, panel FE model of Wang and Ho, Half normal ----------

@load "WangHo2010data.jld2"

df[!, :_cons] .= 1.0

sfmodel_spec( sfpanel(TFE_WH2010), sftype(prod), sfdist(half),
              @timevar(year), @idvar(cnum),
              @depvar(y), 
              @frontier(x1, x2, x3), 
              @hscale(z1, z2),
              @σᵤ²(_cons),
              @σᵥ²(_cons))

sfmodel_init( # frontier(bb),
              # μ(-0.1, -0.1, -0.1, -0.1),
              # σᵤ²(0.1, 0.1, 0.1, 0.1),
              # λ(0.1, 0.1),
              σᵥ²(0.1) )

sfmodel_opt( warmstart_solver(NewtonTrustRegion()), 
             warmstart_maxIT(100),
             main_solver(NewtonTrustRegion()), # BFGS ok
             main_maxIT(2000), 
             tolerance(1e-8))

res = ()

res = sfmodel_fit(useData(df));

@test res.coeff[1:2] ≈ [0.65495, 0.15002] atol=1e-5 #src
@test res.marginal.marg_z1[1:2] ≈ [0.00261,0.00252] atol=1e-5

res_wh2010 = res;


test1  = sfmodel_predict(@eq(hscale), df);
test1a = sfmodel_predict(@eq(log_hscale), df);

test2  = sfmodel_predict(@eq(frontier), df);
test_pre = test2[1:2];
@test test_pre ≈ [0.14480, -0.18794] atol=1e-5 


#* ------- Example 5, scaling property model (truncated) -----------


df = DataFrame(load("dairy.dta"))
df[!, :_cons] .= 1


sfmodel_spec( sftype(prod), sfdist(trun_scale), 
              @depvar(ly), 
              @frontier(llabor, lfeed, lcattle, lland, _cons), 
              @hscale(comp),
              @μ(_cons),
              @σᵤ²(_cons),
              @σᵥ²(_cons))


bb= ones(5)*0.1
sfmodel_init( frontier(bb))              

sfmodel_init( # frontier(bb),
              hscale(-0.1),
              σᵤ²(-0.1),
              μ(0.1),
              σᵥ²(-0.1) )  

sfmodel_opt( warmstart_solver(Newton()),   
             warmstart_maxIT(1000),
             main_solver(Newton()), # may try warmstart_delta=0.2
             main_maxIT(2000), 
             tolerance(1e-8))


#! Note, by commenting out sfmodel_opt(), it uses settings from the previous
#! run.
#! In the current case, `BFGS` does not work in the warmstart run

res = ()

res = sfmodel_fit(useData(df));

@test res.coeff[1:2] ≈ [0.09755, 0.14947] atol=1e-5 
@test res.jlms[1:2] ≈ [0.05200, 0.11586] atol=1e-5
@test res.bc[1:2] ≈ [0.95017, 0.89273] atol=1e-5
@test res.marginal.marg_comp[1:2] ≈ [-0.00109, -0.00106] atol=1e-5

res_scaling = res;

std_ci, bsdata = sfmodel_boot_marginal(result = res, data=df, 
    R=100, seed=123, every=10, iter=200, getBootData=true, level=0.05);


sfmodel_CI(bootdata = bsdata, observed=res.marginal_mean, 
     level = 0.9)


#* -----  Example 4, exponential dist (Stata: chapter3.do, Model 10)  ----
 
 
df = DataFrame(load("dairy.dta"))
# df = CSV.read("dairy2.csv", DataFrame; header=1, delim=",")

df[!, :_cons] .= 1


sfmodel_spec( sftype(prod), sfdist(expo),
              @depvar(ly), 
              @frontier(llabor, lfeed, lcattle, lland,  _cons), 
              @σᵤ²(comp, _cons),
              @σᵥ²(_cons))


sfmodel_init( # frontier(bb),
              σᵤ²(-0.1, -0.1),
              σᵥ²(-0.1) )  


sfmodel_opt( warmstart_solver(),  # empty means don't do warmstart; #* BFGS does not work
             warmstart_maxIT(1000),
             main_solver(Newton()), # may try warmstart_delta=0.2
             main_maxIT(5000), 
             tolerance(1e-8))

             
res = ()

res = sfmodel_fit(useData(df));

@test res.coeff[1:2] ≈ [0.09655, 0.14869] atol=1e-5 
@test res.jlms[1:2] ≈ [0.04817, 0.10286] atol=1e-5 
@test res.bc[1:2] ≈ [0.95375, 0.90431] atol=1e-5 
@test res.marginal.marg_comp[1:2] ≈ [-0.00028, -0.00028] atol=1e-5

res_expo = res;


# * ------ Example 3 --- truncated normal (Wang 2002) ----------------- *#


   @load "exampledata.jld2"  # a little bit faster than reading from the CSV file
     
    #  # * Read in Dataset
    #    df = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")

    #  #** append a column of 1 to be used as a constant
       df[!, :_cons] .=1;



test = sfmodel_spec(sftype(prod), sfdist(trun),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @μ(age, school, yr, _cons),
             @σᵤ²(age, school, yr, _cons),
             @σᵥ²(_cons))


b1a = ones(4)*(0.1)

sfmodel_init(  # frontier(bb),
              μ(b1a),
              σᵤ²(-0.1, -0.1, -0.1, -0.1),
              σᵥ²(-0.1) ) 
            
              
# b0 = [.1011541,  .155526,  .756702,  .0348882, 7.736939,  -.0298775,  -2.950646, -5.308]

# sfmodel_init(all_init(b0))


# sfmodel_opt(warmstart_solver(NelderMead()),   #* BFGS does not work
#            warmstart_maxIT(100),
#            main_solver(NewtonTrustRegion()), #* BFGS not work; may try warmstart_delta=0.2
#            main_maxIT(2000), 
#            tolerance(1e-8))

sfmodel_opt(warmstart_solver(NelderMead()),   #* BFGS does not work
            warmstart_maxIT(400),
            main_solver(Newton()), #* BFGS not work; may try warmstart_delta=0.2
            main_maxIT(2000), 
            tolerance(1e-8))

res = ()

res = sfmodel_fit(useData(df));

testcoe = res.coeff[1:2];
@test testcoe ≈ [0.25821, 0.17173] atol=1e-5 
# @test res.marginal.marg_age[1:2] ≈ [-0.00522, -0.00636] atol=1e-5

res_wang2002 = res;

std_ci = sfmodel_boot_marginal(result=res, data=df, R=100, seed=1232, iter=100)


std_ci, bsdata = sfmodel_boot_marginal(result=res, data=df, 
                 R=250, seed=123, getBootData=true);


sfmodel_CI(bootdata=bsdata, observed=res.marginal_mean, level=0.10)

# manually input observed values
sfmodel_CI(bootdata=bsdata, observed=(-0.00264, -0.01197, -0.0265), level=0.10)



# * ------ Example 2 --- truncated normal (Wang 2002, no age) ----------------- *#


@load "exampledata.jld2"  # a little bit faster than reading from the CSV file
     
#  # * Read in Dataset
#    df = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")

#  #** append a column of 1 to be used as a constant
   df[!, :_cons] .=1;



sfmodel_spec( sftype(prod), sfdist(trun),
              @depvar(yvar), 
              @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
              @μ( school, yr, _cons),
              @σᵤ²(  school, yr, _cons),
              @σᵥ²(_cons))


b1b = ones(3)*(0.1)

sfmodel_init(  # frontier(bb),
             μ(b1b),
             σᵤ²( -0.1, -0.1, -0.1),
             σᵥ²(-0.1) ) 
        
          
# b0 = [.1011541,  .155526,  .756702,  .0348882, 7.736939,  -.0298775,  -2.950646, -5.308]

# sfmodel_init(all_init(b0))


sfmodel_opt( warmstart_solver(NelderMead()),   #* BFGS does not work
             warmstart_maxIT(100),
             main_solver(NewtonTrustRegion()), #* BFGS not work; may try warmstart_delta=0.2
             main_maxIT(2000), 
             tolerance(1e-8))

res = ()

res = sfmodel_fit(useData(df));

testcoe = res.coeff[1:2];
@test testcoe ≈ [0.30533, 0.22575] atol=1e-5 

res_wang2002noage = res;



# * ------ Example 1.5 --- truncated normal (BC1995) ----------------- *#


@load "exampledata.jld2"  # a little bit faster than reading from the CSV file
     
#  # * Read in Dataset
#    df = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")

#  #** append a column of 1 to be used as a constant
   df[!, :_cons] .=1;



sfmodel_spec( sftype(prod), sfdist(trun),
              @depvar(yvar), 
              @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
              @μ(age, school, yr, _cons),
              @σᵤ²(_cons),
              @σᵥ²(_cons))


b1a = ones(4)*(0.1)

sfmodel_init(  # frontier(bb),
             μ(b1a),
             # σᵤ²(-0.1, -0.1, -0.1, -0.1),
             σᵥ²(-0.1) ) 
        

sfmodel_opt( warmstart_solver(NelderMead()),   # BFGS ok
             warmstart_maxIT(100),
             main_solver(Newton()), # BFGS ok
             main_maxIT(2000), 
             tolerance(1e-8))

res = ()

res = sfmodel_fit(useData(df));

@test res.coeff[1:2] ≈ [0.30298, 0.24794] atol=1e-5 
#@test res.marginal.marg_school[1:2] ≈ [0.00323, 0.00281] atol=1e-5

res_bc1995 = res;



# * ------ Example 1.3 --- truncated normal, no age (BC1995) ----------------- *#


@load "exampledata.jld2"  # a little bit faster than reading from the CSV file
     
#  # * Read in Dataset
#    df = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")

#  #** append a column of 1 to be used as a constant
   df[!, :_cons] .=1;



sfmodel_spec( sftype(prod), sfdist(trun),
              @depvar(yvar), 
              @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
              @μ( school, yr, _cons),
              @σᵤ²(_cons),
              @σᵥ²(_cons))


b1a = ones(3)*(0.1)

sfmodel_init(  # frontier(bb),
             μ(b1a),
             # σᵤ²(-0.1, -0.1, -0.1, -0.1),
             σᵥ²(-0.1) ) 
        

sfmodel_opt( warmstart_solver(NelderMead()),   # BFGS OK
             warmstart_maxIT(100),
             main_solver(Newton()), # BFGS ok
             main_maxIT(2000), 
             tolerance(1e-8))

res = ()

res = sfmodel_fit(useData(df));

testcoe = res.coeff[1:2];
# @test testcoe ≈ [0.30298, 0.24794] atol=1e-5 #src

res_bc1995noage = res;



# * ----- Example 1.2 ---- truncated normal, vanilla --------


# @load "exampledata.jld2"  # a little bit faster than reading from the CSV file
     
#  # * Read in Dataset
#   df2 = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")
df2 = DataFrame(CSV.File("sampledata.csv")) 

#  #** append a column of 1 to be used as a constant
   df2[!, :_cons] .=1;



sfmodel_spec( sftype(prod), sfdist(trun),
              @depvar(yvar), 
              @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
              @μ( _cons),
              @σᵤ²(_cons),
              @σᵥ²(_cons))


b1a = ones(1)*(0.1)

sfmodel_init(  # frontier(bb),
             μ(b1a),
             # σᵤ²(-0.1, -0.1, -0.1, -0.1),
             σᵥ²(-0.1) ) 
        

sfmodel_opt( warmstart_solver(NelderMead()),   
             warmstart_maxIT(100),
             main_solver(Newton()), 
             main_maxIT(2000), 
             tolerance(1e-8))

res = ()

res = sfmodel_fit(useData(df2));

@test res.coeff[1:2] ≈ [0.29315, 0.23998] atol=1e-5 #src

res_trun = res;



# * ------ Example 1 --- half normal ----------------- *#


# @load "exampledata.jld2"  # a little bit faster than reading from the CSV file
     
#  # * Read in Dataset
#   df = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")
df = DataFrame(CSV.File("sampledata.csv")) 

#  #** append a column of 1 to be used as a constant
   df[!, :_cons] .=1;


sfmodel_spec(sftype(prod), sfdist(half),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @σᵤ²(age, school, yr, _cons),
             @σᵥ²(_cons))


sfmodel_init(# frontier(bb),
             # μ(-0.1, -0.1, -0.1, -0.1),
             # σᵤ²(0.1, 0.1, 0.1, 0.1),
             σᵥ²(0.1) ) 

sfmodel_opt( warmstart_solver(NelderMead()),   #* BFGS does not work
             warmstart_maxIT(100),
             main_solver(Newton()), #* BFGS does not work; may try warmstart_delta=0.2
             main_maxIT(2000), 
             tolerance(1e-8))

res = ()

res = sfmodel_fit(useData(df));


@test res.coeff[1:2] ≈ [0.29488, 0.23191] atol=1e-5 
@test res.marginal.marg_age[1:2] ≈ [-0.00147, -0.00135] atol=1e-5

res_half = res;

# marg_std, marg_data = sfmodel_boot_marginal(result=res_half, data=df, R=20, getBootData=true, seed=123);

marg_std, marg_data = sfmodel_boot_marginal(result=res_half, data=df, R=20, getBootData=true, seed=123);


@test marg_std[1:3, 1] ≈ [0.00263, 0.01180, 0.01377] atol=1e-5
@test marg_data[1, 1:3] ≈ [-0.00015, 0.00451, -0.03662] atol=1e-5


println("End of Tests\n")

#! ##### end here 底下不重要，測試用###############
aaaaaa

#* ---- Greene TFE, test ------------

#=

df = CSV.read("TRE_half.csv", DataFrame; header=1, delim=",")
df[!, :_cons] .= 1.0

uid = unique(df.id);
df = transform(df, @. :id => ByRow(isequal(uid)) .=> Symbol(:dummy, uid))
describe(df)

xvar = Matrix(df[:, [:x1, :x2]])
alldummy = Matrix(df[:, 8:106]) # skip the first (or anyone of the) dummy to avoid multicollinearity
xMat = hcat(xvar, alldummy) # combine all of the frontier var
yMat = Matrix(df[:, [:y]])
cMat = Matrix(df[:, [:_cons]])


sfmodel_spec(sftype(prod), sfdist(half),  
             depvar(yMat), 
             frontier(xMat), 
             σᵤ²(cMat),
             σᵥ²(cMat))


sfmodel_init(# frontier(bb),
             # μ(-0.1, -0.1, -0.1, -0.1),
             # σᵤ²(0.1, 0.1, 0.1, 0.1),
             σᵥ²(0.1) ) 

sfmodel_opt( warmstart_solver(NelderMead()),   #* BFGS does not work
             warmstart_maxIT(100),
             main_solver(Newton()), #* BFGS does not work; may try warmstart_delta=0.2
             main_maxIT(2000), 
             tolerance(1e-8))

res = ()

res = sfmodel_fit();

res = sfmodel_fit(useData(df));

=#