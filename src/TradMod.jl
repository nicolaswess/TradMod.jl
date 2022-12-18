module TradMod

using LinearAlgebra, DelimitedFiles, Kronecker, NLsolve, Distributed, TypedTables

export AGG_C
export AGG_S
export PREPARE_DATA
export CONVERT_ZERO_DEFICIT
export CNTRY_INDX
export SOLVE_SHOCK
export MLZ_WELFARE_CORRECTED
export MLZ_TRF_SYSTEM_MANY_ARGS_FIXED_COSTS
export MLZ_NEW_DATA
export runTradMod

include("AGG_C.jl")
include("AGG_S.jl")
include("PREPARE_DATA.jl")
include("CONVERT_ZERO_DEFICIT.jl")
include("CNTRY_INDX.jl")
include("SOLVE_SHOCK.jl")
include("MLZ_WELFARE_CORRECTED.jl")
include("MLZ_TRF_SYSTEM_MANY_ARGS_FIXED_COSTS.jl")
include("MLZ_NEW_DATA.jl")


function runTradMod(DATA::Matrix{Float64}; elast = 1, country::String = "USA", tariff = 0.0) 

    Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, alpha_tilde_sj3D, N, S = PREPARE_DATA(DATA,elast)

    Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, yis3D, Ris3D, Ri3D, Rj3D, alpha_tilde_sjk3D, consjs3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, LABOR, mu = CONVERT_ZERO_DEFICIT(Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, alpha_tilde_sj3D, N, S)

    indx, cntrlst = CNTRY_INDX(country)

    x_fsolve, ysjk3D, Dj_h3D, tijs3D, tijs_p3D, phiijs_h3D = SOLVE_SHOCK(indx, tariff, Yi3D,Yj3D,Lijs3D,Dj3D,betajs3D,epsilon_s3D,yis3D,alpha_tilde_sjk3D,theta_s3D,eta_s3D,delta_s3D,sigma_s3D,LABOR,mu,N,S)

    Cj = MLZ_WELFARE_CORRECTED(x_fsolve, N , S, mu, Yi3D, Yj3D, ysjk3D, Ris3D, Dj3D, Dj_h3D, alpha_tilde_sjk3D, betajs3D, consjs3D, epsilon_s3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, tijs3D, tijs_p3D, phiijs_h3D, Lijs3D, LABOR)
    Cj = (Cj .- 1) * 100

    t = Table(Country = cntrlst, Welfare_effect = vec(Cj))

    return t
end

end