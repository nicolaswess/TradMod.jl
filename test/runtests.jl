using TradMod
using Test

@testset "TradMod_tests.jl" begin
    
    # generate toy data
    DATA = rand(100:1000, 1443, 1641)
    epsilon = rand(1:10, 35, 2)
    AggC = TradMod.AGG_C()
    AggS = TradMod.AGG_S()

    # verify that PREPARE_DATA dimensions are correct
    Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, alpha_tilde_sj3D, N, S = TradMod.PREPARE_DATA(DATA,epsilon)

    @test N == 34
    @test S == 31
    @test size(Xjs3D) == (N, N, S)
    @test size(Xijs3D) == (N, N, S)
    @test size(betajs3D) == (N, N, S)
    @test size(Dj3D) == (N, N, S)
    @test size(epsilon_s3D) == (N, N, S)
    @test size(Yis3D) == (N, N, S)
    @test size(Yi3D) == (N, N, S)
    @test size(Yj3D) == (N, N, S)
    @test size(Lijs3D) == (N, N, S)
    @test size(alpha_tilde_sj3D) == (S, N, S)

    # verify that CONVERT_ZERO_DEFICIT dimensions are correct
    Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, yis3D, Ris3D, Ri3D, Rj3D, alpha_tilde_sjk3D, consjs3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, LABOR, mu  = CONVERT_ZERO_DEFICIT(Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, alpha_tilde_sj3D, N, S)
    
    @test size(Xjs3D) == (N, N, S)
    @test size(Xijs3D) == (N, N, S)
    @test size(betajs3D) == (N, N, S)
    @test size(Dj3D) == (N, N, S)
    @test size(epsilon_s3D) == (N, N, S)
    @test size(Yis3D) == (N, N, S)
    @test size(Yi3D) == (N, N, S)
    @test size(Yj3D) == (N, N, S)
    @test size(Lijs3D) == (N, N, S)
    @test size(yis3D) == (N, N, S)
    @test size(Ris3D) == (N, N, S)
    @test size(Ri3D) == (N, N, S)
    @test size(Rj3D) == (N, N, S)
    @test size(alpha_tilde_sjk3D) == (S, N, S)
    @test size(consjs3D) == (N, N, S)
    @test size(theta_s3D) == (N, N, S)
    @test size(eta_s3D) == (N, N, S)
    @test size(delta_s3D) == (N, N, S)
    @test size(sigma_s3D) == (N, N, S)
    @test LABOR == 0
    @test mu == 1

    # verify that incorrect country input yields an error
    @test CNTRY_INDX("XXX") == "Error: country not found"

    # verify that SOLVE_SHOCK dimensions are correct
    indx, cntrlst = CNTRY_INDX("USA")
    x_fsolve, ysjk3D, Dj_h3D, tijs3D, tijs_p3D, phiijs_h3D = SOLVE_SHOCK(indx,0.4,Yi3D,Yj3D,Lijs3D,Dj3D,betajs3D,epsilon_s3D,yis3D,alpha_tilde_sjk3D,theta_s3D,eta_s3D,delta_s3D,sigma_s3D,LABOR,mu,N,S)

    @test size(x_fsolve) == (2142, 1)
    @test size(ysjk3D) == (S, N, S)
    @test size(Dj_h3D) == (N, N, S)
    @test size(tijs3D) == (N, N, S)
    @test size(tijs_p3D) == (N, N, S)
    @test size(phiijs_h3D) == (N, N, S)

    # verify that MLZ_WELFARE_CORRECTED dimensions are correct
    Cj = MLZ_WELFARE_CORRECTED(x_fsolve, N , S, mu, Yi3D, Yj3D, ysjk3D, Ris3D, Dj3D, Dj_h3D, alpha_tilde_sjk3D, betajs3D, consjs3D, epsilon_s3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, tijs3D, tijs_p3D, phiijs_h3D, Lijs3D, LABOR)
    Cj = (Cj .- 1) * 100
    
    @test size(Cj) == (N, 1)

    # verify that zero tariff has zero effects
    x_fsolve, ysjk3D, Dj_h3D, tijs3D, tijs_p3D, phiijs_h3D = SOLVE_SHOCK(indx,0.0,Yi3D,Yj3D,Lijs3D,Dj3D,betajs3D,epsilon_s3D,yis3D,alpha_tilde_sjk3D,theta_s3D,eta_s3D,delta_s3D,sigma_s3D,LABOR,mu,N,S)
    Cj = MLZ_WELFARE_CORRECTED(x_fsolve, N , S, mu, Yi3D, Yj3D, ysjk3D, Ris3D, Dj3D, Dj_h3D, alpha_tilde_sjk3D, betajs3D, consjs3D, epsilon_s3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, tijs3D, tijs_p3D, phiijs_h3D, Lijs3D, LABOR)
    Cj = (Cj .- 1) * 100

    @test sum(Cj) == 0.0
    
end

