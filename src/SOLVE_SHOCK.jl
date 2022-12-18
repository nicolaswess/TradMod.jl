function SOLVE_SHOCK(indx, tariff, Yi3D,Yj3D,Lijs3D,Dj3D,betajs3D,epsilon_s3D,yis3D,alpha_tilde_sjk3D,theta_s3D,eta_s3D,delta_s3D,sigma_s3D,LABOR,mu,N,S)
    # Set shocks 

    # Initial tariff
    t =                     0
    tij2D =                 t * (ones(N,N) - I(N))
    tijs3D =                repeat(tij2D, outer = [1 1 S])

    # New tariffs
    t_p =                   tariff
    tijs_p3D =              zeros(N, N, S)      
    tijs_p3D[:,indx,:] .=      t_p
    tijs_p3D[indx,indx,:] .=     0

    # Changes in transportation costs
    tau_h =                 1
    tauij_h2D =             tau_h * (ones(N,N) - I(N)) + I(N)
    tauijs_h3D =            repeat(tauij_h2D, outer = [1 1 S])

    #Changes in trade deficits
    Dj_h3D =                ones(N,N,S);

    # Compute auxiliary variables (not dependent on hats)
    ysi3D =          permutedims(yis3D,(3, 1, 2,))
    ysi2D =          ysi3D[:,:,1]
    ysjk3D =         repeat(ysi2D, outer = [1 1 S])
    phiijs_h3D =     tauijs_h3D .* (1 .+ tijs_p3D) ./ (1 .+ tijs3D);

    # Solve the model
    X0 = ones(N * (2 * S + 1),1)
    syst(X) = MLZ_TRF_SYSTEM_MANY_ARGS_FIXED_COSTS(X, N , S, mu, Yi3D, Yj3D, yis3D, ysjk3D, Dj3D, Dj_h3D, alpha_tilde_sjk3D, betajs3D, epsilon_s3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, tijs3D, tijs_p3D, phiijs_h3D, Lijs3D, LABOR)

    fsolve = nlsolve(syst, X0, iterations = 100, ftol=1E-10, xtol=1E-12)
    x_fsolve = fsolve.zero
    x_fsolve = (x_fsolve).^2;

    return x_fsolve, ysjk3D, Dj_h3D, tijs3D, tijs_p3D, phiijs_h3D

end