function CONVERT_ZERO_DEFICIT(Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, alpha_tilde_sj3D, N, S)
    # Adjust alphas as (alpha_new_j,ks=alpha_j,s*e_j,k)
    Xj3D = repeat(sum(Xjs3D,dims = 3), outer = [1 1 S])
    ejs3D = Xjs3D ./ Xj3D
    consjs3D = betajs3D
    betajs3D = ejs3D
    ejk_sjk3D = repeat(permutedims(cat(ejs3D[1,:,:], dims=3),(3, 1, 2)), outer = [S 1 1])
    alpha_tilde_sjk3D = alpha_tilde_sj3D .* ejk_sjk3D
    alpha_tilde_sj3D = repeat(sum(alpha_tilde_sjk3D, dims = 3), outer = [1 1 S]);

    # Set shocks

    # Initial tariff
    t =             0
    tij2D =         t * (ones(N,N) - I(N))
    tijs3D =        repeat(tij2D, outer = [1 1 S])

    # New tariffs
    t_p =           0
    tij_p2D =       t_p * (ones(N,N) - I(N))
    tijs_p3D =      repeat(tij_p2D, outer = [1 1 S])

    # Changes in transportation costs
    tau_h =         1
    tauij_h2D =     tau_h * (ones(N,N) - I(N)) + I(N)
    tauijs_h3D =    repeat(tauij_h2D, outer = [1 1 S])

    #Changes in trade deficits
    Dj_h3D =        zeros(N,N,S);


    # Set parameters
    delta_s3D =      zeros(N,N,S)
    eta_s3D =        zeros(N,N,S)
    mu =             1
    LABOR =          0;

    # Compute auxiliary variables (not dependent on hats)
    sigma_s3D =      1 .+ epsilon_s3D ./ (1 .+ eta_s3D)
    theta_s3D =      epsilon_s3D
    yis3D =          Yis3D ./ Yi3D
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

    # Compute the new data
    Dj3D, Lijs3D, Xijs3D, Xjs3D, Yi3D, Yis3D, Yj3D, yis3D, Ris3D, Ri3D, Rj3D = MLZ_NEW_DATA(x_fsolve, N , S, mu, Xijs3D, Yi3D, Yj3D, yis3D, ysjk3D, Dj3D, Dj_h3D, alpha_tilde_sjk3D, betajs3D, epsilon_s3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, tijs3D, tijs_p3D, phiijs_h3D, Lijs3D, LABOR);

    return Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, yis3D, Ris3D, Ri3D, Rj3D, alpha_tilde_sjk3D, consjs3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, LABOR, mu

end