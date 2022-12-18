function MLZ_NEW_DATA(x_fsolve, N , S, mu, Xijs3D, Yi3D, Yj3D, yis3D, ysjk3D, Dj3D, Dj_h3D, alpha_tilde_sjk3D, betajs3D, epsilon_s3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, tijs3D, tijs_p3D, phiijs_h3D, Lijs3D, LABOR)

    ln = ones(N,1)

    # construct 3D cubes from 1D vector X0
    cis_h = x_fsolve[1 : N * S, 1]
    cis_h2D = reshape(cis_h,(S,N))
    cis_h3D = reshape(cis_h2D' ⊗ ln',(N,N,S))
    yis_h = x_fsolve[(N * S + 1) : (2 * N * S), 1]
    yis_h2D = reshape(yis_h,(S,N))
    yis_h3D = reshape(yis_h2D' ⊗ ln',(N,N,S))
    wi_h = x_fsolve[(2 * N * S + 1):end, 1]
    wi_h3D = repeat(wi_h, outer = [1 N S])
    wj_h3D = permutedims(wi_h3D, (2, 1, 3))
    ysjk_h3D = repeat(yis_h2D, outer = [1 1 S])
    wsjk_h3D = repeat(wi_h', outer = [S 1 S])

    # construct various aggregations of alpha
    alpha_sjk3D = alpha_tilde_sjk3D
    alpha_sj3D = repeat(sum(alpha_sjk3D, dims = 3), outer = [1 1 S])
    alpha_si3D = permutedims(repeat(alpha_sj3D[:,:,1], outer = [1 1 N]),(2, 3, 1))

    # construct various aggregations of Y
    Yi=Yi3D[:,1,1]
    Ysjk3D = repeat(Yi', outer = [S 1 S])
    
    # equation 10
    kj3D = 1 ./ (1 .- repeat(sum(sum(tijs3D .* Lijs3D .* betajs3D ./ (1 .+ tijs3D), dims = 3), dims = 1), outer = [N 1 S]))
    AUX5_INTsjk = alpha_sjk3D .* ysjk3D .* Ysjk3D ./ (1 .- alpha_sj3D)
    AUX5_INT = repeat(sum(AUX5_INTsjk, dims = 1), outer = [N 1 1])
    AUX6_INT = repeat(sum(sum(tijs3D .* Lijs3D .* AUX5_INT ./ (1 .+ tijs3D), dims = 3), dims = 1),outer = [N 1 S])
    Ejs3D = betajs3D .* kj3D .* (Yj3D + Dj3D + AUX6_INT) + AUX5_INT
    
    # equation 15
    AUX0 = (yis_h3D .* wi_h3D ./ cis_h3D).^delta_s3D .* (cis_h3D .* phiijs_h3D).^(-theta_s3D) .* (cis_h3D.^(-eta_s3D .* delta_s3D))
    AUX1 = repeat(sum(Lijs3D .*  AUX0, dims = 1), outer = [N 1 1])

    # 1. Compute lambda_hat_ijs
    AUX2 = Lijs3D .* AUX0 ./ AUX1
    Lijs_new3D = AUX2
    Lijs_h3D = AUX0 ./ AUX1

    # 2. Compute "new" Y_i (for D=0)
    Yi_new3D = Yi3D .* wi_h3D

    # 3. Compute "new" y_is, Yis
    yis_new3D = yis3D .* yis_h3D
    Yis_new3D = yis_new3D .* Yi_new3D

    # equation 16
    AUX3sjk = alpha_sjk3D .* ysjk_h3D .* ysjk3D .* Ysjk3D .* wsjk_h3D ./ (1 .- alpha_sj3D)
    AUX3 = repeat(sum(AUX3sjk, dims = 1), outer = [N 1 1])
    AUX3_5 = repeat(sum(sum(tijs_p3D .* AUX2 .* AUX3 ./ (1 .+ tijs_p3D), dims = 3), dims = 1), outer = [N 1 S])
    kj_p3D = 1 ./ (1 .- repeat(sum(sum(tijs_p3D .* AUX2 .* betajs3D ./ (1 .+ tijs_p3D), dims = 3), dims = 1), outer = [N 1 S]))
    AUX4 = betajs3D .* kj_p3D .* (wj_h3D .* Yj3D + Dj3D .* Dj_h3D .* (wj_h3D.^mu) + AUX3_5) + AUX3
    AUX7 = AUX4 ./ Ejs3D
    Ejs_h3D = AUX7

    # 6. Compute Xijs_h3D and Xijs_new3D
    Xijs_h3D = Lijs_h3D .* Ejs_h3D
    Xijs_new3D = Xijs_h3D .* Xijs3D
    Xjs_new3D = repeat(sum(Xijs_new3D, dims = 1), outer = [N 1 1])

    # 7. Set new deficits to zeros
    Dj_new3D = Dj_h3D .* Dj3D
    Ejs_new3D = Ejs3D .* Ejs_h3D

    # 8. Rename variables (drop "_new") and compute all aux. "cubes" (e.g. Yj3D) 
    Ejs3D = Ejs_new3D
    Dj3D = Dj_new3D
    Lijs3D = Lijs_new3D
    Xijs3D = Xijs_new3D
    Xjs3D = Xjs_new3D
    Yi3D = Yi_new3D
    Yis3D = Yis_new3D
    Yj3D = permutedims(Yi3D, (2, 1, 3))
    yis3D = yis_new3D
    Ris3D = repeat(sum(Xijs3D, dims = 2), outer = [1 N 1])
    Ri3D = repeat(sum(Ris3D, dims = 3), outer = [1 1 S])
    Rj3D = permutedims(Ri3D, (2, 1, 3))


    return Dj3D, Lijs3D, Xijs3D, Xjs3D, Yi3D, Yis3D, Yj3D, yis3D, Ris3D, Ri3D, Rj3D
end