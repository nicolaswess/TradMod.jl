function MLZ_TRF_SYSTEM_MANY_ARGS_FIXED_COSTS(X0, N , S, mu, Yi3D, Yj3D, yis3D, ysjk3D, Dj3D, Dj_h3D, alpha_tilde_sjk3D, betajs3D, epsilon_s3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, tijs3D, tijs_p3D, phiijs_h3D, Lijs3D, LABOR)
    
    # summer vectors
    ln = ones(N,1)
    X0 = (X0).^2

    # construct 3D cubes from 1D vector X0
    cis_h = X0[1 : N * S,1]
    cis_h2D = reshape(cis_h,(S,N))
    cis_h3D = reshape(cis_h2D' ⊗ ln',(N,N,S))
    cjs_h3D = permutedims(cis_h3D,(2, 1, 3))
    yis_h = X0[(N * S + 1) : (2 * N * S),1]
    yis_h2D = reshape(yis_h,(S,N))
    yis_h3D = reshape(yis_h2D' ⊗ ln',(N,N,S))
    wi_h = X0[(2 * N * S + 1) : end,1]
    wi_h3D = repeat(wi_h, outer = [1 N S])
    wj_h3D = permutedims(wi_h3D,(2, 1, 3))
    ysjk_h3D = repeat(yis_h2D, outer = [1 1 S])
    wsjk_h3D = repeat(wi_h', outer = [S 1 S])

    # construct various aggregations of alpha
    alpha_sjk3D = alpha_tilde_sjk3D
    alpha_sj3D = repeat(sum(alpha_sjk3D, dims = 3), outer = [1 1 S])
    alpha_si3D = permutedims(repeat(alpha_sj3D[:,:,1], outer = [1 1 N]),(2, 3, 1))

    # construct various aggregations of Y
    Yi = Yi3D[:,1,1]
    Ysjk3D = repeat(Yi', outer = [S 1 S])

    # equation 10
    kj3D = 1 ./ (1 .- repeat(sum(sum(((1 .- 1 ./ sigma_s3D).^delta_s3D) .* tijs3D .* Lijs3D .* betajs3D ./ (1 .+ tijs3D), dims = 3), dims = 1), outer = [N 1 S]))
    AUX5_INTsjk = alpha_sjk3D .* ysjk3D .* Ysjk3D ./ (1 .- alpha_sj3D)
    AUX5_INT = repeat(sum(AUX5_INTsjk, dims = 1), outer = [N 1 1])
    AUX6_INT = repeat(sum(sum(((1 .- 1 ./ sigma_s3D).^delta_s3D) .* tijs3D .* Lijs3D .* AUX5_INT ./ (1 .+ tijs3D), dims = 3), dims = 1), outer = [N 1 S])
    Ejs3D = betajs3D .* kj3D .* (Yj3D + Dj3D + AUX6_INT) + AUX5_INT

    # equation 15
    AUX0 = Lijs3D .* (yis_h3D .* wi_h3D ./ cis_h3D).^delta_s3D .* (cis_h3D .* phiijs_h3D).^(-theta_s3D) .* (cis_h3D.^(-eta_s3D .* delta_s3D))
    AUX1 = repeat(sum(AUX0, dims = 1), outer = [N 1 1])
    AUX2 = AUX0 ./ AUX1

    # equation 16
    AUX3sjk = alpha_sjk3D .* ysjk_h3D .* ysjk3D .* Ysjk3D .* wsjk_h3D ./ (1 .- alpha_sj3D)
    AUX3 = repeat(sum(AUX3sjk, dims = 1), outer = [N 1 1])
    AUX3_5 = repeat(sum(sum(((1 .- 1 ./ sigma_s3D).^delta_s3D) .* tijs_p3D .* AUX2 .* AUX3 ./ (1 .+ tijs_p3D),dims = 3),dims = 1), outer = [N 1 S])
    kj_p3D = 1 ./ (1 .- repeat(sum(sum(((1 .- 1 ./ sigma_s3D).^delta_s3D) .* tijs_p3D .* AUX2 .* betajs3D ./ (1 .+ tijs_p3D), dims = 3), dims = 1), outer = [N 1 S]))
    AUX4 = betajs3D .* kj_p3D .* (wj_h3D .* Yj3D + Dj3D .* Dj_h3D .* (wj_h3D.^mu) + AUX3_5) + AUX3
    AUX7 = AUX4 ./ Ejs3D

    #-------------------------------------------------------------------------------------#
    #-- minimize differences between left- and righthand sides of the following system -- #
    #-------------------------------------------------------------------------------------#

    # === ERR1 === 
    # equation 14
    AUX9 = Lijs3D .* ((yis_h3D .* wi_h3D ./ cis_h3D).^delta_s3D) .* ((cis_h3D .* phiijs_h3D).^(-theta_s3D)) .* ((AUX7 ./ cis_h3D).^(delta_s3D .* eta_s3D))
    AUX9sjk = repeat(sum(AUX9,dims = 1), outer = [S 1 1])
    theta_k3D = repeat(permutedims(cat(theta_s3D[1,1,:], dims=3),(3,2,1)), outer = [S N 1])
    AUX8 = -alpha_sjk3D ./ theta_k3D
    AUX10sjk = AUX9sjk.^AUX8

    # equation 13
    AUX10 = permutedims(repeat(prod(AUX10sjk, dims = 3), outer = [1 1 N]),(3, 2, 1))
    ERR1js3D = cjs_h3D - (wj_h3D.^(1 .- permutedims(alpha_si3D,(2, 1, 3)))) .* AUX10
    ERR1sji = permutedims(ERR1js3D,(3, 2, 1))
    ERR1 = reshape(ERR1sji[:,:,1],(N * S,1))

    # === ERR2 === 
    # equation 18
    AUX11 = wi_h3D .* yis_h3D .* yis3D .* Yi3D ./ (1 .-alpha_si3D)

    # equation 17
    AUX12 = repeat(sum(((1 .+ tijs_p3D ./ sigma_s3D).^delta_s3D) .* AUX2 .* AUX4 ./ (1 .+ tijs_p3D), dims = 2), outer = [1 N 1])
    ERR2ijs = AUX11 - AUX12
    ERR2is = reshape(ERR2ijs[:,1,:],(N,S))
    ERR2 = reshape(ERR2is',(N*S,1))

    # === ERR3 === 
    # equation 19
    EQ3LHS = sum(yis_h3D.*yis3D, dims = 3)
    EQ3LHS = cat(EQ3LHS[:,:,1], dims=2)
    ERR3 = EQ3LHS[:,1] .- 1
    ERR3[N,1] = sum(Yi3D[:,1,1] .* (wi_h .- 1))

    # === ALL ERRORS ===
    ERRis = [ERR1;ERR2;ERR3]

    return ERRis
    
end