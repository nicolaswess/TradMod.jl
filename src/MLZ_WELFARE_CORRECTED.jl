function MLZ_WELFARE_CORRECTED(x_fsolve, N , S, mu, Yi3D, Yj3D, ysjk3D, Ris3D, Dj3D, Dj_h3D, alpha_tilde_sjk3D, betajs3D, consjs3D, epsilon_s3D, theta_s3D, eta_s3D, delta_s3D, sigma_s3D, tijs3D, tijs_p3D, phiijs_h3D, Lijs3D, LABOR)

    ln = ones(N,1)

    # construct 3D cubes from 1D vector X0
    cis_h = x_fsolve[1 : N * S, 1]
    cis_h2D = reshape(cis_h,(S,N))
    cis_h3D = reshape(cis_h2D' ⊗ ln',(N,N,S))
    yis_h = x_fsolve[(N * S + 1) : (2 * N * S), 1]
    yis_h2D = reshape(yis_h,(S,N))
    yis_h3D = reshape(yis_h2D' ⊗ ln',(N,N,S))
    yjs_h3D = permutedims(yis_h3D,(2, 1, 3))
    wi_h = x_fsolve[(2 * N * S + 1):end, 1]
    wi_h3D = repeat(wi_h, outer = [1 N S])
    wj_h3D = permutedims(wi_h3D, (2, 1, 3))
    ysjk_h3D = repeat(yis_h2D, outer = [1 1 S])
    wsjk_h3D = repeat(wi_h', outer = [S 1 S])

    betajk_sjk3D = repeat(permutedims(cat(consjs3D[1,:,:], dims=3),(3,1,2)), outer = [S 1 1])
    epsilon_sjk3D = repeat(reshape(epsilon_s3D[1,1,:],(S,1)), outer = [1 N S])
    deltas_sjk3D = repeat(reshape(delta_s3D[1,1,:],(S,1)), outer = [1 N S])
    etas_sjk3D = repeat(reshape(eta_s3D[1,1,:],(S,1)), outer = [1 N S])
    
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
    AUX0 = Lijs3D .* (yis_h3D .* wi_h3D ./ cis_h3D).^delta_s3D .* (cis_h3D .* phiijs_h3D).^(-theta_s3D) .* (cis_h3D.^(-eta_s3D .* delta_s3D))
    AUX1 = repeat(sum(AUX0, dims = 1), outer = [N 1 1])

    # 1. Compute lambda_hat_ijs
    AUX2 = AUX0 ./ AUX1
    Lijs_new3D = AUX2
    Lijs_h3D = Lijs_new3D ./ Lijs3D
    idx = (LinearIndices(I(N)))[findall(x->x == 1, I(N))]
    Lijs_h2D = reshape(Lijs_h3D, (N * N, S))
    Lijs_p3D = Lijs_new3D
    Liis_h2D = Lijs_h2D[idx,:]
    Ljjs_h3D = permutedims(repeat(Liis_h2D, outer = [1 1 N]),(3, 1, 2))

    pij_p3D = repeat(sum(sum(tijs_p3D .* Lijs_p3D .* betajs3D ./ (1 .+ tijs_p3D), dims = 3),dims = 1), outer = [N 1 S])
    Ri_p3D = repeat(sum(Ris3D .* yis_h3D .* wi_h3D, dims = 3), outer = [1 1 S])
    Rj_p3D = permutedims(Ri_p3D,(2, 1, 3))

    # equation 16
    AUX3sjk = alpha_sjk3D .* ysjk_h3D .* ysjk3D .* Ysjk3D .* wsjk_h3D ./ (1 .- alpha_sj3D)
    AUX3 = repeat(sum(AUX3sjk, dims = 1), outer = [N 1 1])
    AUX3_5 = repeat(sum(sum(tijs_p3D .* AUX2 .* AUX3 ./ (1 .+ tijs_p3D), dims = 3), dims = 1), outer = [N 1 S])
    kj_p3D = 1 ./ (1 .- repeat(sum(sum(tijs_p3D .* AUX2 .* betajs3D ./ (1 .+ tijs_p3D), dims = 3), dims = 1), outer = [N 1 S]))
    AUX4 = betajs3D .* kj_p3D .* (wj_h3D .* Yj3D + Dj3D .* Dj_h3D .* (wj_h3D.^mu) + AUX3_5) + AUX3
    Ejs_h3D = AUX4 ./ Ejs3D

    # compute a's
    alpha_p_sjk3D = alpha_sjk3D .* (1 .+ deltas_sjk3D .* (1 .+ etas_sjk3D) ./ epsilon_sjk3D)
    a_ksj3D = zeros(S,S,N)
    for j in 1:N
        Aj = reshape(alpha_p_sjk3D[:,j,:],(S,S))
        a_ksj3D[:,:,j]=(I(S) - Aj)^(-1)
    end

    a_sjk3D = permutedims(a_ksj3D,(2, 3, 1))

    # compute Cj^
    AUX9 = (yjs_h3D.^delta_s3D) .* (Ejs_h3D ./ wj_h3D).^(eta_s3D .* delta_s3D) ./ (Ljjs_h3D)
    AUX9sjk = permutedims(repeat(permutedims(cat(AUX9[1,:,:], dims=3),(3,1,2)), outer = [S,1,1]),(3, 2, 1))
    AUX10sjk = (AUX9sjk).^(betajk_sjk3D .* a_sjk3D ./ epsilon_sjk3D)
    AUX11 = repeat(prod(prod(AUX10sjk, dims = 3), dims = 1), outer = [N 1 S])
    Cj3D = (wj_h3D + Rj_p3D .* pij_p3D ./ (Yj3D .* (1 .- pij_p3D))) .* AUX11 ./ wj_h3D
    Cj = reshape(Cj3D[1,:,1],(N,1))
    
    return Cj
end