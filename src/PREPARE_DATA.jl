function PREPARE_DATA(DATA,epsilon)

    # generate aggregation schemes
    AggC = AGG_C()
    AggS = AGG_S()

    # Number of regions and sectors
    N = 41
    S = 35;

    # prepare data 
    Zinit = DATA[1:1435,1:1435]
    X = DATA[1:1435,1:1640] # extract data on intermediate input flows
    Rinit = sum(X, dims = 2); # extract data on both flows of intemediate and final goods

    # Make the required corrections
    FIN = X[1:1435, 1436:1640]
    FINsum = sum(FIN, dims = 2)
    F = copy(FIN) 
    F .= ifelse.(F .< 0.0, 0.0, F)
    Fsum = sum(F, dims = 2)
    Rinit .= ifelse.(Rinit .< 0.000001, Rinit .+ 0.000001, Rinit)
    A = Zinit / diagm(vec(Rinit))
    R = (I(N * S) - A) \ (Fsum)
    Z = A * diagm(vec(R));

    # Aggregate sectors & countries
    N = size(AggC,1)
    S = size(AggS,1)
    CC = AggC ⊗ AggS
    FF = AggC ⊗ I(5)
    Z = CC * Z * CC'
    F = CC * F * FF'
    X = hcat(Z,F)
    R = sum(X, dims = 2);

    # Compute the required data
    ls = ones(S,1)
    ln = ones(N,1)
    l5 = ones(5,1)
    AUX1 = I(N) ⊗ ls
    AUX2 = I(N) ⊗ l5
    AUX = vcat(AUX1,AUX2)
    Xijs3D = permutedims(reshape(X*AUX,(S,N,N)),(2, 3, 1))
    Xjs3D = repeat(sum(Xijs3D, dims = 1), outer = [N 1 1])
    Xj3D = repeat(sum(Xjs3D, dims = 3), outer = [1 1 S])
    Lijs3D = Xijs3D ./ Xjs3D

    Ris3D = permutedims(repeat(reshape(sum(X, dims = 2),(S,N)), outer = [1 1 N]),(2, 3, 1))
    Rjs3D = permutedims(Ris3D,(2, 1, 3))
    Ri3D = repeat(sum(Ris3D, dims = 3), outer = [1 1 S])
    Rj3D = permutedims(Ri3D,(2, 1, 3))

    Djs3D = Xjs3D - Rjs3D
    Dj3D = repeat(sum(Djs3D, dims = 3), outer = [1 1 S])

    l_alpha = repeat(I(S), outer = [1 N])
    Zsjk3D = permutedims(reshape((l_alpha*Z),(S,S,N)),(2, 3, 1))
    Rsjk3D = repeat((reshape(Rjs3D[1,:,:],(N,S)))', outer = [1 1 S])
    Rsjk3D .= ifelse.(Rsjk3D .== 0.0, Rsjk3D .+ +0.00000001, Rsjk3D)
    alpha_tilde_sjk3D = Zsjk3D ./ Rsjk3D
    alpha_tilde_sj3D = repeat(sum(alpha_tilde_sjk3D, dims = 3), outer = [1 1 S])
    alpha_tilde_sj2D = alpha_tilde_sj3D[:,:,1]

    ejs3D = Xjs3D ./ Xj3D
    ejk_sjk3D = repeat(permutedims(cat(ejs3D[1,:,:], dims=3),(3, 1, 2)), outer = [S 1 1])
    alpha_BHR_sjk3D = alpha_tilde_sj3D .* ejk_sjk3D
    alpha_BHR_sj3D = repeat(sum(alpha_BHR_sjk3D, dims = 3), outer = [1 1 S])

    AUX0 = I(N) ⊗ l5
    Fijs3D = permutedims(reshape(F*AUX0,(S,N,N)),(2, 3, 1))
    Fjs3D = repeat(sum(Fijs3D, dims = 1), outer = [N 1 1])
    Fj3D = repeat(sum(Fjs3D, dims = 3), outer = [1 1 S])
    betajs3D = Fjs3D ./ Fj3D

    VAis = R' - sum(Z, dims = 1)
    VAis .= ifelse.(VAis .< 0.0, VAis .* +0.0, VAis)
    Yis3D = permutedims(repeat(reshape(VAis,(S,N)), outer = [1 1 N]),(2, 3, 1))
    Yi3D=repeat(sum(Yis3D, dims = 3), outer = [1 1 S])
    Yj3D=permutedims(Yi3D,(2, 1, 3))

    epsilon_s = epsilon[:,2]
    deleteat!(epsilon_s, (5,19,31,34))
    for i in eachindex(epsilon_s)
        epsilon_s[i] = ifelse(i >= 16, 5, epsilon_s[i])
    end
    epsilon_s3D = reshape(epsilon_s' ⊗ ones(N,N),(N,N,S));

    return Xjs3D, Xijs3D, betajs3D, Dj3D, epsilon_s3D, Yis3D, Yi3D, Yj3D, Lijs3D, alpha_tilde_sj3D, N, S
end