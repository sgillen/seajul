include("ars.jl")

using .ARS
using Plots
using LinearAlgebra
using Distributed
using ControlSystems

plotlyjs()

let 

    l = @layout [a; b]

    A = [1.1 2.2; 2.2 1.1]
    B = Matrix(1.0I,2,2)
    R = Matrix(1e-3I,2,2)
    Q = Matrix(1.0I,2,2)

    env = LQREnv(A,B,Q,R,25)
    θ = randn(2,2)

    x = randn(size(env.A)[1])
    A = env.A; B = env.B; Q = env.Q; R = env.R 
    reward = 0.0
    X = zeros(size(A)[1], env.trial_length)

    for t = 1:env.trial_length
        u = θ*x
        x = A*x + B*u
        X[:,t] = x
        reward -= x'*Q*x +  u'*R*u
    end 
    p1 = plot(X')


    @time θ = ars!(env, θ, N=32, num_epochs=1000)
    θ = randn(2,2)
    @time θ = ars!(env, θ, N=32, num_epochs=1000)
    

    x = randn(size(env.A)[1])
    A = env.A; B = env.B; Q = env.Q; R = env.R 
    reward = 0.0
    X = zeros(size(A)[1], env.trial_length)

    for t = 1:env.trial_length
        u = θ*x
        x = A*x + B*u
        X[:,t] = x
        reward -= x'*Q*x +  u'*R*u
    end 

    p2 = plot(X')
    #print(X')

    plot(p1,p2, layout=l)
end