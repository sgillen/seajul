include("ars.jl")

using .ARS
using Plots
using LinearAlgebra
using Distributed
using ControlSystems

plotlyjs()

# ======== Init LQR Problem ========
state_dim = 3
control_dim = 3

A = [1.01 0.01 0.0; 0.01 1.01 0.01; 0.0 0.01 1.01]
B = Matrix(1.0I,state_dim,state_dim)
C = Matrix(1.0I,state_dim,control_dim)
D = 0

R = Matrix(1e-3I,state_dim,state_dim)
Q = Matrix(1.0I,control_dim,control_dim)

env = LQREnv(A,B,Q,R,10)
θ = randn(3,3)

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
plot(X', title="System Rollout Before Training")


# ======== Train LQR ========

@time θ = ars!(env, θ, N=32, num_epochs=1000)
θ = randn(3,3)
@time θ = ars!(env, θ, N=32, num_epochs=50000)

# ======== Eval LQR ========
x_init = randn(size(env.A)[1])*10
print(x_init)
A = env.A; B = env.B; Q = env.Q; R = env.R 

X = zeros(size(A)[1], env.trial_length)
reward = 0.0 

x = copy(x_init)
for t = 1:env.trial_length
    X[:,t] = x
    u = θ*x
    x = A*x + B*u
    reward -= x'*Q*x +  u'*R*u        
end 

plot(X', title="System After Training")

# ======== Compare to optimal ========
sys = ss(A,B,C,D, .01)
K = lqr(sys,Q,R)

X = zeros(size(A)[1], env.trial_length)
reward = 0.0 

print(x_init)
x = copy(x_init)
for t = 1:env.trial_length
    X[:,t] = x
    u = -K*x
    x = A*x + B*u
    reward -= x'*Q*x +  u'*R*u        
end 

plot(X', title="LQR")

