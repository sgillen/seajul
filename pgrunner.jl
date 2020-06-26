include("ppo2.jl")
import .PG
using Flux
using LinearAlgebra



pol = Chain(Dense(2,32,relu), Dense(32,4,relu))
#pol = Chain(Dense(2,4))
@time ls,rews = PG.learn!(pol,500,log_every=1)


# A = [.1 4; 0 .4]
# B = I
# t_len = 100
# x = zeros(2,t_len)
# x[:,1] = [1,2]
# u = zeros(2)

# for t in 1:t_len-1
#     x[:, t+1] = A*x[:,t] + B*u
# end


A = Matrix(1.0I,2,2)
B = Matrix(1.0I,2,2)
C = Matrix(1.0I,2,2)
D = 0

R = Matrix(1.0I,2,2)
Q = Matrix(1.0I,2,2)

using ControlSystems
s = ss(A,B,C,D,.01)
K = lqr(s,Q,R)

cur_obs = [1.0, 1.0]
ep_length = 50
for t = 1:ep_length
    obs[:,t] = cur_obs
    acts[:,t] =-K*cur_obs
    global cur_obs = A*obs[:,t] + B*acts[:,t]
    #obs1[:,t+1] = rk4(dynamics, acts[:,t], 0.0f0, dt, obs1[:,t])
    rews[t] = -(cur_obs'*Q*cur_obs +  acts[:,t]'*R*acts[:,t])
end 
plot(obs')

cur_obs = [1.0, 1.0]
ep_length = 50
for t = 1:ep_length
    obs[:,t] = cur_obs
    acts[:,t] = PG.sample_action(policy, cur_obs)
    global cur_obs = A*obs[:,t] + B*acts[:,t]
    #obs1[:,t+1] = rk4(dynamics, acts[:,t], 0.0f0, dt, obs1[:,t])
    rews[t] = -(cur_obs'*Q*cur_obs +  acts[:,t]'*R*acts[:,t])
end 
plot(obs')