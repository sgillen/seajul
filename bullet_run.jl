module ws

include("seajul.jl")

using .SJ
using Revise
using PyCall

gym = pyimport("gym")
pyimport("pybullet_envs")
env = gym.make("Walker2DBulletEnv-v0")

W = zeros(Float32, 6,22)
W,rews = SJ.ars_v1t!(env,W,5)
#@time W,rews = SJ.ars_v1t!(env,W,num_epochs=10)

# function do_rollout_eval(env::PyObject, W)
#     x = env.reset()
#     done = false
#     reward = 0.0

#     x_hist = zeros(env.observation_space.shape[1],env._max_episode_steps)
#     i = 1

#     act_low = convert(Array{Float64,1}, env.action_space.low)
#     act_high = convert(Array{Float64,1}, env.action_space.high)

#     while !done
#         x_hist[:,i] = copy(x); i+=1
#         print(W*x)
#         u = clamp(W*x, act_low, act_high)
#         x, r, done, _ = env.step(u)
#         reward += r
#     end
#     #println(vec(mean(x_hist[:,1:i-1],dims=2)))
#     return reward, x_hist
# end

# R,X = do_rollout_eval(env,W)

end



# import Pkg
# function pycall_activate(env_name)
#     ENV["Python"] = "/home/sgillen/miniconda3/envs/$(env_name)/bin/python"
#     Pkg.build("PyCall")
#     println("restart Julia now dummy")

# end

# pycall_activate("stable")



function f2(a)
    return sum(a)
end

function f(a::Array{<:AbstractFloat,1})
    return sum(a)
end

function g(a::Array{Float64,1})
    return sum(a)
end

function h(a::Array{AbstractFloat,1})
    return sum(a)
end