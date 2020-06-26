module ws

include("seajul.jl")

using .SJ
using Revise
using PyCall

gym = pyimport("gym")
pyimport("pyullet_envs")
env = gym.make("Walker2DBulletEnv-v0")

end


import Pkg
function pycall_activate(env_name)
    ENV["Python"] = "/home/sgillen/miniconda3/envs/$(env_name)/bin/python"
    Pkg.build("PyCall")
    println("restart Julia now dummy")

end

pycall_activate("stable")