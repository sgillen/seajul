include("seajul.jl")

using PyCall
using .SJ

gym = pyimport("gym")
mj = pyimport("mujoco_py")

env = gym.make("Walker2d-v2")