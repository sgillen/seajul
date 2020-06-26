@time using Flux
@time using Flux.Optimise: update!
@time using Gym
@time using Plots
@time using DifferentialEquations
@time using Distributions

model = Chain(Dense(2,64,relu), Dense(64,2))
loss(x,y) = sum((model(x) - y).^2)/size(y)[2]
θ = Flux.params(model.layers)
xtrain = rand(2, 2000);
ytrain = rand(2, 2000);
opt = ADAM(0.1) # Gradient descent with learning rate 0.1

loss_hist = zeros(10)
for i = 1:100
    gs = gradient(()->loss(xtrain,ytrain), θ)

    for p in θ
      update!(opt, p, gs[p])
    end

    if i % 10 == 0
        loss_hist[i÷10] = loss(xtrain,ytrain)
    end
end
 
#====================================#

# plotlyjs()
# plot(loss_hist)
 
# function linearz!(du,u,p,t)
#     du[1] = u[1]
#     du[2] = 0#model(u)[1]
# end

# tspan = (0.0, 10.0)
# q0 = [1.0, 1.0]
# prob = ODEProblem(linearz!,q0,tspan)

# @time sol = solve(prob)
# plot(sol)

# #====================================#

# #====================================#

# function f(du,u,p,t)
#     du[1] = -0.5*u[1] + model(u)
#     du[2] = -0.5*u[2]
# end


# # Dicrete callback
# #====================================#
# mutable struct SimType{T} <: DEDataVector{T}
#     x::Array{T,1}
#     act::T
# end

# tstop = 0:.01:500
# function pol_update_condition(u,t,integrator)
#     t in tstop
# end

# function change_act!(integrator)
#     act = 5
#     for c in full_cache(integrator)
#         c.act = act
#     end
# end

# save_positions = (true,true)
# cb = DiscreteCallback(pol_update_condition, change_act!, save_positions=save_positions)

# u0 = SimType([10.0;10.0], 0.0)
# prob = ODEProblem(f,u0,(0.0,10.0))
# sol = solve(prob,callback=cb, tstops=tstop)
# plot(sol)

#====================================#
# SDE
#====================================#

u0 = (10.0;10.0)
mask = zeros(Float32, 2,2); mask[1,1] = 1.0f0
prob = SDEProblem(f, ()->mask, u0, (0.0,10.0))
sol = solve(prob)

# Rollout
#====================================#

function wrap(x, low, high)
    return (x - low) % (high - low) + low
end

function rk4(derivs, a, t0, dt, s0)
    k1 = dt * derivs(t0, s0, a)
    k2 = dt * derivs(t0 + dt / 2, s0 + k1 / 2, a)
    k3 = dt * derivs(t0 + dt / 2, s0 + k2 / 2, a)
    k4 = dt * derivs(t0 + dt, s0 + k3, a)
    return s0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
end

function euler(derivs, a, t0, dt, s0)
    return s0 + dt * derivs(t0 + dt, s0, a)
end

function sample_action(policy,o)
    out = policy(o)
    n_act = size(out)[1]÷2
    
    means = out[1:n_act]
    logstds = out[n_act+1:end]
    stds = ℯ.^logstds
    
    actions = stds.*randn(Float32, n_act) + means
    logprob = -.5*((actions - means)./(stds)).^2 - logstds .- log(√(2.f0*π))
    
    return actions, logprob
end 

function env_fn(du,u,p,t)
    du[1] = -0.5*u[1] + p[1]
    du[2] = -0.5*u[2]
end

function do_rollout(policy)
    ep_length=500
    acts = zeros(Float32, ep_length)
    obs1 = zeros(Float32, ep_length)
    
    
    obs = [10.0f0, 10.0f0]
    dt = .01f0
    i = 0
    for t = 1:dt:ep_length
        act,_ = sample_action(policy, obs)

        

        #ode = ODEProblem(env_fn,obs,[0.0,10.0],act)
        #sol = solve(ode,tspan=(t,t+dt))
        #obs = sol[end]

        obs1[i] = obs
        acts[i] = act          
    end 

    return (acts, obs1)
end

sample_action(model,[0.0, 0.0])