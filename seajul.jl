module SJ

using Statistics
using LinearAlgebra
using Random
using SharedArrays
using Distributed
using PyCall
using OpenAIGym

export ars!
export ars_v1t!
export ars_v2t!
export LQREnv
export LinearZEnv

# ===== PyGym Env ====================================
function do_rollout_with_stats(env::PyObject, policy)
    x = env.reset()
    done = false
    reward = 0.0

    x_hist = zeros(typeof(x[1]), env.observation_space.shape[1],env._max_episode_steps)
    i = 1

    act_low = convert(typeof(x), env.action_space.low)
    act_high = convert(typeof(x), env.action_space.high)

    while !done
        x_hist[:,i] = copy(x); i+=1
        u = clamp(policy(x), act_low, act_high)
        x, r, done, _ = env.step(u)
        reward += r
    end
    #println(vec(mean(x_hist[:,1:i-1],dims=2)))
    return reward::Float64, vec(mean(x_hist[:,1:i-1],dims=2)), vec(std(x_hist[:,1:i-1],dims=2)).+1e-6
end

function do_rollout(env::PyObject, policy)
    x = env.reset()::Array{<:AbstractFloat, 1}
    done = false
    reward = 0.0
    
    while !done
        u = clamp(policy(x), env.action_space.low, env.action_space.high)
        x, r, done, _ = env.step(u)::Tuple{Array{<:AbstractFloat,1}, Float64, Bool, PyObject}
        reward += r
    end
    #println(vec(mean(x_hist[:,1:i-1],dims=2)))
    return reward
end

## ==== LQR ======================================
struct LQREnv
    A::AbstractArray
    B::AbstractArray
    Q::AbstractArray
    R::AbstractArray
    trial_length::Int
end

function do_rollout_ars(env::LQREnv, policy)
    
    x = randn(size(env.A)[1])
    A = env.A; B = env.B; Q = env.Q; R = env.R 
    reward = 0.0

    for t = 1:env.trial_length
        u = policy(x)
        x = A*x + B*u
        reward -= x'*Q*x +  u'*R*u
    end 

    return reward
end

## ==== Original LinearZ ==========================
struct LinearZEnv
    dt::Float64
    trial_length::Int
    act_hold::Int
    act_limit::Float64
end

function do_rollout_ars(env::LinearZEnv, policy)
    x = randn(3)*10
    r = 10.0
    s = [x;r]

    act_limit = ones(size(policy(s)))*env.act_limit

    reward = 0.0 
    dynamics = (t,s,a)->[a[1],a[2],s[1]]

    for t in 1:env.trial_length
        u = clamp(policy(s), -act_limit, act_limit)
        for _ in 1:env.act_hold
            x = rk4(dynamics, u, t, env.dt, x)
            if r > 0.0
                if x[1] >= 10 && x[3] >= 10
                    reward += 5.0
                    r = -10.0
                end
            elseif r < 0.0
                if x[1] <= 10 && x[3] <= 10
                    reward += 5.0
                    r = 10.0
                end
            end
        end

        s[1:3] = x
        s[4] = r
    end
    return reward
end


## ==== Integration ==========================
function wrap(x, low, high)
    return (x - low) % (high - low) + low
end

function rk4(derivs, a, t0, dt, s0)
    k1 = dt * derivs(t0, s0, a)
    k2 = dt * derivs(t0 + dt / 2, s0 + k1 / 2, a)
    k3 = dt * derivs(t0 + dt / 2, s0 + k2 / 2, a)
    k4 = dt * derivs(t0 + dt, s0 + k3, a)
    return s0 + 1/6.0f0 * (k1 + 2 * k2 + 2 * k3 + k4)
end

function euler(derivs, a, t0, dt, s0)
    return s0 + dt * derivs(t0 + dt, s0, a)
end 


## ==== Integration ==========================
function ars_v1t!(env, θ::Array{<:AbstractFloat,2}, num_epochs::Int; α = .01, N = 32, b = 16, σ = .02)
    T = typeof(θ[1])
    δ =  zeros(T, (N, size(θ)[1], size(θ)[2]))
    r₊ = zeros(T, (N))
    r₋ = zeros(T, (N) )
    rₘ = zeros(T,N)
    r_hist = zeros(T,num_epochs)
    σ = convert(T, σ)
    α = convert(T, α)

    for i in 1:num_epochs
        for j in 1:N
            δ[j,:,:] = randn(T,size(θ))*σ
            r₊[j] = do_rollout(env, (x)->(θ+δ[j,:,:])*x)
            r₋[j] = do_rollout(env, (x)->(θ-δ[j,:,:])*x)
        end
        
        for j in 1:N
            rₘ[j] = max(r₋[j], r₊[j])
        end

        top = sortperm(rₘ,rev=true)[1:b]

        r_hist[i] = mean(rₘ[top])
        σᵣ = √(std(r₋)^2 + std(r₊)^2) + convert(T, 1e-6)
        
        ∇ = α/(N*σᵣ) * sum((r₊[top] - r₋[top]).*reshape(δ[top,:,:],(b,size(θ)[1]*size(θ)[2])),dims=1)
        θ = θ + reshape(∇,size(θ))
    end

    return θ, r_hist

end


function ars_v2t!(env, θ, μ, Σ; α = .01, N = 32, σ = .02, num_epochs=1000, b=16)
    r₊ = SharedArray{Float64}(N)
    r₋ = SharedArray{Float64}(N)
    rₘ = zeros(Float64,N)
    r_hist = zeros(Float64,num_epochs)
    δ =  SharedArray{Float64}((N, size(θ)[1], size(θ)[2]))

    sp = SharedArray{Float64}((env.observation_space.shape[1], N))
    sm = SharedArray{Float64}((env.observation_space.shape[1], N))
    mp = SharedArray{Float64}((env.observation_space.shape[1], N))
    mm = SharedArray{Float64}((env.observation_space.shape[1], N))

    for i in 1:num_epochs
        @sync @distributed for j in 1:N
            δ[j,:,:] = randn(size(θ))*σ
            r₊[j], mp[:,j], sp[:,j] = do_rollout_with_stats(env, (x)->(θ+δ[j,:,:])*((x - μ)./Σ))
            r₋[j], mm[:,j], sm[:,j] = do_rollout_with_stats(env, (x)->(θ-δ[j,:,:])*((x - μ)./Σ))
        end
        
        for j in 1:N
            rₘ[j] = max(r₋[j], r₊[j])
        end

        top = sortperm(rₘ,rev=true)[1:b]

        r_hist[i] = mean(rₘ[top])
        σᵣ = √(std(r₋)^2 + std(r₊)^2) + 1e-6

        ∇ = α/(b*σᵣ) * sum((r₊[top] - r₋[top]).*reshape(δ[top,:,:],(b,size(θ)[1]*size(θ)[2])),dims=1)
        θ = θ + reshape(∇,size(θ))

        μ = vec(μ*(i-1)*2*N + mean(mp,dims=2)*N + mean(mm,dims=2)*N /(i*2*N))
        Σ = vec(sqrt.((Σ.^2*(i-1)*2*N + mean(sp,dims=2).^2*N + mean(sm,dims=2).^2*N)/(i*2*N)))
        #print(Σ)
    
    end

    return r_hist, θ, μ, Σ

end

end