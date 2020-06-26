module PG
    using Flux
    using Flux.Optimise: update!
    using Statistics
    using LinearAlgebra

    export do_rollout!
    
    function sample_action(policy,o)
        out = policy(o)
        n_act = size(out)[1]÷2
        
        means = out[1:n_act]
        logstds = out[n_act+1:end]
        stds = ℯ.^logstds
        
        actions = stds.*randn(Float32, n_act) + means
        #logprob = -.5*((actions - means)./(stds)).^2 - logstds .- log(√(2.f0*π))
        
        return actions#, logprob
    end 

    function get_logp(policy,o,actions)
        out = policy(o)
        n_act = size(out)[1]÷2
        
        means = out[1:n_act,:]
        logstds = out[n_act+1:end,:]
        stds = exp.(logstds)
        
        return -.5f0*((actions - means)./(stds)).^2f0 - logstds .- log(√(2.f0*π))
    end 


    function do_rollout!(policy,obs,acts,rews)
        ep_length = length(rews)
        obs[:,1] = [1.0f0, 1.0f0]
        cur_obs =  obs[:,1]
        dt = .01f0

        A = I#[.1 4; 0 .4]
        B = I

        R = I
        Q = I

        for t = 1:ep_length
            obs[:,t] = cur_obs
            acts[:,t] = sample_action(policy, cur_obs)
            cur_obs = A*obs[:,t] + B*acts[:,t]
            #obs1[:,t+1] = rk4(dynamics, acts[:,t], 0.0f0, dt, obs1[:,t])
            rews[t] = -(cur_obs'*Q*cur_obs +  acts[:,t]'*R*acts[:,t])
        end 
        return
    end

    function learn!(policy, num_epochs; ep_length=10,log_every=1,sgd_epochs=10)
        
        θ = params(policy.layers)
        opt = ADAM(1e-3) 

        acts = zeros(Float32, 2, ep_length)
        obs = zeros(Float32, 2, ep_length)
        rews = zeros(Float32, 1, ep_length)

        loss_hist = zeros(Float32, num_epochs÷log_every)
        rew_hist  = zeros(Float32, num_epochs÷log_every)

        loss(lgp,R) = -mean(sum(lgp,dims=1).*R)
    
        for epoch in range(1,stop=num_epochs)
            do_rollout!(policy,obs,acts,rews)       
            for i = 1:sgd_epochs
                gs = gradient(()->loss(get_logp(policy,obs,acts), cumsum(rews,dims=1)), θ)
                for p in θ
                    update!(opt, p, gs[p])
                end            end

            if epoch % log_every == 0
                loss_hist[epoch÷log_every] = loss(get_logp(policy,obs,acts),cumsum(rews,dims=1))
                rew_hist[epoch÷log_every] = sum(rews)
            end

        end

        return loss_hist, rew_hist
    
    end
end

# ep_length=500
# acts = zeros(Float32, ep_length,2)
# obs1 = zeros(Float32, ep_length,2)
# rews = zeros(Float32, ep_length)


# #@code_llvm PG.do_rollout(pol,obs1,acts,rews)
# @time obs, acts, rews = PG.do_rollout(pol,obs1,acts,rews)
# @time obs, acts, rews = PG.do_rollout(pol,obs1,acts,rews)
#end
