
# ======== Linear Z  ========
include("seajul.jl")
import .SJ
using Plots
plotlyjs()

env = SJ.LinearZEnv(.01, 50, 10, 25.0)
θ = randn(2,4)

@time θ = SJ.ars!(env, θ, N=32, num_epochs=1000)
@time θ = SJ.ars!(env, θ, N=32, num_epochs=1000)

#θ = zeros(2,4)
#θ[1,4] = 100

# ======== Linear Z eval  ========

let
x = randn(3)*10
r = 10.0
s = [x;r]
X = zeros(env.trial_length,4)

reward = 0.0 
dynamics = (t,s,a)->[a[1],a[2],s[1]]

for t in 1:env.trial_length
    u = θ*s
    
    for _ in 1:env.act_hold
        x = SJ.rk4(dynamics, u, t, env.dt, x)

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
    X[t,:] = s
end

print(reward)
    
plot(X)

# i = 1 
# for var in ("x","y","z","r")
#     println(var)
#     plot!(X[:,i], label=var)
#     i+=1
# end

end