# # Process Decomposition
#
# This example demonstrates perhaps the simplest GPPP example, in which the posterior over a
# GP which is given by the sum of two other GPs is decomposed.

# ## Preamble

using AbstractGPs
using Plots
using Random
using Stheno



# ## Define our model

# Define a distribution over f₁, f₂, and f₃, where f₃(x) = f₁(x) + f₂(x).
f = @gppp let
    f1 = GP(randn(), SEKernel())
    f2 = GP(SEKernel())
    f3 = f1 + f2
end;

# Randomly sample `N₁` and `N₃` locations at which to observe `f₁` and `f₃` respectively.
rng, N1, N3 = MersenneTwister(123546), 10, 11;
x1 = GPPPInput(:f1, sort(rand(rng, N1) * 10));
x3 = GPPPInput(:f3, sort(rand(rng, N3) * 10));
x = BlockData(x1, x3);

# Generate some toy of `f1` and `f3`, `y1` and `y3` respectively.
fx = f(x);
y = rand(rng, f(x));
y1, y3 = split(x, y);

# Compute the posterior processes.
f_post = posterior(fx, y);

# Define some plotting stuff.
Np, S = 500, 25;
xp_ = range(-2.5, stop=12.5, length=Np);
xp = BlockData(GPPPInput(:f1, xp_), GPPPInput(:f2, xp_), GPPPInput(:f3, xp_));

# Sample jointly from the posterior over each process.
f_samples = rand(rng, f_post(xp, 1e-9), S);
f1_post_xp, f2_post_xp, f3_post_xp = split(xp, f_samples);

# Compute posterior marginals.
ms = marginals(f_post(xp, 1e-9));
f1_post_m, f2_post_m, f3_post_m = split(xp, mean.(ms));
f1_post_s, f2_post_s, f3_post_s = split(xp, std.(ms));



# ## Plot results

gr();
plt = plot();

# Plot posterior over f1.
plot!(plt, xp_, f1_post_m; ribbon=3 * f1_post_s, color=:red, label="f1", fillalpha=0.3);
plot!(plt, xp_, f1_post_xp; color=:red, label="", alpha=0.2, linewidth=1);
scatter!(plt, x1.x, y1;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);

# Plot posterior over f2.
plot!(plt, xp_, f2_post_m; ribbon=3 * f2_post_s, color=:green, label="f2", fillalpha=0.3);
plot!(plt, xp_, f2_post_xp; color=:green, label="", alpha=0.2, linewidth=1);

# Plot posterior over f3
plot!(plt, xp_, f3_post_m; ribbon=3 * f3_post_s, color=:blue, label="f3", fillalpha=0.3);
plot!(plt, xp_, f3_post_xp; color=:blue, label="", alpha=0.2, linewidth=1);
scatter!(plt, x3.x, y3;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);
plt
