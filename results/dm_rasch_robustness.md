# DM Robustness Under Proficiency Noise: A Rasch Model Argument

## Observation

In the sensitivity analysis (attempts dataset, σ ∈ {0.0, 0.1, 0.2}), the DM estimator
is nearly unaffected by Gaussian noise added to proficiency θ, while SNIPS and DR
degrade moderately and IPS becomes unreliable. This robustness has a clean theoretical
explanation rooted in the structure of the Rasch model.

## The Rasch Model Is Translation-Invariant on (θ, δ)

The 1PL IRT (Rasch) reward model is:

    P(correct | θ, δ) = σ(θ − δ)

The reward depends **only on the difference** (θ − δ), not on θ and δ individually.
The (θ, δ) space has a translational symmetry along the diagonal: shifting both by the
same constant leaves the reward unchanged.

    σ((θ + c) − (δ + c)) = σ(θ − δ)   for any c ∈ ℝ

## Why DM Inherits This Symmetry

The DM estimator evaluates:

    DM(π) = E_{δ ~ π(·|θ_noisy)} [ f(θ_noisy, δ) ]

where f is the learned reward model and θ_noisy = θ + ε, ε ~ N(0, σ_noise).

The learned policy π is a Gaussian policy whose mean is a polynomial in θ. Since it was
trained on the same noisy θ, it adapts its recommended difficulty by approximately the
same shift ε:

    π(δ | θ + ε)  ≈  π(δ − ε | θ)     (policy shifts its action by ε)

So the policy picks δ' ≈ δ* + ε, and the reward becomes:

    f(θ + ε, δ') = σ((θ + ε) − (δ* + ε)) = σ(θ − δ*)

The noise cancels. DM is invariant to the noise because the Rasch symmetry propagates
consistently through both the reward model and the policy — both were trained on the same
noisy θ, so they shift coherently.

## Why IPS/SNIPS Do Not Have This Property

Importance weights are:

    w = π_learned(δ | θ_noisy) / π_behavior(δ | θ_noisy)

Both numerator and denominator shift when θ is noisy, but not in a self-correcting way.
The behavior policy π_behavior is re-fitted to noisy data, changing its estimated variance
and mean in ways that depend on the noise realization. The ratio w is sensitive to the
absolute shape of π_behavior, which does not respect the (θ − δ) symmetry.
As a result, noisy θ corrupts propensity estimates and destabilizes IPS/SNIPS.

## Empirical Confirmation

| Estimator | σ=0.0 (best learned) | σ=0.1 | σ=0.2 | Δ (0→0.2) |
|-----------|----------------------|-------|-------|-----------|
| IPS       | 3.125                | 3.169 | 3.214 | +0.089 (inflated artifact) |
| SNIPS     | 2.616                | 2.476 | 2.392 | −0.224 |
| DR        | 2.637                | 2.492 | 2.432 | −0.205 |
| MIS       | 3.439                | 3.211 | 3.260 | −0.179 |
| DM        | 2.346                | 2.351 | 2.376 | +0.030 (negligible) |

IPS artificially increases because noisy θ makes the behavior policy appear more diffuse,
inflating importance weights — a known pathology when the propensity model is misspecified.
DM is essentially flat, consistent with the theoretical argument above.

## Implication

In IRT-based educational settings where the reward model is grounded in the Rasch model,
DM is the preferred evaluator when proficiency estimates are uncertain or noisy. The
translational symmetry of the Rasch model guarantees that consistent noise on θ (shared
between the reward model and policy training) does not bias DM evaluation.

This is a strong argument for using DM as the primary estimator in practice, where KTM
proficiency estimates always carry some estimation error.
