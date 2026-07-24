# Lean 4 formalization of the Gini-prametric results

Machine-checked proofs associated with *KNN and K-means in Gini Prametric Spaces*. The package covers Definitions 1–2 and Propositions 1–4. Where the paper's convergence arguments require an assumption not established in the paper, the Lean theorem exposes that assumption rather than treating it as automatic.

The project builds without `sorry` or `admit`. `#print axioms` reports only `propext`, `Classical.choice`, and `Quot.sound` for the headline declarations. `catalog.json` gives a statement-by-statement map to the paper, exact source and PDF locations, hypotheses, proof status, and known limitations.

## Build

Install Lean through `elan`, then run:

```text
lake build
```

The Lean and Mathlib versions are pinned to v4.31.0.

## Files

- `Basic.lean` defines the empirical and generalized Gini prametrics over a finite coordinate type.
- `GiniPrametric.lean` proves all properties listed in Proposition 1.
- `GeneralizedGini.lean` proves all properties listed in Proposition 2.
- `CoverHartBound.lean` proves the pointwise multiclass Cover–Hart inequalities.
- `KNNConvergence.lean` proves the finite weighted averaging step and the Proposition 3 error bound conditional on the pointwise Cover–Hart premises.
- `WeightedCentroid.lean` proves the complete-square identity and the correct weighted least-squares center.
- `KMeansCounterexample.lean` checks an exact fixed-rank counterexample to the arithmetic-mean minimizer step used in the supplementary proof of Proposition 4.
- `KMeansConvergence.lean` proves termination under an explicit natural-valued strict-descent potential.
- `KnnGini.lean` is the umbrella import.

## Scope and mathematical findings

For Proposition 1, non-negativity follows from monotonicity of each ascending rank profile; nullity, symmetry, translation invariance, and common rank-shift invariance follow by finite-sum algebra. Proposition 2 is the sign-reversed analogue, with antitonicity of the transformed descending-rank score stated explicitly.

For Proposition 3, `coverHart_pointwise_bound` proves the exact finite-class probability inequality. `giniKNN_error_bound_of_pointwise` then proves

$$
R^\* \le R_{\mathrm{NN}} \le R^\*\left(2-\frac{M R^\*}{M-1}\right)
$$

on a finite probability space. Its pointwise hypotheses are the consequences that a Cover–Hart locality argument must provide. The paper does not establish that locality/convergence step for its sample-dependent, non-metric Gini prametric, so the Lean result does not claim that it follows solely from using that prametric.

For Proposition 4, the supplementary proof differentiates a sum of squared Gini distances but omits the outer residual factor. The resulting claim that the arithmetic mean uniquely minimizes the fixed-rank objective is not valid. For the one-dimensional cluster $0,1,10$ with $\nu=2$, both candidate centers remain in the same rank cell, while

$$
f(11/6)=485/6<101=f(11/3).
$$

Here $11/3$ is the arithmetic mean. The Lean development proves these values directly from the generalized Gini definition. It also proves the corrected weighted-center minimization theorem and a conditional termination theorem: an update reaches a fixed point when every non-fixed step strictly decreases a natural-valued potential. This identifies a sound sufficient condition without asserting that the paper's original arithmetic-mean argument establishes it.
