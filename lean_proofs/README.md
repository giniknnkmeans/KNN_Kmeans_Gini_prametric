# Lean 4 formalization of the Gini-prametric results

Machine-checked proofs associated with *KNN and K-means in Gini Prametric Spaces*. The package covers Definitions 1–2 and Propositions 1–4. Where the paper's convergence arguments require an assumption not established in the paper, the Lean theorem exposes that assumption rather than treating it as automatic.

The project builds without `sorry` or `admit`. `#print axioms` reports no axioms beyond `propext`, `Classical.choice`, and `Quot.sound` for the cataloged declarations. `catalog.json` gives a statement-by-statement map to the paper, exact source and PDF locations, hypotheses, proof status, and known limitations.

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
- `WeightedCentroid.lean` proves the one-dimensional complete-square identity and the fixed-rank weighted least-squares center.
- `KMeansConvergence.lean` follows Proposition 4's fixed-rank route: it derives the rank-weighted normal equation, proves minimization and uniqueness, specializes the result to the paper's arithmetic mean under explicit premises, and proves finite-labeling Lloyd termination from strict descent.
- `KMeansCounterexample.lean` checks whether the constant-rank condition alone makes the arithmetic mean satisfy that normal equation in an exact one-dimensional instance.
- `KnnGini.lean` is the umbrella import.

## Scope and mathematical findings

For Proposition 1, non-negativity follows from monotonicity of each ascending rank profile; nullity, symmetry, translation invariance, and common rank-shift invariance follow by finite-sum algebra. Proposition 2 is the sign-reversed analogue, with antitonicity of the transformed descending-rank score stated explicitly.

For Proposition 3, `coverHart_pointwise_bound` proves the exact finite-class probability inequality. `giniKNN_error_bound_of_pointwise` then proves

$$
R^\* \le R_{\mathrm{NN}} \le R^\*\left(2-\frac{M R^\*}{M-1}\right)
$$

on a finite probability space. Its pointwise hypotheses are the consequences that a Cover–Hart locality argument must provide. The paper does not establish that locality/convergence step for its sample-dependent, non-metric Gini prametric, so the Lean result does not claim that it follows solely from using that prametric.

For Proposition 4, hold the transformed descending-rank gaps $a_{ij}$ constant and write the paper's residual as

$$
r_i(z)=-\sum_j(x_{ij}-z_j)a_{ij}.
$$

The Lean derivation gives the coordinatewise normal equation

$$
\sum_i r_i(\mu)a_{ij}=0.
$$

It proves by completing the square that any center $\mu$ satisfying these equations minimizes the paper's fixed-rank squared objective. Uniqueness additionally requires the rank-gap vectors to separate center displacements. The paper's arithmetic mean is therefore a minimizer under the explicit premise that it satisfies this normal equation; this is the single additional condition in the paper-specific Lean bridge.

`KMeansCounterexample.lean` tests whether constant ranks alone imply that premise under the formalization's reading of the rank vectors. For the one-dimensional cluster $0,1,10$ with $\nu=2$, both candidate centers remain in the same rank cell, while

$$
f(11/6)=485/6<101=f(11/3).
$$

Here $11/3$ is the arithmetic mean. The Lean development proves these values directly from Definition 2, proves that the same residuals instantiate the normal-equation bridge, and checks that the arithmetic mean does not satisfy that equation in this instance. This is a question about whether the formalization matches the rank convention intended in the supplementary proof; it does not show that the algorithm itself fails to converge. Once strict descent of the paper's squared objective is available, `proposition4_converges_of_fixedRank_strictDescent` proves that the update reaches a fixed clustering because the set of labelings is finite.
