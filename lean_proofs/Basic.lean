import Mathlib

/-!
# Gini prametrics

Definitions 1 and 2 of *KNN and K-means in Gini Prametric Spaces*,
arXiv:2501.18028v3.

Catalog items: `gini_prametric_definition`,
`generalized_gini_prametric_definition`.

The paper's empirical ranks enter the definitions only through their coordinatewise
rank profiles. This file separates that algebraic layer from a particular sorting
implementation: ascending profiles are monotone, while transformed descending-rank
profiles are antitone.
-/

open scoped BigOperators

namespace KnnGini

/-- Definition 1: the Gini prametric associated with coordinatewise ascending-rank
profiles. -/
def giniPrametric {ι : Type*} [Fintype ι]
    (rank : ι → ℝ → ℝ) (x y : ι → ℝ) : ℝ :=
  ∑ j, (x j - y j) * (rank j (x j) - rank j (y j))

/-- Definition 2, with `score j t = descendingRank(j,t)^(ν-1)` already packaged as
the transformed descending-rank score. -/
def generalizedGiniPrametric {ι : Type*} [Fintype ι]
    (score : ι → ℝ → ℝ) (x y : ι → ℝ) : ℝ :=
  -∑ j, (x j - y j) * (score j (x j) - score j (y j))

/--
Definition 2 in the paper's notation: `descendingRank j t` is
`\overline R_{X_j}(t)` and the exponent is `ν - 1`.
-/
noncomputable def generalizedGiniPrametricOfRank {ι : Type*} [Fintype ι]
    (ν : ℝ) (descendingRank : ι → ℝ → ℝ) (x y : ι → ℝ) : ℝ :=
  generalizedGiniPrametric
    (fun j t ↦ Real.rpow (descendingRank j t) (ν - 1)) x y

/-- The paper-notation definition visibly instantiates the abstract score form. -/
theorem generalizedGiniPrametricOfRank_eq {ι : Type*} [Fintype ι]
    (ν : ℝ) (descendingRank : ι → ℝ → ℝ) (x y : ι → ℝ) :
    generalizedGiniPrametricOfRank ν descendingRank x y =
      -∑ j, (x j - y j) *
        (Real.rpow (descendingRank j (x j)) (ν - 1) -
          Real.rpow (descendingRank j (y j)) (ν - 1)) := by
  rfl

/-- Translate every coordinate of a point by the same scalar. -/
def translatePoint {ι : Type*} (c : ℝ) (x : ι → ℝ) : ι → ℝ :=
  fun j ↦ x j + c

/-- Transport a rank profile along a common translation. -/
def translateProfile {ι : Type*} (c : ℝ) (rank : ι → ℝ → ℝ) : ι → ℝ → ℝ :=
  fun j t ↦ rank j (t - c)

/-- Shift every rank value by the same scalar. -/
def shiftProfile {ι : Type*} (a : ℝ) (rank : ι → ℝ → ℝ) : ι → ℝ → ℝ :=
  fun j t ↦ rank j t + a

end KnnGini
