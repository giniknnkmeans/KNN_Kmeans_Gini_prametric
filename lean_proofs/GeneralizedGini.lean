import Basic

/-!
# Proposition 2: properties of the generalized Gini prametric

This file formalizes Proposition 2 of *KNN and K-means in Gini Prametric Spaces*
(Sec. 3.4, PDF page 4). The transformed descending-rank score
`descendingRank^(ν-1)` is represented by `score`; its required antitonicity is
made explicit.

Catalog item: `generalized_gini_prametric_properties`.
-/

open scoped BigOperators

namespace KnnGini

variable {ι : Type*} [Fintype ι]

/-- Proposition 2, Nullity. -/
theorem generalizedGiniPrametric_self
    (score : ι → ℝ → ℝ) (x : ι → ℝ) :
    generalizedGiniPrametric score x x = 0 := by
  simp [generalizedGiniPrametric]

/-- Proposition 2, Rank-Nullity. -/
theorem generalizedGiniPrametric_eq_zero_of_score_eq
    (score : ι → ℝ → ℝ) (x y : ι → ℝ)
    (h : ∀ j, score j (x j) = score j (y j)) :
    generalizedGiniPrametric score x y = 0 := by
  simp [generalizedGiniPrametric, h]

/-- Proposition 2, Non-Negativity. -/
theorem generalizedGiniPrametric_nonneg
    (score : ι → ℝ → ℝ) (x y : ι → ℝ)
    (hscore : ∀ j, Antitone (score j)) :
    0 ≤ generalizedGiniPrametric score x y := by
  unfold generalizedGiniPrametric
  rw [neg_nonneg]
  refine Finset.sum_nonpos ?_
  intro j hj
  rcases le_total (x j) (y j) with hxy | hyx
  · exact mul_nonpos_of_nonpos_of_nonneg
      (sub_nonpos.mpr hxy) (sub_nonneg.mpr (hscore j hxy))
  · exact mul_nonpos_of_nonneg_of_nonpos
      (sub_nonneg.mpr hyx) (sub_nonpos.mpr (hscore j hyx))

/-- Proposition 2, Symmetry. -/
theorem generalizedGiniPrametric_comm
    (score : ι → ℝ → ℝ) (x y : ι → ℝ) :
    generalizedGiniPrametric score x y =
      generalizedGiniPrametric score y x := by
  unfold generalizedGiniPrametric
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- Proposition 2, Linear Invariance. -/
theorem generalizedGiniPrametric_translate
    (score : ι → ℝ → ℝ) (x y : ι → ℝ) (c : ℝ) :
    generalizedGiniPrametric (translateProfile c score)
      (translatePoint c x) (translatePoint c y) =
      generalizedGiniPrametric score x y := by
  simp [generalizedGiniPrametric, translateProfile, translatePoint]

/--
For `ν > 1`, taking the `(ν - 1)`-st real power preserves the antitonicity of
nonnegative descending ranks.
-/
theorem poweredDescendingRank_antitone
    {κ : Type*} (ν : ℝ) (descendingRank : κ → ℝ → ℝ)
    (hν : 1 < ν)
    (hrank : ∀ j, Antitone (descendingRank j))
    (hnonneg : ∀ j t, 0 ≤ descendingRank j t) :
    ∀ j, Antitone (fun t ↦ Real.rpow (descendingRank j t) (ν - 1)) := by
  intro j a b hab
  exact Real.rpow_le_rpow (hnonneg j b) (hrank j hab)
    (sub_nonneg.mpr hν.le)

/--
Proposition 2, Non-Negativity, instantiated with Definition 2's descending ranks
and real exponent `ν - 1`.
-/
theorem generalizedGiniPrametricOfRank_nonneg
    (ν : ℝ) (descendingRank : ι → ℝ → ℝ) (x y : ι → ℝ)
    (hν : 1 < ν)
    (hrank : ∀ j, Antitone (descendingRank j))
    (hnonneg : ∀ j t, 0 ≤ descendingRank j t) :
    0 ≤ generalizedGiniPrametricOfRank ν descendingRank x y := by
  exact generalizedGiniPrametric_nonneg
    (fun j t ↦ Real.rpow (descendingRank j t) (ν - 1)) x y
    (poweredDescendingRank_antitone ν descendingRank hν hrank hnonneg)

end KnnGini
