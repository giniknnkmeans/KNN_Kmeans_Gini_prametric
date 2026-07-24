import Basic

/-!
# Proposition 2: properties of the generalized Gini prametric

This file formalizes Proposition 2 of *KNN and K-means in Gini Prametric Spaces*
(Sec. 3.4, PDF page 4). The transformed descending-rank score
`descendingRank^(ν-1)` is represented by `score`; its required antitonicity is
made explicit.
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

end KnnGini
