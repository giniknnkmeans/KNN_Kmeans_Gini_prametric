import Basic

/-!
# Proposition 1: properties of the Gini prametric

This file formalizes Proposition 1 of *KNN and K-means in Gini Prametric Spaces*
(Sec. 3.3, PDF page 3). The paper refers to properties of Gini covariance; here the
essential order assumption is stated explicitly: every ascending-rank profile is
monotone.

Catalog item: `gini_prametric_properties`.
-/

open scoped BigOperators

namespace KnnGini

variable {ι : Type*} [Fintype ι]

/-- Proposition 1, Nullity. -/
theorem giniPrametric_self (rank : ι → ℝ → ℝ) (x : ι → ℝ) :
    giniPrametric rank x x = 0 := by
  simp [giniPrametric]

/-- Proposition 1, Rank-Nullity. -/
theorem giniPrametric_eq_zero_of_rank_eq
    (rank : ι → ℝ → ℝ) (x y : ι → ℝ)
    (h : ∀ j, rank j (x j) = rank j (y j)) :
    giniPrametric rank x y = 0 := by
  simp [giniPrametric, h]

/-- Proposition 1, Non-Negativity. -/
theorem giniPrametric_nonneg
    (rank : ι → ℝ → ℝ) (x y : ι → ℝ)
    (hrank : ∀ j, Monotone (rank j)) :
    0 ≤ giniPrametric rank x y := by
  unfold giniPrametric
  refine Finset.sum_nonneg ?_
  intro j hj
  rcases le_total (x j) (y j) with hxy | hyx
  · exact mul_nonneg_of_nonpos_of_nonpos
      (sub_nonpos.mpr hxy) (sub_nonpos.mpr (hrank j hxy))
  · exact mul_nonneg
      (sub_nonneg.mpr hyx) (sub_nonneg.mpr (hrank j hyx))

/-- Proposition 1, Symmetry. -/
theorem giniPrametric_comm
    (rank : ι → ℝ → ℝ) (x y : ι → ℝ) :
    giniPrametric rank x y = giniPrametric rank y x := by
  unfold giniPrametric
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- Proposition 1, Linear Invariance, written as transport of both the data and
their empirical rank profiles by a common translation. -/
theorem giniPrametric_translate
    (rank : ι → ℝ → ℝ) (x y : ι → ℝ) (c : ℝ) :
    giniPrametric (translateProfile c rank)
      (translatePoint c x) (translatePoint c y) =
      giniPrametric rank x y := by
  simp [giniPrametric, translateProfile, translatePoint]

/-- Proposition 1, Rank Invariance: a common additive shift of all rank values
cancels from every rank difference. -/
theorem giniPrametric_shiftProfile
    (rank : ι → ℝ → ℝ) (x y : ι → ℝ) (a : ℝ) :
    giniPrametric (shiftProfile a rank) x y =
      giniPrametric rank x y := by
  simp [giniPrametric, shiftProfile]

end KnnGini
