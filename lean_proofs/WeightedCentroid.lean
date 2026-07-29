import Mathlib

/-!
# A fixed-rank one-dimensional centroid calculation

Inside a rank cell, the squared generalized Gini objective in one dimension is a
weighted least-squares objective. This file proves its complete-square identity and
identifies its weighted centroid. The paper-specific arithmetic-mean bridge is stated
separately in `KMeansConvergence.lean`.

Catalog item: `kmeans_convergence`.
-/

open scoped BigOperators

namespace KnnGini

variable {ι : Type*} [Fintype ι]

/-- A finite weighted squared-distance objective. For fixed generalized-Gini rank
gaps, the weights are the squares of those gaps. -/
def weightedSqObjective (w x : ι → ℝ) (z : ℝ) : ℝ :=
  ∑ i, w i * (x i - z) ^ 2

/-- Total weight. -/
def totalWeight (w : ι → ℝ) : ℝ :=
  ∑ i, w i

/-- Weighted sum of the observations. -/
def weightedSum (w x : ι → ℝ) : ℝ :=
  ∑ i, w i * x i

/-- Complete-square identity around any point satisfying the weighted normal
equation. -/
theorem weightedSqObjective_completeSquare
    (w x : ι → ℝ) (μ z : ℝ)
    (hμ : weightedSum w x = totalWeight w * μ) :
    weightedSqObjective w x z =
      weightedSqObjective w x μ + totalWeight w * (z - μ) ^ 2 := by
  have hcross : ∑ i, w i * (x i - μ) = 0 := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib]
    simp only [weightedSum, totalWeight] at hμ ⊢
    rw [← Finset.sum_mul]
    linarith
  unfold weightedSqObjective totalWeight
  calc
    ∑ i, w i * (x i - z) ^ 2 =
        ∑ i, (w i * (x i - μ) ^ 2 +
          w i * (z - μ) ^ 2 +
          2 * (μ - z) * (w i * (x i - μ))) := by
            apply Finset.sum_congr rfl
            intro i hi
            ring
    _ = (∑ i, w i * (x i - μ) ^ 2) +
        (∑ i, w i) * (z - μ) ^ 2 := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
          rw [Finset.sum_mul]
          simp [← Finset.mul_sum, hcross]

/-- The weighted center globally minimizes the fixed-rank squared objective when
all weights are nonnegative. -/
theorem weightedCenter_minimizes
    (w x : ι → ℝ) (μ z : ℝ)
    (hw : ∀ i, 0 ≤ w i)
    (hμ : weightedSum w x = totalWeight w * μ) :
    weightedSqObjective w x μ ≤ weightedSqObjective w x z := by
  rw [weightedSqObjective_completeSquare w x μ z hμ]
  exact le_add_of_nonneg_right
    (mul_nonneg (Finset.sum_nonneg fun i hi ↦ hw i) (sq_nonneg (z - μ)))

/-- If the total weight is positive, the weighted center is the unique minimizer. -/
theorem weightedCenter_unique
    (w x : ι → ℝ) (μ z : ℝ)
    (hμ : weightedSum w x = totalWeight w * μ)
    (hweight : 0 < totalWeight w)
    (hobj : weightedSqObjective w x z = weightedSqObjective w x μ) :
    z = μ := by
  rw [weightedSqObjective_completeSquare w x μ z hμ] at hobj
  have hzero : totalWeight w * (z - μ) ^ 2 = 0 := by
    linarith
  have hsquare : (z - μ) ^ 2 = 0 :=
    (mul_eq_zero.mp hzero).resolve_left (ne_of_gt hweight)
  nlinarith

end KnnGini
