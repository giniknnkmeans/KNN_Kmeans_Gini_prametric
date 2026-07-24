import CoverHartBound

/-!
# Proposition 3: finite averaging

This file lifts the pointwise Cover–Hart algebra to a finite probability space.
The statistical identification of the limiting generalized-Gini nearest-neighbor
error remains an explicit premise of any application.

Catalog item: `knn_convergence_error_bound`.
-/

open scoped BigOperators

namespace KnnGini

variable {Ω : Type*} [Fintype Ω]

/-- Expectation on a finite space with weights `μ`. -/
def finiteAverage (μ f : Ω → ℝ) : ℝ :=
  ∑ x, μ x * f x

/-- The second moment dominates the square of the mean. -/
theorem finiteAverage_sq_le_average_sq
    (μ f : Ω → ℝ)
    (hμ : ∀ x, 0 ≤ μ x)
    (hsum : ∑ x, μ x = 1) :
    finiteAverage μ f ^ 2 ≤ finiteAverage μ (fun x ↦ f x ^ 2) := by
  have hvar : 0 ≤ ∑ x, μ x * (f x - finiteAverage μ f) ^ 2 := by
    exact Finset.sum_nonneg fun x _ ↦ mul_nonneg (hμ x) (sq_nonneg _)
  have hid :
      (∑ x, μ x * (f x - finiteAverage μ f) ^ 2) =
        finiteAverage μ (fun x ↦ f x ^ 2) - finiteAverage μ f ^ 2 := by
    unfold finiteAverage
    calc
      (∑ x, μ x * (f x - ∑ y, μ y * f y) ^ 2) =
          ∑ x, (μ x * f x ^ 2 -
            2 * (∑ y, μ y * f y) * (μ x * f x) +
            μ x * (∑ y, μ y * f y) ^ 2) := by
              apply Finset.sum_congr rfl
              intro x hx
              ring
      _ = (∑ x, μ x * f x ^ 2) -
            2 * (∑ y, μ y * f y) * (∑ x, μ x * f x) +
            (∑ x, μ x) * (∑ y, μ y * f y) ^ 2 := by
              rw [Finset.sum_add_distrib, Finset.sum_sub_distrib,
                ← Finset.mul_sum, ← Finset.sum_mul]
      _ = (∑ x, μ x * f x ^ 2) - (∑ x, μ x * f x) ^ 2 := by
              rw [hsum]
              ring
  rw [hid] at hvar
  linarith

/-- Averaging a pointwise concave-quadratic bound preserves the same bound. -/
theorem finiteAverage_concaveQuadratic
    (μ error nnError : Ω → ℝ) (c : ℝ)
    (hμ : ∀ x, 0 ≤ μ x)
    (hsum : ∑ x, μ x = 1)
    (hc : 0 ≤ c)
    (hpoint : ∀ x, nnError x ≤ error x * (2 - c * error x)) :
    finiteAverage μ nnError ≤
      finiteAverage μ error * (2 - c * finiteAverage μ error) := by
  have havgPoint :
      finiteAverage μ nnError ≤
        finiteAverage μ (fun x ↦ error x * (2 - c * error x)) := by
    apply Finset.sum_le_sum
    intro x hx
    exact mul_le_mul_of_nonneg_left (hpoint x) (hμ x)
  have hsecond := finiteAverage_sq_le_average_sq μ error hμ hsum
  have hmul := mul_le_mul_of_nonneg_left hsecond hc
  have hquad :
      finiteAverage μ (fun x ↦ error x * (2 - c * error x)) =
        2 * finiteAverage μ error -
          c * finiteAverage μ (fun x ↦ error x ^ 2) := by
    unfold finiteAverage
    calc
      (∑ x, μ x * (error x * (2 - c * error x))) =
          ∑ x, (2 * (μ x * error x) - c * (μ x * error x ^ 2)) := by
            apply Finset.sum_congr rfl
            intro x hx
            ring
      _ = 2 * (∑ x, μ x * error x) -
            c * (∑ x, μ x * error x ^ 2) := by
              rw [Finset.sum_sub_distrib, Finset.mul_sum, Finset.mul_sum]
  rw [hquad] at havgPoint
  calc
    finiteAverage μ nnError ≤
        2 * finiteAverage μ error -
          c * finiteAverage μ (fun x ↦ error x ^ 2) := havgPoint
    _ ≤ 2 * finiteAverage μ error - c * finiteAverage μ error ^ 2 := by
      linarith
    _ = finiteAverage μ error * (2 - c * finiteAverage μ error) := by
      ring

/--
Finite-space form of Proposition 3.

The two pointwise assumptions are precisely what the Cover–Hart locality argument
must supply for a proposed nearest-neighbor rule. In particular, this theorem
does not assume that they follow merely because a prametric was used.
-/
theorem giniKNN_error_bound_of_pointwise
    (μ bayesError nnError : Ω → ℝ) (M : ℕ)
    (hμ : ∀ x, 0 ≤ μ x)
    (hsum : ∑ x, μ x = 1)
    (hM : 2 ≤ M)
    (hlower : ∀ x, bayesError x ≤ nnError x)
    (hupper : ∀ x, nnError x ≤
      bayesError x * (2 - (M : ℝ) * bayesError x / ((M : ℝ) - 1))) :
    finiteAverage μ bayesError ≤ finiteAverage μ nnError ∧
      finiteAverage μ nnError ≤
        finiteAverage μ bayesError *
          (2 - (M : ℝ) * finiteAverage μ bayesError / ((M : ℝ) - 1)) := by
  have hMR : (2 : ℝ) ≤ (M : ℝ) := by exact_mod_cast hM
  have hden : 0 < (M : ℝ) - 1 := by linarith
  constructor
  · unfold finiteAverage
    apply Finset.sum_le_sum
    intro x hx
    exact mul_le_mul_of_nonneg_left (hlower x) (hμ x)
  · have hc : 0 ≤ (M : ℝ) / ((M : ℝ) - 1) :=
      div_nonneg (by positivity) hden.le
    have hpoint : ∀ x, nnError x ≤
        bayesError x * (2 - ((M : ℝ) / ((M : ℝ) - 1)) * bayesError x) := by
      intro x
      simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hupper x
    have h := finiteAverage_concaveQuadratic μ bayesError nnError
      ((M : ℝ) / ((M : ℝ) - 1)) hμ hsum hc hpoint
    simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using h

/--
The finite global Cover–Hart bound obtained directly from posterior class
probabilities. This theorem connects the pointwise algebra in
`CoverHartBound.lean` to the averaging theorem above.
-/
theorem coverHart_finite_global_bound
    {κ : Type*} [Fintype κ]
    (μ : Ω → ℝ) (p : Ω → κ → ℝ) (m : Ω → κ)
    (hμ : ∀ x, 0 ≤ μ x)
    (hsumμ : ∑ x, μ x = 1)
    (hp : ∀ x i, 0 ≤ p x i)
    (hsump : ∀ x, ∑ i, p x i = 1)
    (hmax : ∀ x i, p x i ≤ p x (m x))
    (hcard : 2 ≤ Fintype.card κ) :
    finiteAverage μ (fun x ↦ coverHartBayesError (p x) (m x)) ≤
        finiteAverage μ (fun x ↦ coverHartNNError (p x)) ∧
      finiteAverage μ (fun x ↦ coverHartNNError (p x)) ≤
        finiteAverage μ (fun x ↦ coverHartBayesError (p x) (m x)) *
          (2 - (Fintype.card κ : ℝ) *
            finiteAverage μ (fun x ↦ coverHartBayesError (p x) (m x)) /
              ((Fintype.card κ : ℝ) - 1)) := by
  apply giniKNN_error_bound_of_pointwise
    μ
    (fun x ↦ coverHartBayesError (p x) (m x))
    (fun x ↦ coverHartNNError (p x))
    (Fintype.card κ) hμ hsumμ hcard
  · intro x
    exact (coverHart_pointwise_bound (p x) (m x) (hp x) (hsump x)
      (hmax x) hcard).1
  · intro x
    exact (coverHart_pointwise_bound (p x) (m x) (hp x) (hsump x)
      (hmax x) hcard).2

end KnnGini
