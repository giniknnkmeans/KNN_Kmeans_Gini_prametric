import Basic

/-!
# Proposition 3: the multiclass Cover–Hart algebra

For posterior class probabilities `p`, the local Bayes error is `1 - max p` and
the limiting one-nearest-neighbor error appearing in the Cover–Hart argument is
`1 - ∑ pᵢ²`. This file proves the exact multiclass bounds displayed in
Proposition 3. The separate statistical statement that generalized-Gini nearest
neighbors have this limiting error requires a locality/convergence hypothesis.

Catalog item: `knn_convergence_error_bound`.
-/

open scoped BigOperators

namespace KnnGini

variable {κ : Type*} [Fintype κ]

/-- The limiting local one-nearest-neighbor error in the Cover–Hart argument. -/
def coverHartNNError (p : κ → ℝ) : ℝ :=
  1 - ∑ i, p i ^ 2

/-- The local Bayes error when `m` is a most probable class. -/
def coverHartBayesError (p : κ → ℝ) (m : κ) : ℝ :=
  1 - p m

/-- The Bayes error is no larger than the limiting one-nearest-neighbor error. -/
theorem coverHart_pointwise_lower
    (p : κ → ℝ) (m : κ)
    (hp : ∀ i, 0 ≤ p i)
    (hsum : ∑ i, p i = 1)
    (hmax : ∀ i, p i ≤ p m) :
    coverHartBayesError p m ≤ coverHartNNError p := by
  have hsq : (∑ i, p i ^ 2) ≤ p m := by
    calc
      (∑ i, p i ^ 2) ≤ ∑ i, p m * p i := by
        apply Finset.sum_le_sum
        intro i hi
        nlinarith [hp i, hmax i]
      _ = p m * ∑ i, p i := by rw [Finset.mul_sum]
      _ = p m := by rw [hsum, mul_one]
  unfold coverHartBayesError coverHartNNError
  linarith

/-- Exact multiclass upper bound at one feature value. -/
theorem coverHart_pointwise_upper
    (p : κ → ℝ) (m : κ)
    (hsum : ∑ i, p i = 1)
    (hcard : 2 ≤ Fintype.card κ) :
    coverHartNNError p ≤
      coverHartBayesError p m *
        (2 - (Fintype.card κ : ℝ) * coverHartBayesError p m /
          ((Fintype.card κ : ℝ) - 1)) := by
  classical
  let s : Finset κ := Finset.univ.erase m
  have hm : m ∈ (Finset.univ : Finset κ) := Finset.mem_univ m
  have hcardR : (2 : ℝ) ≤ (Fintype.card κ : ℝ) := by
    exact_mod_cast hcard
  have hden : 0 < (Fintype.card κ : ℝ) - 1 := by linarith
  have hsumRest : ∑ i ∈ s, p i = 1 - p m := by
    have herase := Finset.sum_erase_add (s := (Finset.univ : Finset κ))
      (f := p) hm
    rw [hsum] at herase
    dsimp [s]
    linarith
  have hcauchy :
      (∑ i ∈ s, p i) ^ 2 ≤
        ((Fintype.card κ : ℝ) - 1) * ∑ i ∈ s, p i ^ 2 := by
    have h := sq_sum_le_card_mul_sum_sq (s := s) (f := p)
    have hcardS : s.card = Fintype.card κ - 1 := by
      dsimp [s]
      rw [Finset.card_erase_of_mem hm, Finset.card_univ]
    rw [hcardS] at h
    norm_num [Nat.cast_sub (by omega : 1 ≤ Fintype.card κ)] at h ⊢
    exact h
  have hrest :
      (1 - p m) ^ 2 / ((Fintype.card κ : ℝ) - 1) ≤
        ∑ i ∈ s, p i ^ 2 := by
    apply (div_le_iff₀ hden).2
    rw [← hsumRest]
    simpa [mul_comm] using hcauchy
  have hdecomp :
      (∑ i, p i ^ 2) = p m ^ 2 + ∑ i ∈ s, p i ^ 2 := by
    have h := Finset.sum_erase_add
      (s := (Finset.univ : Finset κ)) (f := fun i ↦ p i ^ 2) hm
    dsimp [s]
    linarith
  have halg :
      1 - (p m ^ 2 + (1 - p m) ^ 2 /
        ((Fintype.card κ : ℝ) - 1)) =
      (1 - p m) *
        (2 - (Fintype.card κ : ℝ) * (1 - p m) /
          ((Fintype.card κ : ℝ) - 1)) := by
    field_simp
    ring
  unfold coverHartNNError coverHartBayesError
  calc
    1 - ∑ i, p i ^ 2 ≤
        1 - (p m ^ 2 + (1 - p m) ^ 2 /
          ((Fintype.card κ : ℝ) - 1)) := by
            rw [hdecomp]
            linarith
    _ = _ := halg

/-- Proposition 3's complete pointwise probability bound. -/
theorem coverHart_pointwise_bound
    (p : κ → ℝ) (m : κ)
    (hp : ∀ i, 0 ≤ p i)
    (hsum : ∑ i, p i = 1)
    (hmax : ∀ i, p i ≤ p m)
    (hcard : 2 ≤ Fintype.card κ) :
    coverHartBayesError p m ≤ coverHartNNError p ∧
      coverHartNNError p ≤
        coverHartBayesError p m *
          (2 - (Fintype.card κ : ℝ) * coverHartBayesError p m /
            ((Fintype.card κ : ℝ) - 1)) :=
  ⟨coverHart_pointwise_lower p m hp hsum hmax,
    coverHart_pointwise_upper p m hsum hcard⟩

end KnnGini
