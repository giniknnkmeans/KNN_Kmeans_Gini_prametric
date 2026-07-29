import Basic

/-!
# Proposition 4: the fixed-rank descent argument

This file follows the supplementary proof of Proposition 4 in *KNN and K-means
in Gini Prametric Spaces* up to its centroid-minimization step.

When the rank vectors stay constant, write

`gap i j = Rbar(zᵢⱼ)^(ν-1) - Rbar(zⱼ)^(ν-1)`.

The paper's Gini residual for observation `i` and candidate center `z` is then

`-∑ j, (x i j - z j) * gap i j`,

and its within-cluster objective is the sum of the squares of these residuals.
The exact fixed-rank normal equation is

`∑ i, residual(i, μ) * gap i j = 0`

for every coordinate `j`. The complete-square identity below proves that a
center satisfying this equation globally minimizes the fixed-rank objective.
The paper's arithmetic mean therefore gives the required Lloyd center-update
inequality whenever it satisfies this normal equation.

Constant ranks alone do not imply that additional condition; the exact check in
`KMeansCounterexample.lean` asks whether the paper intended a further property
of the rank vectors at this point. The final theorem records the remaining
finite-state Lloyd argument once strict descent has been obtained.

Catalog item: `kmeans_convergence`.
-/

open scoped BigOperators

namespace KnnGini

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- The paper's generalized-Gini residual inside one fixed rank cell.

`gap i j` is the constant transformed descending-rank difference
`Rbar(xᵢⱼ)^(ν-1) - Rbar(zⱼ)^(ν-1)`. -/
def fixedRankGiniResidual (gap : ι → κ → ℝ) (x : ι → κ → ℝ)
    (z : κ → ℝ) (i : ι) : ℝ :=
  ∑ j, (z j - x i j) * gap i j

/-- The sum of squared generalized-Gini residuals used in the supplementary
proof of Proposition 4. -/
def fixedRankGiniSquaredObjective (gap : ι → κ → ℝ)
    (x : ι → κ → ℝ) (z : κ → ℝ) : ℝ :=
  ∑ i, fixedRankGiniResidual gap x z i ^ 2

/-- The exact first-order normal equation for the paper's fixed-rank squared
objective. -/
def FixedRankNormalEquation (gap : ι → κ → ℝ)
    (x : ι → κ → ℝ) (μ : κ → ℝ) : Prop :=
  ∀ j, ∑ i, fixedRankGiniResidual gap x μ i * gap i j = 0

/-- The rank-gap vectors identify every displacement of the center. This is the
exact condition needed for uniqueness: a displacement orthogonal to every
fixed rank-gap vector must vanish. -/
def FixedRankGapsSeparateCenters (gap : ι → κ → ℝ) : Prop :=
  ∀ v : κ → ℝ, (∀ i, ∑ j, v j * gap i j = 0) → v = 0

/-- The paper's arithmetic-mean center for one nonempty or empty finite cluster.
For the empty case this definition is harmless; Proposition 4's update argument
concerns clusters to which points are assigned. -/
noncomputable def paperArithmeticMean (x : ι → κ → ℝ) (j : κ) : ℝ :=
  (∑ i, x i j) / Fintype.card ι

omit [Fintype ι] in
/-- Changing the center from `μ` to `z` adds the rank-weighted displacement to
each fixed-rank Gini residual. -/
theorem fixedRankGiniResidual_changeCenter (gap : ι → κ → ℝ)
    (x : ι → κ → ℝ) (μ z : κ → ℝ) (i : ι) :
    fixedRankGiniResidual gap x z i =
      fixedRankGiniResidual gap x μ i +
        ∑ j, (z j - μ j) * gap i j := by
  unfold fixedRankGiniResidual
  calc
    ∑ j, (z j - x i j) * gap i j =
        ∑ j, ((μ j - x i j) + (z j - μ j)) * gap i j := by
          apply Finset.sum_congr rfl
          intro j hj
          ring
    _ = (∑ j, (μ j - x i j) * gap i j) +
        ∑ j, (z j - μ j) * gap i j := by
          simp_rw [add_mul]
          exact Finset.sum_add_distrib

/-- The cross term in the complete-square expansion vanishes exactly when the
fixed-rank normal equation holds. -/
theorem fixedRankGini_crossTerm_eq_zero (gap : ι → κ → ℝ)
    (x : ι → κ → ℝ) (μ z : κ → ℝ)
    (hμ : FixedRankNormalEquation gap x μ) :
    ∑ i, fixedRankGiniResidual gap x μ i *
      (∑ j, (z j - μ j) * gap i j) = 0 := by
  calc
    ∑ i, fixedRankGiniResidual gap x μ i *
        (∑ j, (z j - μ j) * gap i j) =
        ∑ i, ∑ j, (z j - μ j) *
          (fixedRankGiniResidual gap x μ i * gap i j) := by
            apply Finset.sum_congr rfl
            intro i hi
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro j hj
            ring
    _ = ∑ j, ∑ i, (z j - μ j) *
        (fixedRankGiniResidual gap x μ i * gap i j) := by
          rw [Finset.sum_comm]
    _ = ∑ j, (z j - μ j) *
        (∑ i, fixedRankGiniResidual gap x μ i * gap i j) := by
          apply Finset.sum_congr rfl
          intro j hj
          rw [Finset.mul_sum]
    _ = 0 := by
          apply Finset.sum_eq_zero
          intro j hj
          rw [hμ j, mul_zero]

/-- Complete-square identity for the paper's fixed-rank squared Gini objective.
This is the corrected form of the differentiation step in the supplementary
proof. -/
theorem fixedRankGiniSquaredObjective_completeSquare
    (gap : ι → κ → ℝ) (x : ι → κ → ℝ) (μ z : κ → ℝ)
    (hμ : FixedRankNormalEquation gap x μ) :
    fixedRankGiniSquaredObjective gap x z =
      fixedRankGiniSquaredObjective gap x μ +
        ∑ i, (∑ j, (z j - μ j) * gap i j) ^ 2 := by
  let δ : ι → ℝ := fun i ↦ ∑ j, (z j - μ j) * gap i j
  have hcross : ∑ i, fixedRankGiniResidual gap x μ i * δ i = 0 := by
    simpa [δ] using fixedRankGini_crossTerm_eq_zero gap x μ z hμ
  have hcross_two :
      ∑ i, 2 * fixedRankGiniResidual gap x μ i * δ i = 0 := by
    calc
      ∑ i, 2 * fixedRankGiniResidual gap x μ i * δ i =
          2 * ∑ i, fixedRankGiniResidual gap x μ i * δ i := by
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro i hi
            ring
      _ = 0 := by rw [hcross]; ring
  unfold fixedRankGiniSquaredObjective
  simp_rw [fixedRankGiniResidual_changeCenter gap x μ z]
  change (∑ i, (fixedRankGiniResidual gap x μ i + δ i) ^ 2) =
    (∑ i, fixedRankGiniResidual gap x μ i ^ 2) + ∑ i, δ i ^ 2
  calc
    ∑ i, (fixedRankGiniResidual gap x μ i + δ i) ^ 2 =
        ∑ i, (fixedRankGiniResidual gap x μ i ^ 2 +
          2 * fixedRankGiniResidual gap x μ i * δ i + δ i ^ 2) := by
            apply Finset.sum_congr rfl
            intro i hi
            ring
    _ = (∑ i, fixedRankGiniResidual gap x μ i ^ 2) +
        (∑ i, 2 * fixedRankGiniResidual gap x μ i * δ i) +
        ∑ i, δ i ^ 2 := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
    _ = (∑ i, fixedRankGiniResidual gap x μ i ^ 2) + ∑ i, δ i ^ 2 := by
          rw [hcross_two, add_zero]

/-- A center satisfying the paper's exact fixed-rank normal equation globally
minimizes its squared Gini objective. -/
theorem fixedRankNormalEquation_minimizes (gap : ι → κ → ℝ)
    (x : ι → κ → ℝ) (μ z : κ → ℝ)
    (hμ : FixedRankNormalEquation gap x μ) :
    fixedRankGiniSquaredObjective gap x μ ≤
      fixedRankGiniSquaredObjective gap x z := by
  rw [fixedRankGiniSquaredObjective_completeSquare gap x μ z hμ]
  exact le_add_of_nonneg_right (Finset.sum_nonneg fun i hi ↦ sq_nonneg _)

/-- Under the corresponding rank-gap identifiability condition, the
normal-equation center is the unique minimizer. -/
theorem fixedRankNormalEquation_uniqueMinimizer (gap : ι → κ → ℝ)
    (x : ι → κ → ℝ) (μ z : κ → ℝ)
    (hμ : FixedRankNormalEquation gap x μ)
    (hseparate : FixedRankGapsSeparateCenters gap)
    (hobj : fixedRankGiniSquaredObjective gap x z =
      fixedRankGiniSquaredObjective gap x μ) :
    z = μ := by
  rw [fixedRankGiniSquaredObjective_completeSquare gap x μ z hμ] at hobj
  have hsum :
      ∑ i, (∑ j, (z j - μ j) * gap i j) ^ 2 = 0 := by
    linarith
  have hall :
      ∀ i, ∑ j, (z j - μ j) * gap i j = 0 := by
    intro i
    have hsquares := (Finset.sum_eq_zero_iff_of_nonneg
      (s := Finset.univ)
      (fun k hk ↦ sq_nonneg (∑ j, (z j - μ j) * gap k j))).mp hsum
    have hi := hsquares i (Finset.mem_univ i)
    nlinarith
  have hzero := hseparate (fun j ↦ z j - μ j) hall
  funext j
  have hj := congrFun hzero j
  simp only [Pi.zero_apply] at hj
  linarith

/-- The centroid-update step claimed in Proposition 4, in the paper's notation.
The sole additional premise is exposed exactly where it is needed: the
arithmetic mean must satisfy the fixed-rank normal equation. -/
theorem proposition4_arithmeticMean_minimizes_of_normalEquation
    (gap : ι → κ → ℝ) (x : ι → κ → ℝ) (z : κ → ℝ)
    (hmean : FixedRankNormalEquation gap x (paperArithmeticMean x)) :
    fixedRankGiniSquaredObjective gap x (paperArithmeticMean x) ≤
      fixedRankGiniSquaredObjective gap x z :=
  fixedRankNormalEquation_minimizes gap x (paperArithmeticMean x) z hmean

/-- The paper's claimed uniqueness of the arithmetic-mean update, with the two
precise fixed-rank conditions separated: the mean satisfies the normal equation,
and the rank-gap vectors identify the center. -/
theorem proposition4_arithmeticMean_unique_of_normalEquation
    (gap : ι → κ → ℝ) (x : ι → κ → ℝ) (z : κ → ℝ)
    (hmean : FixedRankNormalEquation gap x (paperArithmeticMean x))
    (hseparate : FixedRankGapsSeparateCenters gap)
    (hobj : fixedRankGiniSquaredObjective gap x z =
      fixedRankGiniSquaredObjective gap x (paperArithmeticMean x)) :
    z = paperArithmeticMean x :=
  fixedRankNormalEquation_uniqueMinimizer gap x (paperArithmeticMean x) z
    hmean hseparate hobj

/-- An iterative update with a strictly decreasing natural-valued potential at
every non-fixed state reaches a fixed point. This helper is retained for uses
where such a discrete potential is already available. -/
theorem lloyd_terminates_of_strictDescent
    {σ : Type*} (step : σ → σ) (potential : σ → ℕ)
    (hdescent : ∀ s, step s ≠ s → potential (step s) < potential s) :
    ∀ s, ∃ n ≤ potential s,
      (step^[n]) s = step ((step^[n]) s) := by
  have aux : ∀ k s, potential s = k → ∃ n ≤ potential s,
      (step^[n]) s = step ((step^[n]) s) := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
      intro s hpot
      by_cases hfixed : step s = s
      · exact ⟨0, Nat.zero_le _, by simpa using hfixed.symm⟩
      · have hlt : potential (step s) < k := by
          simpa [hpot] using hdescent s hfixed
        obtain ⟨n, hn, hfix⟩ := ih (potential (step s)) hlt (step s) rfl
        refine ⟨n + 1, ?_, ?_⟩
        · omega
        · simpa [Function.iterate_succ_apply] using hfix
  intro s
  exact aux (potential s) s rfl

/-- Proposition 4's finite-state Lloyd conclusion once the paper's squared Gini
objective strictly decreases at every non-fixed labeling update.

The state is an actual clustering `Point → Cluster`, rather than an unrelated
abstract state. The remaining premise is precisely the descent statement for
which the supplementary proof invokes arithmetic-mean minimization. -/
theorem proposition4_converges_of_fixedRank_strictDescent
    {Point Cluster : Type*} [Finite Point] [Finite Cluster]
    (giniLloydStep : (Point → Cluster) → (Point → Cluster))
    (paperSquaredObjective : (Point → Cluster) → ℝ)
    (hdescent : ∀ labels, giniLloydStep labels ≠ labels →
      paperSquaredObjective (giniLloydStep labels) <
        paperSquaredObjective labels) :
    ∀ initial, ∃ n,
      (giniLloydStep^[n]) initial =
        giniLloydStep ((giniLloydStep^[n]) initial) := by
  classical
  letI := Fintype.ofFinite Point
  letI := Fintype.ofFinite Cluster
  let r : (Point → Cluster) → (Point → Cluster) → Prop :=
    fun a b ↦ paperSquaredObjective a < paperSquaredObjective b
  letI : Std.Irrefl r := ⟨fun a ↦ lt_irrefl _⟩
  letI : IsTrans (Point → Cluster) r :=
    ⟨fun _ _ _ hab hbc ↦ lt_trans hab hbc⟩
  letI : IsStrictOrder (Point → Cluster) r := {}
  have hwf : (Set.univ : Set (Point → Cluster)).WellFoundedOn r :=
    Set.toFinite Set.univ |>.wellFoundedOn
  letI : IsWellFounded {x // x ∈ (Set.univ : Set (Point → Cluster))}
      (Subrel r (· ∈ (Set.univ : Set (Point → Cluster)))) :=
    ⟨hwf⟩
  intro initial
  obtain ⟨n, hn⟩ := WellFounded.not_rel_apply_succ
    (r := Subrel r (· ∈ (Set.univ : Set (Point → Cluster))))
    (fun m ↦
      ⟨(giniLloydStep^[m]) initial, Set.mem_univ _⟩)
  refine ⟨n, ?_⟩
  by_contra hnotfixed
  apply hn
  change paperSquaredObjective ((giniLloydStep^[n + 1]) initial) <
    paperSquaredObjective ((giniLloydStep^[n]) initial)
  simpa [Function.iterate_succ_apply'] using
    hdescent ((giniLloydStep^[n]) initial) (Ne.symm hnotfixed)

end KnnGini
