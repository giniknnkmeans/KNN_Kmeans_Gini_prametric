import KMeansConvergence

/-!
# Proposition 4: fixed-rank arithmetic-mean diagnostic

The supplementary proof of Proposition 4 uses the arithmetic mean as the unique
minimizer of the sum of squared generalized Gini distances when ranks remain constant.
This file checks that step under the rank convention used by the formalization.

For data `0, 1, 10` and a center `z ∈ (1,10)`, the descending ranks of
`0, 1, z, 10` are `4, 3, 2, 1`. At `ν = 2`, the squared objective is
`(2z)^2 + (z-1)^2 + (10-z)^2`. Its minimizer `11/6` lies in the same
rank cell as the arithmetic mean `11/3`, but has strictly smaller objective. Thus,
under this interpretation, constant ranks alone do not imply the normal equation
needed by the paper's arithmetic-mean update.

Catalog item: `kmeans_arithmetic_mean_counterexample`.
-/

namespace KnnGini

/-- Descending ranks of the ordered four-point configuration `0 < 1 < z < 10`.
The open interval `(1,10)` is one fixed-rank cell for the candidate center. -/
noncomputable def counterexampleDescendingRank (t : ℝ) : ℝ :=
  if t < 1 then 4
  else if t = 1 then 3
  else if t < 10 then 2
  else 1

/-- The one-coordinate generalized Gini prametric used in the counterexample. -/
noncomputable def counterexampleDistance (x z : ℝ) : ℝ :=
  generalizedGiniPrametric
    (fun _ : Unit ↦ counterexampleDescendingRank)
    (fun _ : Unit ↦ x) (fun _ : Unit ↦ z)

theorem counterexampleDescendingRank_center
    {z : ℝ} (h1 : 1 < z) (h10 : z < 10) :
    counterexampleDescendingRank z = 2 := by
  simp [counterexampleDescendingRank, not_lt.mpr (le_of_lt h1),
    ne_of_gt h1, h10]

theorem counterexampleDistance_zero
    {z : ℝ} (h1 : 1 < z) (h10 : z < 10) :
    counterexampleDistance 0 z = 2 * z := by
  have hznot : ¬z < 1 := not_lt.mpr (le_of_lt h1)
  have hzne : z ≠ 1 := ne_of_gt h1
  simp [counterexampleDistance, generalizedGiniPrametric,
    counterexampleDescendingRank, hznot, hzne, h10]
  ring

theorem counterexampleDistance_one
    {z : ℝ} (h1 : 1 < z) (h10 : z < 10) :
    counterexampleDistance 1 z = z - 1 := by
  have hznot : ¬z < 1 := not_lt.mpr (le_of_lt h1)
  have hzne : z ≠ 1 := ne_of_gt h1
  simp [counterexampleDistance, generalizedGiniPrametric,
    counterexampleDescendingRank, hznot, hzne, h10]
  ring

theorem counterexampleDistance_ten
    {z : ℝ} (h1 : 1 < z) (h10 : z < 10) :
    counterexampleDistance 10 z = 10 - z := by
  have hznot : ¬z < 1 := not_lt.mpr (le_of_lt h1)
  have hzne : z ≠ 1 := ne_of_gt h1
  simp [counterexampleDistance, generalizedGiniPrametric,
    counterexampleDescendingRank, hznot, hzne, h10]
  ring

/-- The squared within-cluster objective used in the supplementary proof. -/
noncomputable def counterexampleSquaredObjective (z : ℝ) : ℝ :=
  counterexampleDistance 0 z ^ 2 +
  counterexampleDistance 1 z ^ 2 +
  counterexampleDistance 10 z ^ 2

theorem counterexampleSquaredObjective_eq
    {z : ℝ} (h1 : 1 < z) (h10 : z < 10) :
    counterexampleSquaredObjective z = 6 * z ^ 2 - 22 * z + 101 := by
  rw [counterexampleSquaredObjective, counterexampleDistance_zero h1 h10,
    counterexampleDistance_one h1 h10, counterexampleDistance_ten h1 h10]
  ring

/-- The arithmetic mean of `0,1,10` is `11/3`. -/
theorem counterexample_arithmeticMean :
    ((0 : ℝ) + 1 + 10) / 3 = 11 / 3 := by
  norm_num

/-- The exact objective value at the weighted least-squares minimizer. -/
theorem counterexample_objective_at_weightedCenter :
    counterexampleSquaredObjective (11 / 6 : ℝ) = 485 / 6 := by
  rw [counterexampleSquaredObjective_eq (by norm_num) (by norm_num)]
  norm_num

/-- The exact objective value at the arithmetic mean claimed by the paper. -/
theorem counterexample_objective_at_arithmeticMean :
    counterexampleSquaredObjective (11 / 3 : ℝ) = 101 := by
  rw [counterexampleSquaredObjective_eq (by norm_num) (by norm_num)]
  norm_num

/-- Under the formalization's rank convention, the arithmetic mean is not a
minimizer of this fixed-rank squared generalized Gini objective. -/
theorem proposition4_arithmeticMean_not_minimizer :
    counterexampleSquaredObjective (11 / 6 : ℝ) <
      counterexampleSquaredObjective (11 / 3 : ℝ) := by
  rw [counterexample_objective_at_weightedCenter,
    counterexample_objective_at_arithmeticMean]
  norm_num

/-- The three observations in the counterexample, written as one-coordinate
points for the fixed-rank Proposition 4 bridge. -/
def counterexamplePoints (i : Fin 3) (_ : Unit) : ℝ :=
  match i.val with
  | 0 => 0
  | 1 => 1
  | _ => 10

/-- The transformed descending-rank gaps relative to a center in `(1,10)`:
`4-2`, `3-2`, and `1-2`. -/
def counterexampleRankGaps (i : Fin 3) (_ : Unit) : ℝ :=
  match i.val with
  | 0 => 2
  | 1 => 1
  | _ => -1

/-- The paper's arithmetic-mean definition specializes to `11/3` for the
counterexample cluster. -/
theorem counterexample_paperArithmeticMean :
    paperArithmeticMean counterexamplePoints () = 11 / 3 := by
  norm_num [paperArithmeticMean, counterexamplePoints, Fin.sum_univ_succ]

/-- In the fixed rank cell `(1,10)`, the residuals in the paper-specific bridge
are exactly the generalized Gini distances already checked above. -/
theorem counterexample_fixedRankResidual_eq_distance
    {z : ℝ} (h1 : 1 < z) (h10 : z < 10) (i : Fin 3) :
    fixedRankGiniResidual counterexampleRankGaps counterexamplePoints
        (fun _ ↦ z) i =
      counterexampleDistance (counterexamplePoints i ()) z := by
  have hznot : ¬z < 1 := not_lt.mpr (le_of_lt h1)
  have hzne : z ≠ 1 := ne_of_gt h1
  have hzne0 : z ≠ 0 := ne_of_gt (lt_trans (by norm_num) h1)
  fin_cases i <;>
    norm_num [fixedRankGiniResidual, counterexampleRankGaps,
      counterexamplePoints, counterexampleDistance,
      generalizedGiniPrametric, counterexampleDescendingRank,
      hznot, hzne, hzne0, h10]

/-- Constant ranks do not by themselves force the paper's arithmetic mean to
satisfy the exact fixed-rank normal equation. This is the precise additional
condition exposed by `proposition4_arithmeticMean_minimizes_of_normalEquation`.
-/
theorem counterexample_arithmeticMean_not_normalEquation :
    ¬FixedRankNormalEquation counterexampleRankGaps counterexamplePoints
      (paperArithmeticMean counterexamplePoints) := by
  intro h
  have hcoord := h ()
  norm_num [fixedRankGiniResidual, counterexampleRankGaps,
    counterexamplePoints, paperArithmeticMean, Fin.sum_univ_succ] at hcoord

end KnnGini
