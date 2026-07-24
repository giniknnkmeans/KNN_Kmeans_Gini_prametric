import Basic

/-!
# Proposition 4: fixed-rank arithmetic-mean counterexample

The supplementary proof of Proposition 4 claims that the arithmetic mean uniquely
minimizes the sum of squared generalized Gini distances when ranks remain constant.
This file checks an exact one-dimensional counterexample in the paper's notation.

For data `0, 1, 10` and a center `z ∈ (1,10)`, the descending ranks of
`0, 1, z, 10` are `4, 3, 2, 1`. At `ν = 2`, the squared objective is
`(2z)^2 + (z-1)^2 + (10-z)^2`. Its minimizer `11/6` lies in the same
rank cell as the arithmetic mean `11/3`, but has strictly smaller objective.
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

/-- The arithmetic mean is not a minimizer of the fixed-rank squared generalized
Gini objective used in the supplementary proof of Proposition 4. -/
theorem proposition4_arithmeticMean_not_minimizer :
    counterexampleSquaredObjective (11 / 6 : ℝ) <
      counterexampleSquaredObjective (11 / 3 : ℝ) := by
  rw [counterexample_objective_at_weightedCenter,
    counterexample_objective_at_arithmeticMean]
  norm_num

end KnnGini
