import Mathlib

/-!
# Proposition 4: a sound conditional termination theorem

The paper's arithmetic-mean minimization step is not valid for its squared Gini
objective. This file records the standard corrected termination principle: an
iteration reaches a fixed point when every non-fixed update strictly decreases a
natural-valued potential.
-/

namespace KnnGini

/--
An iterative update with a strictly decreasing natural-valued potential at every
non-fixed state reaches a fixed point in at most the initial potential.
-/
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

end KnnGini
