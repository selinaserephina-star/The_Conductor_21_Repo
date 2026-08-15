# K-modulus ledger extended: void-pressure Hankel to m = 17, Li turnover to n ≈ 10

**Merkabit Research · 2026-07-11 · Stenberg + Claude · Status: [C]**
Extends the m = 12 ledger with the five new certified zeros γ₁₃–γ₁₇ (the pre-registered
batch: γ₁₃/₁₄/₁₅ HIT, γ₁₆ NEAR, γ₁₇ bonus [R-110–114]). Same corner-scanned mpmath
pipeline (dps 60) as the m = 11/12 notes; **cross-checked against them before extending**
(H₇ at 11 zeros 6.891…6.942e−27 vs note 6.884…6.935e−27; Li at 11 zeros
l₁/l₅/l₉ = 2.869/29.92/32.04, exact). Not RH/GRH — the in-range margin ledger only.

## Setup
Atoms 1/γ² at the certified spectrum — now **17 zeros to t = 4.888** (8 audited at
14 digits `chi8_zeros14_list.txt` + γ₉–γ₁₇ at certified brackets). Power sums
S_k = Σ 1/γ^{2k}; Stieltjes–Hankel H_m = det[S_{i+j−1}]. Every determinant evaluated at
all **2⁹ = 512 bracket-corner combinations** of γ₉–γ₁₇.

## Results — the Hankel ledger

| m | H_m (bracket min…max) | spread | with density tail | log₁₀H_m/m² |
|---|---|---|---|---|
| 7  | 2.564…2.586 e−25  | 0.8%  | 7.6e−23   | −0.502 |
| 9  | 5.662…5.757 e−51  | 1.7%  | 1.6e−46   | −0.620 |
| 11 | 1.748…1.802 e−87  | 3.0%  | 6.6e−80   | −0.717 |
| 12 | 2.922…3.051 e−110 | 4.2%  | 1.6e−100  | −0.760 |
| 13 | 1.757…1.856 e−136 | 5.4%  | 5.4e−124  | −0.803 |
| 14 | 3.284…3.664 e−166 | 10.4% | 2.2e−150  | −0.844 |
| 15 | 1.197…1.369 e−199 | 12.6% | 7.1e−180  | −0.884 |
| 16 | 1.282…1.713 e−237 | 25.1% | 1.9e−212  | −0.925 |
| 17 | 4.974…7.889 e−280 | 36.9% | 3.6e−248  | −0.966 |

- **Verified depth extends m = 12 → m = 17.** Every H_m is **positive at all 512
  corners** through m = 17 — the imprecision of the new zeros does not endanger any
  entry. The **density-tail-corrected column** (atoms + passport density-law tail past
  4.888, no longer finitely atomic — the NON-automatic positivity content) is likewise
  positive through m = 17.
- **Honest note on precision.** Bracket spreads stay ≤ 5% through m = 13 but grow to
  **37% at m = 17**, driven by the widest certified bracket — γ₁₄'s pass-1 sign-change
  interval [4.357, 4.379] (±0.011). Positivity is robust (min > 0 at every corner); the
  *value* precision at high m is bracket-limited. Tightening γ₁₄'s bracket (its midpoint
  crossing is ≈ 4.367) would restore the ≤ 7% grade throughout.
- **Collapse profile — a reconciliation.** log₁₀H_m/m² continues its monotone drift, now
  reaching **−0.966 at m = 17**. The m = 11/12 notes' extrapolated "≈ −0.8 asymptote" was
  premature: with depth the profile heads toward the original 06-25 eight-zero rate fit
  **c ≈ 0.99** (det_m ~ 10^{−c·m²}). The intermediate −0.77/−0.80 and the early c ≈ 0.99
  are the same curve seen at different depths — the extension closes that gap.

## Li coefficients (zero-side partial sums, ±pairs)

| n | 12 zeros | 17 zeros |
|---|---|---|
| 1  | 2.93  | 3.18  |
| 5  | 31.3  | 36.83 |
| 9  | 35.31 | 49.37 |
| 10 | —     | **50.81 (peak)** |
| 12 | 29.4  | 48.12 |
| 14 | 23.31 | 43.11 |
| 18 | 18.91 | 35.37 |

The truncation turnover (partials peaking then falling — the signature of missing zeros)
moves from n ≈ 9–10 (12 zeros) to **n ≈ 10** (17 zeros), peak partial ≈ 51 (was ≈ 35);
five new zeros lift both the peak and the trustworthy stretch. Partials remain lower
bounds converging from below toward the prime-side λ₁…λ₁₀ = 4.3…155.

## Registry line
**K-modulus / void-pressure margin ledger now runs to m = 17 at certified-bracket
grade** (positive at all 512 corners; density-tail-corrected positive through m = 17);
collapse constant reconciled to c ≈ 0.99; Li turnover pushed to n ≈ 10. Every future
certified zero extends both tables by the same corner-scanned pipeline.

Not claimed: RH, GRH — positive through m = 17 is not positive for all m; that
difference IS the wall.

— Selina + Claude, 2026-07-11
