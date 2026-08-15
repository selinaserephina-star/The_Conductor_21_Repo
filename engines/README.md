# engines/ — reusable instruments (Appendix C.2)

The object-independent instruments behind the certificates. **Every file here was extracted
verbatim from a deposited certificate package** — i.e. each engine is *exactly the version that
produced the certificate it accompanies*, so there is no version-matching question and no
dependence on any external repository. (The private `Conductor_21_scripts` repo does **not**
contain these core engines and is not a source for the deposit.)

Each engine's authoritative, self-contained copy also remains inside its certificate `.zip` under
`certificates/`; the copies here are surfaced for convenient reuse.

| engine | dir | version-authentic source (in `certificates/`) |
|---|---|---|
| Rescaled-FFT completeness engine | `rescaled_fft/` | `II_5_.../completeness_CLOSED` (`engine/*.py`) + `II_3_.../CERTIFIED_spectrum` (`certified_engine.py`) |
| Ball-arithmetic certification (Arb / python-flint) **+ NK-certified GL-node balls + ball-Cholesky** | `ball_cert/` | `III_4_6_.../INFINITE_SIDE_closeout` (`interval_grids_arb.py`) |
| M1-HORIZON exact-layer determinant engine | `m1_horizon/` | `III_2_3_.../THEOREM_M1_HORIZON` (`m1_horizon_*.py`) |
| Hybrid backward-Miller / log-mantissa row engine | `row_engine/` | `III_7_.../CAUSTIC_THEOREM` (`t2_d2_*.py`; `rows()` is defined inline in the consuming scripts) |
| Olver interval ladder + adaptive ball-cell bisection (`olver_ivl_*`) | `olver_ivl/` | `III_8_.../COMPONENT_F_CLOSED` (`olver_ivl_*.py`) |
| Arb-Lanczos (plain 3-term, no reorth) | `arb_lanczos/` | `IV_8_.../KLMN_ARB_CERT` (`diag_lanczos_growth.py`, `test_lanczos_radii.py`) |

**NK-certified GL-node balls** are part of the `interval_grids_arb` engine (see `ball_cert/`),
not a separate file — the manifest's 7th instrument lives inside the 2nd.

**The only modification to any file here:** in `arb_lanczos/`, two hardcoded local absolute
input paths were reduced to their basenames (privacy). The unmodified originals are byte-identical
inside `certificates/IV_8_transport_KLMN/…KLMN_ARB_CERT….zip`. No numeric content changed.
