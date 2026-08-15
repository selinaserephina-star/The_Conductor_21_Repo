# The Conductor 21 — verification deposit

Public verification artifact for the paper

> **The Conductor-21 L-Function: Certified Zeros, Exact Laws, and a Density–Prime Operator**
> Selina Stenberg and Ilya Balashov

(the χ₈ / conductor-21, PSL(2,7) genome program). This repository is the source tree for the
Zenodo deposit named in the paper's **Appendix C** (DOI pending), released under **MIT** (code) +
**CC BY 4.0** (data), verifiable against a consolidated `SHA256SUMS`.

> **Scope.** This is exactly the set a third party needs to *re-check the published claims* —
> nothing more. Internal working record, correspondence, exploration threads, and the operator
> *construction/development* pipeline are deliberately held back (see the paper's Appendix C and
> `DEPOSIT_STATUS.md`).

## Layout

```
certificates/<section>/   one directory per Appendix C.3 map row;
                          each holds the frozen named package(s):
                          script(s) + certificate data + frozen inputs + run logs,
                          each package carrying its own internal SHA256SUMS.txt
engines/                  Appendix C.2 reusable instruments (see DEPOSIT_STATUS.md — pending)
SHA256SUMS                consolidated, over every file in the deposit
verify_deposit.py         recompute-and-compare checker
LICENSE                   MIT (code)
LICENSE-CC-BY-4.0.txt     CC BY 4.0 (data)
DEPOSIT_STATUS.md         build state, coverage checklist, open decisions
```

## How to verify

```bash
sha256sum -c SHA256SUMS          # consolidated manifest
python verify_deposit.py         # recompute tree + check each package's internal SHA256SUMS.txt
```

The 8-hex prefixes cited in the paper's Appendix C resolve against the package zips here — every
cited prefix was checked byte-for-byte against this tree at build time (see `DEPOSIT_STATUS.md`).

## Authorship

The program is joint work of **Selina Stenberg** and **Ilya Balashov**.

## License

- **MIT** → all code (`engines/`, `certificates/**/*.py`).
- **CC BY 4.0** → all data (`*.csv`, certification scans, the certified spectrum).

The construction *development* pipeline is not deposited and therefore not MIT-licensed; MIT
applies only to what ships here.
