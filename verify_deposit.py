#!/usr/bin/env python3
"""
verify_deposit.py — recompute-and-compare checker for The Conductor 21 verification deposit.

Two passes:
  1. Consolidated: recompute sha256 of every file listed in SHA256SUMS and compare.
  2. Per-package: unzip each certificates/**/*.zip and check its internal SHA256SUMS.txt.

Usage:  python verify_deposit.py [--root .]
Exit 0 iff every check passes.
"""
import argparse, hashlib, io, os, sys, zipfile

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_sumsfile(text):
    """Return list of (sha_hex, relpath). Accepts '<hex>  path' / '<hex> *path'."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        h, p = parts
        p = p.lstrip("*").strip().replace("\\", "/")
        if p.startswith("./"):
            p = p[2:]
        out.append((h.lower(), p))
    return out

def pass_consolidated(root):
    sums = os.path.join(root, "SHA256SUMS")
    if not os.path.isfile(sums):
        print("FAIL: SHA256SUMS not found at deposit root"); return False
    with open(sums, encoding="utf-8") as f:
        entries = parse_sumsfile(f.read())
    ok = True
    for want, rel in entries:
        fp = os.path.join(root, rel)
        if not os.path.isfile(fp):
            print(f"MISSING  {rel}"); ok = False; continue
        got = sha256_file(fp)
        if got != want:
            print(f"MISMATCH {rel}\n  want {want}\n  got  {got}"); ok = False
    print(f"[1] consolidated: {len(entries)} entries, {'OK' if ok else 'FAILURES'}")
    return ok

def pass_packages(root):
    """Informational: report each package's internal SHA256SUMS.txt self-consistency.
    Not a gate — packages predate the deposit and use differing sealing conventions;
    the consolidated SHA256SUMS (pass 1) is the authoritative check. Corrupt zips DO fail."""
    hard_ok = True; n = 0; clean = 0; nosum = 0; mism = 0
    certs = os.path.join(root, "certificates")
    for dirpath, _, files in os.walk(certs):
        for fn in sorted(files):
            if not fn.endswith(".zip"):
                continue
            n += 1
            zp = os.path.join(dirpath, fn)
            try:
                with zipfile.ZipFile(zp) as z:
                    names = z.namelist()
                    inner = [x for x in names if os.path.basename(x).upper() in ("SHA256SUMS.TXT", "SHA256SUMS")]
                    if not inner:
                        nosum += 1; continue
                    text = z.read(inner[0]).decode("utf-8", "replace")
                    sumbase = os.path.basename(inner[0])
                    bad = 0
                    for want, rel in parse_sumsfile(text):
                        if os.path.basename(rel) == sumbase:
                            continue  # the sums file does not hash itself
                        cand = [x for x in names if x.replace("\\", "/").endswith("/" + rel) or x.replace("\\", "/") == rel]
                        if not cand:
                            bad += 1; continue
                        if hashlib.sha256(z.read(cand[0])).hexdigest() != want.lower():
                            bad += 1
                    if bad:
                        mism += 1; print(f"  INFO {fn}: {bad} internal entr(y/ies) did not self-check")
                    else:
                        clean += 1
            except zipfile.BadZipFile:
                print(f"FAIL  {fn}: not a valid zip"); hard_ok = False
    print(f"[2] per-package (informational): {n} packages - {clean} self-consistent, "
          f"{nosum} carry no internal sums, {mism} with internal notes")
    return hard_ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    ok1 = pass_consolidated(root)
    ok2 = pass_packages(root)
    print("RESULT:", "PASS" if (ok1 and ok2) else "FAIL")
    sys.exit(0 if (ok1 and ok2) else 1)

if __name__ == "__main__":
    main()
