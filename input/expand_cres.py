#!/usr/bin/env python3
"""
Expand CALVADOS custom restraints to multiple molecule copies.

Input line format (one per restraint):
<compA> <copyA> <resA> | <compB> <copyB> <resB> | <r0_nm> <k>

Example:
hpl-dimer 1 14 | hpl-dimer 1 185 | 0.539 700.0
"""

from pathlib import Path
import re
from typing import Iterable

LINE_RE = re.compile(
    r"""^\s*
        (\S+)\s+(\d+)\s+(\d+)\s*     # compA copyA resA
        \|\s*
        (\S+)\s+(\d+)\s+(\d+)\s*     # compB copyB resB
        \|\s*
        ([0-9.]+)\s+([0-9.]+)\s*     # r0_nm  k
        $""",
    re.VERBOSE,
)

def expand_restraints_lines(lines: Iterable[str], nmol: int) -> list[str]:
    """
    Replicate each parsed restraint for copies 1..nmol.
    Only the copy indices are changed; components, residues, r0, k are kept.
    """
    out: list[str] = []
    for raw in lines:
        s = raw.rstrip("\n")
        if not s.strip() or s.lstrip().startswith("#"):
            out.append(s)  # keep blank/comment lines
            continue

        m = LINE_RE.match(s)
        if not m:
            raise ValueError(f"Cannot parse restraint line: {s!r}")

        compA, copyA, resA, compB, copyB, resB, r0, k = m.groups()
        resA = int(resA); resB = int(resB)
        r0f = float(r0); kf = float(k)

        for c in range(1, nmol + 1):
            out.append(f"{compA} {c} {resA} | {compB} {c} {resB} | {r0f:.3f} {kf:.1f}")
    return out

def expand_restraints_file(input_path: str, output_path: str, nmol: int) -> None:
    inp = Path(input_path).read_text().splitlines()
    out_lines = expand_restraints_lines(inp, nmol)
    Path(output_path).write_text("\n".join(out_lines) + "\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Expand CALVADOS custom restraints to N molecule copies.")
    ap.add_argument("--input", help="input restraints file (for copy 1)")
    ap.add_argument("--output", default = "custom_restraints.txt", required = False, help="output restraints file (for copies 1..N)")
    ap.add_argument("--nmol", type=int, required=True, help="number of molecules/copies in the system")
    args = ap.parse_args()
    expand_restraints_file(args.input, args.output, args.nmol)
