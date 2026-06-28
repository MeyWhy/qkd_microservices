from __future__ import annotations

import csv
import math
import os
from pathlib import Path

# Standard telecom fiber parameters (SMF-28 at 1550 nm)
ALPHA_DB_PER_KM_DEFAULT = 0.2   # dB/km
DISTANCE_RANGE_KM       = list(range(0, 130, 10))  # 0, 10, 20, ..., 120 km

#coeff du pmd
D_PMD_PS_PER_SQRT_KM_DEFAULT = 0.1   # ps / sqrt(km)

# Default output location — sibling to this file
_DATA_DIR     = Path(__file__).parent / "data"
_DEFAULT_CSV  = _DATA_DIR / "attenuation_table.csv"
_DEFAULT_PMD_CSV = _DATA_DIR / "pmd_table.csv"


def _transmission(distance_km: float, alpha: float = ALPHA_DB_PER_KM_DEFAULT) -> float:
    """
    Standard Beer-Lambert attenuation for optical fiber.
    T(d) = 10 ^ (-alpha * d / 10)

    At alpha=0.2 dB/km:
      0 km  → T = 1.000  (no loss)
     50 km  → T = 0.010  (99% lost — typical QKD limit)
    100 km  → T = 0.0001 (99.99% lost)
    """
    return 10 ** (-(alpha * distance_km) / 10)


def generate_synthetic_csv(
    output_path: str | Path = _DEFAULT_CSV,
    alpha_db_per_km: float  = ALPHA_DB_PER_KM_DEFAULT,
    distances_km: list[int] = None,
    overwrite: bool         = False,
) -> Path:
  
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return output_path

    distances = distances_km or DISTANCE_RANGE_KM

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance_km", "transmission_prob"])
        for d in distances:
            t = _transmission(d, alpha_db_per_km)
            writer.writerow([d, f"{t:.8f}"])

    return output_path

def _dgd_ps(distance_km: float, d_pmd: float = D_PMD_PS_PER_SQRT_KM_DEFAULT) -> float:
    """
    Differential group delay from the standard sqrt(length) PMD scaling
    for fiber with random mode coupling:
        DGD(d) = D_PMD * sqrt(d)
    At D_PMD=0.1 ps/sqrt(km) (SMF-28-class, ITU-T G.652):
      10 km  -> DGD ≈ 0.32 ps
      50 km  -> DGD ≈ 0.71 ps
     100 km  -> DGD ≈ 1.00 ps
    """
    if distance_km <= 0:
        return 0.0
    return d_pmd * math.sqrt(distance_km)
 
 
def generate_synthetic_pmd_csv(
    output_path:    str | Path = _DEFAULT_PMD_CSV,
    d_pmd_ps_per_sqrt_km: float = D_PMD_PS_PER_SQRT_KM_DEFAULT,
    distances_km:   list[int] = None,
    overwrite:      bool = False,
) -> Path:
    """
    Generate a synthetic PMD table (distance_km, dgd_ps), standing in for
    a real Ansys/lab-measured DGD sweep until one is available. Swap the
    CSV file for a measured one later — ChannelModel's interface doesn't
    change.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    if output_path.exists() and not overwrite:
        return output_path
 
    distances = distances_km or DISTANCE_RANGE_KM
 
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance_km", "dgd_ps"])
        for d in distances:
            dgd = _dgd_ps(d, d_pmd_ps_per_sqrt_km)
            writer.writerow([d, f"{dgd:.6f}"])
 
    return output_path
 
 
def validate_pmd_csv(csv_path: str | Path) -> dict:
    """
    Validate a PMD CSV: correct columns, non-negative DGD, monotonically
    non-decreasing with distance (sqrt scaling is monotonic by construction).
    """
    csv_path = Path(csv_path)
    issues: list[str] = []
 
    if not csv_path.exists():
        return {"valid": False, "rows": 0, "issues": [f"File not found: {csv_path}"], "summary": {}}
 
    rows: list[dict] = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "distance_km" not in (reader.fieldnames or []):
                issues.append("Missing column: distance_km")
            if "dgd_ps" not in (reader.fieldnames or []):
                issues.append("Missing column: dgd_ps")
            if issues:
                return {"valid": False, "rows": 0, "issues": issues, "summary": {}}
            for i, row in enumerate(reader):
                try:
                    rows.append({
                        "distance_km": float(row["distance_km"]),
                        "dgd_ps":      float(row["dgd_ps"]),
                    })
                except ValueError:
                    issues.append(f"Row {i+2}: non-numeric value: {row}")
    except Exception as e:
        return {"valid": False, "rows": 0, "issues": [str(e)], "summary": {}}
 
    if not rows:
        issues.append("CSV has no data rows")
        return {"valid": False, "rows": 0, "issues": issues, "summary": {}}
 
    prev_d = prev_dgd = None
    for row in rows:
        d, dgd = row["distance_km"], row["dgd_ps"]
        if d < 0:
            issues.append(f"Negative distance: {d}")
        if dgd < 0:
            issues.append(f"Negative DGD at d={d}: {dgd}")
        if prev_d is not None and d <= prev_d:
            issues.append(f"Distances not strictly increasing: {prev_d} → {d}")
        if prev_dgd is not None and dgd < prev_dgd - 1e-9:
            issues.append(f"DGD decreased at d={d}: {prev_dgd:.6f} → {dgd:.6f} (should be non-decreasing)")
        prev_d, prev_dgd = d, dgd
 
    distances = [r["distance_km"] for r in rows]
    dgds      = [r["dgd_ps"]      for r in rows]
 
    # Implied D_PMD from the last point: DGD = D_PMD * sqrt(d)
    implied_d_pmd = (
        round(dgds[-1] / math.sqrt(distances[-1]), 6)
        if distances[-1] > 0 and dgds[-1] > 0
        else None
    )
 
    return {
        "valid":   len(issues) == 0,
        "rows":    len(rows),
        "issues":  issues,
        "summary": {
            "min_distance_km":          min(distances),
            "max_distance_km":          max(distances),
            "max_dgd_ps":               max(dgds),
            "implied_d_pmd_ps_per_sqrt_km": implied_d_pmd,
        },
    }
 

def validate_csv(csv_path: str | Path) -> dict:

    csv_path = Path(csv_path)
    issues: list[str] = []

    if not csv_path.exists():
        return {
            "valid":   False,
            "rows":    0,
            "issues":  [f"File not found: {csv_path}"],
            "summary": {},
        }

    rows: list[dict] = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "distance_km" not in (reader.fieldnames or []):
                issues.append("Missing column: distance_km")
            if "transmission_prob" not in (reader.fieldnames or []):
                issues.append("Missing column: transmission_prob")
            if issues:
                return {"valid": False, "rows": 0, "issues": issues, "summary": {}}
            for i, row in enumerate(reader):
                try:
                    rows.append({
                        "distance_km":       float(row["distance_km"]),
                        "transmission_prob": float(row["transmission_prob"]),
                    })
                except ValueError:
                    issues.append(f"Row {i+2}: non-numeric value: {row}")
    except Exception as e:
        return {"valid": False, "rows": 0, "issues": [str(e)], "summary": {}}

    if not rows:
        issues.append("CSV has no data rows")
        return {"valid": False, "rows": 0, "issues": issues, "summary": {}}

    prev_d = prev_t = None
    for row in rows:
        d, t = row["distance_km"], row["transmission_prob"]
        if d < 0:
            issues.append(f"Negative distance: {d}")
        if not (0.0 < t <= 1.0):
            issues.append(f"Transmission out of (0,1] at d={d}: {t}")
        if prev_d is not None and d <= prev_d:
            issues.append(f"Distances not strictly increasing: {prev_d} → {d}")
        if prev_t is not None and t > prev_t + 1e-9:
            issues.append(
                f"Transmission increased at d={d}: {prev_t:.6f} → {t:.6f} "
                f"(should be non-increasing)"
            )
        prev_d, prev_t = d, t

    distances     = [r["distance_km"]       for r in rows]
    transmissions = [r["transmission_prob"]  for r in rows]

    return {
        "valid":   len(issues) == 0,
        "rows":    len(rows),
        "issues":  issues,
        "summary": {
            "min_distance_km":  min(distances),
            "max_distance_km":  max(distances),
            "max_transmission": max(transmissions),
            "min_transmission": min(transmissions),
            "alpha_implied_db_per_km": (
                round(
                    -10 * math.log10(transmissions[-1]) / distances[-1], 4
                )
                if distances[-1] > 0 and transmissions[-1] > 0
                else None
            ),
        },
    }


def print_table(csv_path: str | Path = _DEFAULT_CSV) -> None:
    """Pretty-print the attenuation table for inspection."""
    csv_path = Path(csv_path)
    result   = validate_csv(csv_path)
    if not result["valid"]:
        print(f"Invalid CSV: {result['issues']}")
        return

    print(f"\n{'Distance (km)':>14}  {'T(d)':>12}  {'Loss (dB)':>10}  {'Survival %':>11}")
    print("-" * 54)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = float(row["distance_km"])
            t = float(row["transmission_prob"])
            loss_db = -10 * math.log10(t) if t > 0 else float("inf")
            print(
                f"{d:>14.0f}  {t:>12.6f}  {loss_db:>10.2f}  {t*100:>10.4f}%"
            )
    s = result["summary"]
    print(f"\n  α (implied) = {s['alpha_implied_db_per_km']} dB/km")
    print(f"  Range: {s['min_distance_km']} – {s['max_distance_km']} km\n")


if __name__ == "__main__":
    path = generate_synthetic_csv(overwrite=True)
    print(f"Generated: {path}")
    print_table(path)

    pmd_path = generate_synthetic_pmd_csv(overwrite=True)
    print(f"Generated: {pmd_path}")
    pmd_result=validate_pmd_csv(pmd_path)
    print(f"\nPMD table valid: {pmd_result['valid']}")
    if pmd_result["valid"]:
        s = pmd_result["summary"]
        print(f"  Implied D_PMD = {s['implied_d_pmd_ps_per_sqrt_km']} ps/√km")
        print(f"  Max DGD = {s['max_dgd_ps']:.4f} ps at {s['max_distance_km']} km")
        
    """
 Ansys attenuation table generator / validator.

This module does two things:

  A) generate_synthetic_csv()
       Produces the exact same CSV structure that Ansys would export,
       using the standard attenuation formula:
           T(d) = 10 ^ (- alpha_dB_per_km * d / 10)
       with alpha = 0.2 dB/km (SMF-28 at 1550 nm, telecom standard).
       Use this as a stand-in until you run the real Ansys simulation.
       When you do run Ansys, just replace the CSV file — zero code changes.

  B) validate_csv()
       Checks that a CSV (synthetic or Ansys-exported) has the expected
       columns, monotonically decreasing transmission, and no missing values.
       Call this once at startup to catch bad exports early.

ROLE IN THE ARCHITECTURE
-------------------------
In a real workflow:
  1. You open Ansys Lumerical MODE or FDTD
  2. Define a silica SMF-28 fiber waveguide
  3. Sweep length 0 → 120 km
  4. Export transmission power ratio at each distance
  5. Save as  optical/data/attenuation_table.csv

CSV FORMAT (what Ansys should export / what this generates)
-----------------------------------------------------------
  distance_km, transmission_prob
  0,           1.000000
  10,          0.630957
  20,          0.398107
  ...
  120,         0.000631

REAL ANSYS WORKFLOW (for reference)
-------------------------------------
  In Lumerical MODE:
    1. File → New → Waveguide simulation
    2. Material: SiO2 (silica), n = 1.4682 at 1550 nm
    3. Geometry: length = variable, core diameter = 9 μm (SMF-28)
    4. Source: fundamental HE11 mode at λ = 1550 nm
    5. Monitor: transmission at fiber output
    6. Parameter sweep: length from 0 to 120 km, step 10 km
    7. Export: Results → Transmission → Export to CSV
    8. Rename columns to: distance_km, transmission_prob
    9. Place at: optical/data/attenuation_table.csv
"""
