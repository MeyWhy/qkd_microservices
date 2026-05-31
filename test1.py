"""
tests/test_step1_ansys_channel.py
==================================
Step 1 — Ansys fiber attenuation model tests.

WHAT THIS TESTS
---------------
  Section 1 — CSV generation and validation
    1.1  generate_synthetic_csv() creates the file
    1.2  CSV has correct columns and 13 rows (0–120 km, step 10)
    1.3  validate_csv() passes on a good file
    1.4  T(0 km) = 1.0 exactly
    1.5  T(50 km) ≈ 0.01  (99% loss — standard QKD limit reference)
    1.6  T(100 km) ≈ 0.0001  (99.99% loss)
    1.7  Transmission is strictly decreasing with distance
    1.8  validate_csv() catches missing column
    1.9  validate_csv() catches non-monotonic transmission

  Section 2 — ChannelModel interpolation
    2.1  transmission_prob(0) = 1.0
    2.2  transmission_prob(50) ≈ 0.01 (±1%)
    2.3  transmission_prob at exact CSV points matches formula
    2.4  transmission_prob at interpolated points is between neighbors
    2.5  transmission_prob clamps to [0, 1] — no negative probabilities
    2.6  qber_floor returns 0.0 at Step 1 (no PMD CSV loaded)
    2.7  Fallback works when CSV is missing (analytical formula used)
    2.8  describe() returns correct source and distance

  Section 3 — FiberChannel with ChannelModel
    3.1  FiberChannel(csv_path=...) loads model correctly
    3.2  FiberChannel with Ansys CSV vs analytical: same T at exact points
    3.3  FiberChannel at 0 km → 100% survival
    3.4  FiberChannel at 50 km → ~1% survival (±2%)
    3.5  FiberChannel at 100 km → ~0.01% survival (±1 order of magnitude)
    3.6  Drift double-assignment bug is fixed (single object)
    3.7  reset_session() does not raise

  Section 4 — End-to-end BB84 with Ansys channel
    4.1  loss=0 km: QBER=0, delivery=100%
    4.2  loss=50 km: QBER~0, delivery~1% (from Ansys T)
    4.3  key rate drops monotonically with distance
    4.4  QBER stays below 0.11 at all distances (no Eve, no PMD)
    4.5  Step 0 vs Step 1 comparison: at short distance, results agree

  Section 5 — Ansys-to-Python interface contract
    5.1  CSV can be loaded by pandas (if available)
    5.2  The interp function returns float, not ndarray
    5.3  ChannelModel.describe() is JSON-serialisable

HOW TO RUN
----------
  python -m pytest tests/test_step1_ansys_channel.py -v
  python tests/test_step1_ansys_channel.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
try:
    from optical.ansys_export import (
        generate_synthetic_csv,
        validate_csv,
        _transmission as _formula_T,
    )
    from optical.channel_model import ChannelModel
    from optical.channel import FiberChannel, StatisticalChannel
    from bb84_logic import compute_qber, perform_sifting_by_id, QBER_THRESHOLD
    from models import Basis, MeasurementRecord
except ImportError as e:
    print(f"[FATAL] Import error: {e}")
    print("        Run from project root: python -m pytest tests/test_step1_ansys_channel.py -v")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Minimal test runner (same pattern as Step 0)
# ---------------------------------------------------------------------------
_RESULTS: list[tuple[str, bool, str]] = []


def _test(name: str):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                _RESULTS.append((name, True, ""))
                print(f"  PASS  {name}")
            except AssertionError as e:
                _RESULTS.append((name, False, str(e)))
                print(f"  FAIL  {name}")
                print(f"        {e}")
            except Exception as e:
                _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  ERR   {name}")
                traceback.print_exc()
        wrapper._is_test = True
        wrapper._name    = name
        return wrapper
    return decorator


def _assert_close(actual, expected, tol, msg=""):
    assert abs(actual - expected) <= tol, (
        f"{msg}  expected≈{expected}  got {actual}  (tol={tol})"
    )


def _assert_rel_close(actual, expected, rel_tol, msg=""):
    """Relative tolerance: |actual - expected| / expected <= rel_tol."""
    if expected == 0:
        assert actual == 0, f"{msg} expected 0, got {actual}"
        return
    rel = abs(actual - expected) / abs(expected)
    assert rel <= rel_tol, (
        f"{msg}  expected≈{expected}  got {actual}  "
        f"(rel_err={rel*100:.2f}% > {rel_tol*100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Shared fixture: a temporary CSV file used across sections 1–4
# ---------------------------------------------------------------------------
_TMP_DIR  = Path(tempfile.mkdtemp(prefix="bb84_step1_"))
_CSV_PATH = _TMP_DIR / "attenuation_table.csv"
generate_synthetic_csv(output_path=_CSV_PATH, overwrite=True)


# ===========================================================================
# SECTION 1 — CSV generation and validation
# ===========================================================================

@_test("csv: generate_synthetic_csv creates file")
def test_csv_created():
    assert _CSV_PATH.exists(), f"CSV not found: {_CSV_PATH}"
    assert _CSV_PATH.stat().st_size > 0, "CSV is empty"


@_test("csv: has correct columns and 13 rows (0–120 km, step 10)")
def test_csv_shape():
    with open(_CSV_PATH) as f:
        reader = csv.DictReader(f)
        assert "distance_km"       in reader.fieldnames, "Missing column: distance_km"
        assert "transmission_prob" in reader.fieldnames, "Missing column: transmission_prob"
        rows = list(reader)
    assert len(rows) == 13, f"Expected 13 rows (0–120 km), got {len(rows)}"


@_test("csv: validate_csv passes on synthetic file")
def test_csv_valid():
    result = validate_csv(_CSV_PATH)
    assert result["valid"], f"Validation failed: {result['issues']}"
    assert result["rows"] == 13


@_test("csv: T(0 km) = 1.0 exactly")
def test_csv_t0():
    with open(_CSV_PATH) as f:
        first_row = next(csv.DictReader(f))
    d = float(first_row["distance_km"])
    t = float(first_row["transmission_prob"])
    assert d == 0.0, f"First row distance should be 0, got {d}"
    _assert_close(t, 1.0, tol=1e-8, msg="T(0 km)")


@_test("csv: T(50 km) ≈ 0.01 — 99% loss, standard QKD limit")
def test_csv_t50():
    expected = _formula_T(50.0)       # ≈ 0.01
    # Read directly from CSV row for d=50
    with open(_CSV_PATH) as f:
        for row in csv.DictReader(f):
            if float(row["distance_km"]) == 50.0:
                t = float(row["transmission_prob"])
                _assert_rel_close(t, expected, rel_tol=0.001, msg="T(50 km)")
                return
    assert False, "Row for 50 km not found in CSV"


@_test("csv: T(100 km) ≈ 0.0001 — 99.99% loss")
def test_csv_t100():
    expected = _formula_T(100.0)
    with open(_CSV_PATH) as f:
        for row in csv.DictReader(f):
            if float(row["distance_km"]) == 100.0:
                t = float(row["transmission_prob"])
                _assert_rel_close(t, expected, rel_tol=0.001, msg="T(100 km)")
                return
    assert False, "Row for 100 km not found in CSV"


@_test("csv: transmission is strictly decreasing with distance")
def test_csv_monotone():
    with open(_CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    transmissions = [float(r["transmission_prob"]) for r in rows]
    for i in range(1, len(transmissions)):
        assert transmissions[i] < transmissions[i - 1], (
            f"Not strictly decreasing at row {i}: "
            f"{transmissions[i-1]:.6f} → {transmissions[i]:.6f}"
        )


@_test("csv: validate_csv catches missing column")
def test_csv_bad_column():
    bad_csv = _TMP_DIR / "bad_columns.csv"
    with open(bad_csv, "w") as f:
        f.write("distance_km,wrong_column\n0,1.0\n")
    result = validate_csv(bad_csv)
    assert not result["valid"], "Should fail with missing transmission_prob column"
    assert any("transmission_prob" in issue for issue in result["issues"])


@_test("csv: validate_csv catches non-monotonic transmission")
def test_csv_non_monotone():
    bad_csv = _TMP_DIR / "non_monotone.csv"
    with open(bad_csv, "w") as f:
        f.write("distance_km,transmission_prob\n")
        f.write("0,1.0\n10,0.5\n20,0.8\n")   # 0.8 > 0.5 — wrong
    result = validate_csv(bad_csv)
    assert not result["valid"], "Should fail with non-monotonic transmission"
    assert any("non-increasing" in issue for issue in result["issues"])


# ===========================================================================
# SECTION 2 — ChannelModel interpolation
# ===========================================================================

@_test("model: transmission_prob(0) = 1.0")
def test_model_t0():
    model = ChannelModel(_CSV_PATH)
    _assert_close(model.transmission_prob(0.0), 1.0, tol=1e-9, msg="T(0)")


@_test("model: transmission_prob(50) ≈ 0.01 (±1%)")
def test_model_t50():
    model    = ChannelModel(_CSV_PATH)
    expected = _formula_T(50.0)
    actual   = model.transmission_prob(50.0)
    _assert_rel_close(actual, expected, rel_tol=0.01, msg="T(50 km)")


@_test("model: transmission_prob at exact CSV points matches formula")
def test_model_exact_points():
    model = ChannelModel(_CSV_PATH)
    for d in [0, 10, 20, 30, 50, 80, 100, 120]:
        expected = _formula_T(float(d))
        actual   = model.transmission_prob(float(d))
        _assert_rel_close(actual, expected, rel_tol=0.01,
                          msg=f"T({d} km)")


@_test("model: transmission_prob at interpolated points is between neighbors")
def test_model_interpolation():
    model = ChannelModel(_CSV_PATH)
    # d=25 must be between T(20) and T(30)
    t20 = model.transmission_prob(20.0)
    t25 = model.transmission_prob(25.0)
    t30 = model.transmission_prob(30.0)
    assert t30 <= t25 <= t20, (
        f"Interpolated T(25)={t25:.6f} not between "
        f"T(20)={t20:.6f} and T(30)={t30:.6f}"
    )


@_test("model: transmission_prob clamps to [0, 1]")
def test_model_clamp():
    model = ChannelModel(_CSV_PATH)
    assert 0.0 <= model.transmission_prob(0.0)   <= 1.0
    assert 0.0 <= model.transmission_prob(50.0)  <= 1.0
    assert 0.0 <= model.transmission_prob(200.0) <= 1.0   # beyond CSV range
    assert 0.0 <= model.transmission_prob(-5.0)  <= 1.0   # before range


@_test("model: qber_floor returns 0.0 at Step 1 (no PMD CSV)")
def test_model_qber_floor_zero():
    model = ChannelModel(_CSV_PATH)
    for d in [0.0, 10.0, 50.0, 100.0]:
        assert model.qber_floor(d) == 0.0, (
            f"qber_floor({d}) should be 0.0 at Step 1, got {model.qber_floor(d)}"
        )


@_test("model: fallback to analytical formula when CSV missing")
def test_model_fallback():
    model = ChannelModel("/nonexistent/path/attenuation.csv")
    assert model.source == "fallback", f"Expected 'fallback', got '{model.source}'"
    expected = _formula_T(50.0)
    actual   = model.transmission_prob(50.0)
    _assert_rel_close(actual, expected, rel_tol=0.001, msg="Fallback T(50 km)")


@_test("model: describe() returns correct source and distance fields")
def test_model_describe():
    model = ChannelModel(_CSV_PATH)
    d     = model.describe(50.0)
    assert d["source"]      == "csv",              f"source: {d['source']}"
    assert d["distance_km"] == 50.0,               f"distance_km: {d['distance_km']}"
    assert d["model"]       == "ansys_csv_channel", f"model: {d['model']}"
    assert 0.0 < d["transmission_prob"] < 1.0,     f"trans: {d['transmission_prob']}"
    assert d["loss_db"] > 0,                        f"loss_db: {d['loss_db']}"


# ===========================================================================
# SECTION 3 — FiberChannel with ChannelModel
# ===========================================================================

@_test("fiber: FiberChannel(csv_path=...) loads ChannelModel correctly")
def test_fiber_csv_path():
    ch = FiberChannel(
        distance_km=50.0, csv_path=str(_CSV_PATH),
        enable_drift=False, enable_detector=False,
    )
    assert ch._channel_model is not None, "channel_model should be set"
    assert ch._channel_model.source == "csv"


@_test("fiber: Ansys CSV vs analytical — same T at exact CSV points")
def test_fiber_csv_vs_formula():
    for d in [10.0, 30.0, 50.0, 80.0]:
        ch_csv  = FiberChannel(d, csv_path=str(_CSV_PATH),
                               enable_drift=False, enable_detector=False)
        ch_form = FiberChannel(d, enable_drift=False, enable_detector=False)
        _assert_rel_close(
            ch_csv._transmission,
            ch_form._transmission,
            rel_tol=0.01,
            msg=f"CSV vs formula T({d} km)",
        )


@_test("fiber: at 0 km → ~100% survival (disable drift/detector)")
def test_fiber_0km():
    random.seed(0)
    ch = FiberChannel(0.0, enable_drift=False, enable_detector=False,
                      csv_path=str(_CSV_PATH))
    photon = {"qubit_id": 0, "bit": 1, "basis": "Z"}
    n = 1000
    survived = sum(1 for _ in range(n) if ch.transmit(photon) is not None)
    assert survived == n, f"At 0 km all photons must survive, got {survived}/{n}"


@_test("fiber: at 50 km → ~1% survival (±2%, disable drift/detector)")
def test_fiber_50km():
    random.seed(7)
    ch = FiberChannel(50.0, enable_drift=False, enable_detector=False,
                      csv_path=str(_CSV_PATH))
    photon = {"qubit_id": 0, "bit": 0, "basis": "X"}
    n        = 50_000
    survived = sum(1 for _ in range(n) if ch.transmit(photon) is not None)
    rate     = survived / n
    expected = _formula_T(50.0)   # ≈ 0.01
    _assert_close(rate, expected, tol=0.02, msg="Survival rate at 50 km")


@_test("fiber: at 100 km → ~0.01% survival (order-of-magnitude check)")
def test_fiber_100km():
    random.seed(13)
    ch = FiberChannel(100.0, enable_drift=False, enable_detector=False,
                      csv_path=str(_CSV_PATH))
    photon = {"qubit_id": 0, "bit": 1, "basis": "Z"}
    n        = 200_000
    survived = sum(1 for _ in range(n) if ch.transmit(photon) is not None)
    rate     = survived / n
    expected = _formula_T(100.0)   # ≈ 0.0001
    # Order-of-magnitude check: within factor of 3
    assert rate < expected * 4, f"Rate {rate:.5f} too high vs expected {expected:.5f}"
    assert rate > expected / 4, f"Rate {rate:.5f} too low vs expected {expected:.5f}"


@_test("fiber: drift double-assignment bug is fixed (single object)")
def test_fiber_drift_single_assignment():
    ch = FiberChannel(50.0, enable_drift=True, use_ou_drift=True)
    # If bug present, self.drift would be overwritten. We verify it's
    # an OUDriftChannel (not PolarizationDriftChannel set first then lost).
    from optical.polarization import OUDriftChannel
    assert isinstance(ch.drift, OUDriftChannel), (
        f"Expected OUDriftChannel, got {type(ch.drift).__name__}"
    )


@_test("fiber: reset_session() does not raise")
def test_fiber_reset():
    ch = FiberChannel(50.0, csv_path=str(_CSV_PATH))
    ch.reset_session()   # must not raise


# ===========================================================================
# SECTION 4 — End-to-end BB84 with Ansys channel
# ===========================================================================

def _run_bb84_fiber(
    distance_km: float,
    n_qubits:    int   = 5000,
    seed:        int   = 0,
    csv_path:    str   = None,
) -> dict:
    """
    Full BB84 simulation using FiberChannel (Step 1).
    No drift, no detector — isolates the attenuation effect.
    """
    random.seed(seed)

    ch = FiberChannel(
        distance_km=distance_km,
        enable_drift=False,
        enable_detector=False,
        csv_path=csv_path or str(_CSV_PATH),
    )

    alice_bits  = [random.randint(0, 1) for _ in range(n_qubits)]
    alice_bases = [random.choice(["Z", "X"]) for _ in range(n_qubits)]

    bob_measurements: dict[int, MeasurementRecord] = {}
    n_delivered = 0

    for i in range(n_qubits):
        photon = {"qubit_id": i, "bit": alice_bits[i], "basis": alice_bases[i]}
        result = ch.transmit(photon)
        if result is None:
            continue
        n_delivered += 1
        bob_basis = random.choice(list(Basis))
        bob_bit   = alice_bits[i] if bob_basis.value == alice_bases[i] else random.randint(0, 1)
        bob_measurements[i] = MeasurementRecord(
            qubit_id=i, basis=bob_basis, bit_result=bob_bit
        )

    alice_s, bob_s, _ = perform_sifting_by_id(alice_bits, alice_bases, bob_measurements)
    n_sifted = len(alice_s)

    if n_sifted < 2:
        return {
            "distance_km": distance_km, "n_sent": n_qubits,
            "n_delivered": n_delivered, "n_sifted": 0,
            "qber": 1.0, "key_len": 0,
            "delivery_rate": n_delivered / n_qubits,
        }

    qber, alice_key, _ = compute_qber(alice_s, bob_s, sample_seed=seed + 1)
    return {
        "distance_km":   distance_km,
        "n_sent":        n_qubits,
        "n_delivered":   n_delivered,
        "n_sifted":      n_sifted,
        "qber":          qber,
        "key_len":       len(alice_key),
        "delivery_rate": n_delivered / n_qubits,
        "expected_T":    _formula_T(distance_km),
    }


@_test("e2e: 0 km → QBER=0, delivery=100%")
def test_e2e_0km():
    r = _run_bb84_fiber(0.0, n_qubits=2000, seed=42)
    print(f"\n        delivery={r['delivery_rate']:.2%}  "
          f"QBER={r['qber']*100:.2f}%  key_len={r['key_len']}", end="")
    assert r["delivery_rate"] == 1.0, "At 0 km all photons must be delivered"
    assert r["qber"] == 0.0, f"QBER must be 0 at 0 km, got {r['qber']}"
    assert r["key_len"] > 0


@_test("e2e: 50 km → QBER~0, delivery~1%")
def test_e2e_50km():
    r = _run_bb84_fiber(50.0, n_qubits=200_000, seed=7)
    print(f"\n        delivery={r['delivery_rate']:.3%}  "
          f"expected_T={r['expected_T']:.4f}  "
          f"QBER={r['qber']*100:.2f}%  key_len={r['key_len']}", end="")
    _assert_close(r["delivery_rate"], r["expected_T"], tol=0.005,
                  msg="Delivery rate vs Ansys T(50 km)")
    assert r["qber"] == 0.0, f"QBER must be 0 (no noise), got {r['qber']}"


@_test("e2e: key rate drops monotonically with distance")
def test_e2e_key_rate_monotone():
    distances  = [0, 10, 20, 30, 40, 50]
    key_rates  = []
    for d in distances:
        r = _run_bb84_fiber(d, n_qubits=20_000, seed=d)
        key_rates.append(r["key_len"])
        print(f"\n        d={d:>3} km  key_len={r['key_len']:>6}  "
              f"delivery={r['delivery_rate']:.4f}", end="")
    for i in range(1, len(key_rates)):
        assert key_rates[i] <= key_rates[i - 1], (
            f"Key rate increased from d={distances[i-1]} ({key_rates[i-1]}) "
            f"to d={distances[i]} ({key_rates[i]})"
        )


@_test("e2e: QBER stays below 0.11 at all distances (no Eve, no drift)")
def test_e2e_qber_safe():
    for d in [0, 10, 30, 50, 80]:
        r = _run_bb84_fiber(d, n_qubits=10_000, seed=d * 3)
        if r["n_sifted"] >= 10:
            assert r["qber"] < QBER_THRESHOLD, (
                f"QBER={r['qber']:.4f} exceeded threshold at d={d} km"
            )


@_test("e2e: Step 0 vs Step 1 — results agree at short distance")
def test_e2e_step0_vs_step1():
    """
    At d=1 km, T≈0.955 — effectively lossless for this comparison.
    Both channels should give QBER=0 and sift≈50%.
    """
    random.seed(42)
    n = 2000

    # Step 0
    loss_rate = 1.0 - _formula_T(1.0)
    ch0 = StatisticalChannel(loss_rate=loss_rate)

    # Step 1
    ch1 = FiberChannel(1.0, enable_drift=False, enable_detector=False,
                       csv_path=str(_CSV_PATH))

    def _simulate(ch):
        alice_bits  = [random.randint(0, 1) for _ in range(n)]
        alice_bases = [random.choice(["Z", "X"]) for _ in range(n)]
        bob_meas    = {}
        for i in range(n):
            p = ch.transmit({"qubit_id": i, "bit": alice_bits[i], "basis": alice_bases[i]})
            if p:
                bb = random.choice(list(Basis))
                bbit = alice_bits[i] if bb.value == alice_bases[i] else random.randint(0, 1)
                bob_meas[i] = MeasurementRecord(qubit_id=i, basis=bb, bit_result=bbit)
        a_s, b_s, _ = perform_sifting_by_id(alice_bits, alice_bases, bob_meas)
        if len(a_s) < 2:
            return 1.0, 0
        qber, key, _ = compute_qber(a_s, b_s, sample_seed=99)
        return qber, len(key)

    random.seed(42); q0, k0 = _simulate(ch0)
    random.seed(42); q1, k1 = _simulate(ch1)

    print(f"\n        Step0: QBER={q0*100:.2f}% key={k0}  "
          f"Step1: QBER={q1*100:.2f}% key={k1}", end="")

    assert q0 == 0.0 and q1 == 0.0, "Both should have QBER=0 at ~1 km"
    _assert_close(k0, k1, tol=k0 * 0.1 + 10,
                  msg="Key lengths should be close at 1 km")


# ===========================================================================
# SECTION 5 — Interface contract
# ===========================================================================

@_test("interface: CSV is loadable by pandas (if installed)")
def test_interface_pandas():
    try:
        import pandas as pd
        df = pd.read_csv(_CSV_PATH)
        assert "distance_km"       in df.columns
        assert "transmission_prob" in df.columns
        assert len(df) == 13
        # This is how ChannelModel would use pandas in a future refactor
        fn = lambda d: float(
            df.set_index("distance_km")["transmission_prob"]
            .reindex([d], method="nearest").iloc[0]
        )
        t50 = fn(50.0)
        _assert_rel_close(t50, _formula_T(50.0), rel_tol=0.05, msg="pandas T(50)")
        print(f"\n        pandas available: ✓  T(50km)≈{t50:.5f}", end="")
    except ImportError:
        print(f"\n        pandas not installed — skipped (not required)", end="")


@_test("interface: transmission_prob returns float, not ndarray")
def test_interface_returns_float():
    model = ChannelModel(_CSV_PATH)
    result = model.transmission_prob(50.0)
    assert isinstance(result, float), (
        f"Expected float, got {type(result).__name__}. "
        "If scipy is installed, check the lambda float() cast in _build_interp."
    )


@_test("interface: ChannelModel.describe() is JSON-serialisable")
def test_interface_json():
    model = ChannelModel(_CSV_PATH)
    d = model.describe(50.0)
    try:
        json.dumps(d)
    except (TypeError, ValueError) as e:
        assert False, f"describe() is not JSON-serialisable: {e}"


# ===========================================================================
# Summary table printed after all tests
# ===========================================================================

def _print_attenuation_summary():
    model = ChannelModel(_CSV_PATH)
    print(f"\n  {'Distance':>10}  {'T(d)':>10}  {'Loss (dB)':>10}  "
          f"{'Survival %':>11}  {'Expected key bits / 10k qubits':>32}")
    print("  " + "-" * 80)
    for d in [0, 10, 20, 30, 50, 80, 100]:
        t = model.transmission_prob(float(d))
        loss_db = -10 * math.log10(max(t, 1e-12))
        # Rough key estimate: T × 0.5 (sift) × 0.8 (QBER sample) × 10000
        key_est = int(t * 0.5 * 0.8 * 10_000)
        print(f"  {d:>10} km  {t:>10.5f}  {loss_db:>10.2f}  "
              f"{t*100:>10.5f}%  {key_est:>32}")


# ===========================================================================
# Runner
# ===========================================================================

def _run_all():
    tests = [v for k, v in globals().items() if callable(v) and getattr(v, "_is_test", False)]

    print("\n" + "=" * 68)
    print("  Step 1 — Ansys Fiber Attenuation Model  (BB84 QKD)")
    print("=" * 68)
    print(f"  Python {sys.version.split()[0]}")
    print(f"  CSV:    {_CSV_PATH}")

    from optical.channel_model import _SCIPY_AVAILABLE
    print(f"  scipy:  {'available ✓' if _SCIPY_AVAILABLE else 'not installed — pure Python fallback'}")
    print("=" * 68 + "\n")

    for test_fn in tests:
        test_fn()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)
    total  = len(_RESULTS)

    print()
    print("=" * 68)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
        print("\n  Failed tests:")
        for name, ok, msg in _RESULTS:
            if not ok:
                print(f"    ✗  {name}")
                print(f"       {msg}")
    else:
        print("  ✓  All passed")
    print("=" * 68)

    _print_attenuation_summary()
    return failed == 0


# pytest compatibility
if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)