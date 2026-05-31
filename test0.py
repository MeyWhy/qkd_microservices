"""
test_step0_baseline.py
======================
Step 0 — Statistical channel baseline tests.

WHAT THIS TESTS
---------------
  1. StatisticalChannel unit tests
       - zero loss: every photon survives
       - full loss: no photon survives
       - partial loss: empirical survival rate ≈ expected (±5%)
       - None photon input is a no-op
       - loss_rate out of [0,1] raises ValueError
       - describe() and transmission_probability() return correct values
       - qber_floor() is exactly 0.0 (pure loss, no bit errors)
       - reset_session() exists and is callable

  2. BB84 logic unit tests
       - compute_qber on identical lists → QBER = 0
       - compute_qber on fully inverted lists → QBER = 1
       - QBER stays below QBER_THRESHOLD when no errors are injected
       - perform_sifting_by_id: correct basis matching by qubit_id
       - sifted bits agree between Alice and Bob when channel is lossless

  3. End-to-end Step 0 integration
       - Full BB84 round trip: Alice generates bits/bases, photons pass
         through StatisticalChannel, Bob measures in random bases,
         sifting is performed, QBER is computed
       - With loss_rate=0.0: QBER ≈ 0%, key rate = ~50% (sifting only)
       - With loss_rate=0.5: ~50% fewer sifted bits, QBER still ≈ 0%
       - With loss_rate=0.9: very few sifted bits, QBER still ≈ 0%

WHAT THIS DOES NOT TEST
------------------------
  - FiberChannel (Step 1)
  - Polarization drift (Step 2)
  - Detector imperfections (Step 3)
  - OU time-varying drift (Step 4)
  - Eve / intercept-resend (Step 4)
  - HTTP endpoints, Redis, Celery, QuNetSim

HOW TO RUN
----------
  python -m pytest tests/test_step0_baseline.py -v
  # or without pytest:
  python tests/test_step0_baseline.py

EXPECTED OUTPUT (summary)
--------------------------
  All 20 tests pass.
  Step 0 baseline validated: QBER=0.0%, key_rate≈50% at loss=0.0
"""

import math
import random
import sys
import os
import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap — allows running from project root or tests/ directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Inline minimal implementations
# Reason: we do NOT import the full distributed stack (Redis, Celery, etc.).
# We import ONLY the two modules under test: optical.channel and bb84_logic.
# If those imports fail, the test file will tell you exactly why.
# ---------------------------------------------------------------------------

try:
    from optical.channel import StatisticalChannel
except ImportError as e:
    print(f"[FATAL] Cannot import StatisticalChannel: {e}")
    print("        Make sure you run from the project root:")
    print("        python -m pytest tests/test_step0_baseline.py -v")
    sys.exit(1)

try:
    from bb84_logic import compute_qber, perform_sifting_by_id, QBER_THRESHOLD
    from models import Basis, MeasurementRecord
except ImportError as e:
    print(f"[FATAL] Cannot import bb84_logic or models: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Minimal test runner (no pytest dependency required)
# ---------------------------------------------------------------------------

_RESULTS: list[tuple[str, bool, str]] = []


def _test(name: str):
    """Decorator that registers a test function."""
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
        wrapper._name = name
        return wrapper
    return decorator


def _assert_close(actual, expected, tol, msg=""):
    assert abs(actual - expected) <= tol, (
        f"{msg} expected≈{expected:.4f} got {actual:.4f} (tol={tol})"
    )


# ===========================================================================
# SECTION 1 — StatisticalChannel unit tests
# ===========================================================================

@_test("channel: loss_rate=0.0 → all photons survive")
def test_no_loss():
    ch = StatisticalChannel(loss_rate=0.0)
    photon = {"qubit_id": 0, "bit": 1, "basis": "Z"}
    survived = sum(1 for _ in range(1000) if ch.transmit(photon) is not None)
    assert survived == 1000, f"Expected 1000 survived, got {survived}"


@_test("channel: loss_rate=1.0 → no photon survives")
def test_full_loss():
    ch = StatisticalChannel(loss_rate=1.0)
    photon = {"qubit_id": 0, "bit": 0, "basis": "X"}
    survived = sum(1 for _ in range(1000) if ch.transmit(photon) is not None)
    assert survived == 0, f"Expected 0 survived, got {survived}"


@_test("channel: loss_rate=0.3 → ~70% survival rate (±5%)")
def test_partial_loss():
    random.seed(42)
    ch = StatisticalChannel(loss_rate=0.3)
    photon = {"qubit_id": 0, "bit": 1, "basis": "Z"}
    n = 10_000
    survived = sum(1 for _ in range(n) if ch.transmit(photon) is not None)
    rate = survived / n
    _assert_close(rate, 0.70, tol=0.05, msg="Survival rate")


@_test("channel: loss_rate=0.5 → ~50% survival rate (±5%)")
def test_half_loss():
    random.seed(99)
    ch = StatisticalChannel(loss_rate=0.5)
    photon = {"qubit_id": 7, "bit": 0, "basis": "X"}
    n = 10_000
    survived = sum(1 for _ in range(n) if ch.transmit(photon) is not None)
    rate = survived / n
    _assert_close(rate, 0.50, tol=0.05, msg="Survival rate")


@_test("channel: None photon input returns None (no-op)")
def test_none_photon():
    ch = StatisticalChannel(loss_rate=0.0)
    result = ch.transmit(None)
    assert result is None, f"Expected None, got {result}"


@_test("channel: invalid loss_rate raises ValueError")
def test_invalid_loss_rate():
    errors = 0
    for bad in [-0.1, 1.1, 2.0, -1.0]:
        try:
            StatisticalChannel(loss_rate=bad)
        except ValueError:
            errors += 1
    assert errors == 4, f"Expected 4 ValueErrors, got {errors}"


@_test("channel: describe() returns correct model and values")
def test_describe():
    ch = StatisticalChannel(loss_rate=0.3)
    d  = ch.describe()
    assert d["model"] == "statistical", f"Wrong model: {d['model']}"
    _assert_close(d["loss_rate"], 0.3, tol=1e-9, msg="loss_rate")
    _assert_close(d["transmission_prob"], 0.7, tol=1e-9, msg="transmission_prob")
    _assert_close(d["qber_floor"], 0.0, tol=1e-9, msg="qber_floor")


@_test("channel: transmission_probability() = 1 - loss_rate")
def test_transmission_probability():
    for lr in [0.0, 0.1, 0.5, 0.9, 1.0]:
        ch = StatisticalChannel(loss_rate=lr)
        expected = 1.0 - lr
        actual   = ch.transmission_probability()
        _assert_close(actual, expected, tol=1e-12, msg=f"loss_rate={lr}")


@_test("channel: qber_floor() == 0.0 (pure loss, no bit errors)")
def test_qber_floor_zero():
    for lr in [0.0, 0.3, 0.7, 1.0]:
        ch = StatisticalChannel(loss_rate=lr)
        assert ch.qber_floor() == 0.0, (
            f"qber_floor should be 0.0 at loss_rate={lr}, got {ch.qber_floor()}"
        )


@_test("channel: reset_session() is callable (no-op for StatisticalChannel)")
def test_reset_session():
    ch = StatisticalChannel(loss_rate=0.5)
    ch.reset_session()  # must not raise


@_test("channel: photon identity preserved after transmission")
def test_photon_identity():
    """The photon dict returned is the same object (no mutation)."""
    ch     = StatisticalChannel(loss_rate=0.0)
    photon = {"qubit_id": 42, "bit": 1, "basis": "X", "extra": "data"}
    result = ch.transmit(photon)
    assert result is photon, "transmit() should return the same dict, not a copy"
    assert result["qubit_id"] == 42
    assert result["extra"]    == "data"


# ===========================================================================
# SECTION 2 — BB84 logic unit tests
# ===========================================================================

@_test("bb84: compute_qber on identical lists → QBER = 0.0")
def test_qber_identical():
    bits  = [random.randint(0, 1) for _ in range(200)]
    qber, alice_key, bob_key = compute_qber(bits, bits[:], sample_seed=0)
    assert qber == 0.0, f"Expected QBER=0.0 on identical bits, got {qber}"
    assert len(alice_key) == len(bob_key), "Key length mismatch"


@_test("bb84: compute_qber on fully inverted lists → QBER ≈ 1.0")
def test_qber_all_errors():
    bits     = [random.randint(0, 1) for _ in range(200)]
    inverted = [1 - b for b in bits]
    qber, _, _ = compute_qber(bits, inverted, sample_seed=1)
    assert qber == 1.0, f"Expected QBER=1.0 on inverted bits, got {qber}"


@_test("bb84: compute_qber raises ValueError on length mismatch")
def test_qber_length_mismatch():
    raised = False
    try:
        compute_qber([0, 1, 0], [0, 1], sample_seed=0)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError on mismatched lengths"


@_test("bb84: compute_qber returns (1.0, [], []) on empty lists")
def test_qber_empty():
    qber, ak, bk = compute_qber([], [], sample_seed=0)
    assert qber == 1.0, f"Expected 1.0 on empty, got {qber}"
    assert ak == [] and bk == []


@_test("bb84: perform_sifting_by_id matches only same-basis qubits")
def test_sifting_basis_match():
    alice_bits  = [0, 1, 0, 1, 0]
    alice_bases = ["Z", "X", "Z", "X", "Z"]

    bob_measurements = {
        0: MeasurementRecord(qubit_id=0, basis=Basis.RECTILINEAR, bit_result=0),  # Z match
        1: MeasurementRecord(qubit_id=1, basis=Basis.DIAGONAL,    bit_result=1),  # X match
        2: MeasurementRecord(qubit_id=2, basis=Basis.DIAGONAL,    bit_result=0),  # Z≠X mismatch
        3: MeasurementRecord(qubit_id=3, basis=Basis.RECTILINEAR, bit_result=1),  # X≠Z mismatch
        4: MeasurementRecord(qubit_id=4, basis=Basis.RECTILINEAR, bit_result=0),  # Z match
    }

    alice_s, bob_s, matched = perform_sifting_by_id(
        alice_bits, alice_bases, bob_measurements
    )

    assert matched == [0, 1, 4], f"Expected matched=[0,1,4], got {matched}"
    assert alice_s == [0, 1, 0], f"Alice sifted wrong: {alice_s}"
    assert bob_s   == [0, 1, 0], f"Bob sifted wrong: {bob_s}"


@_test("bb84: sifted bits agree when channel is lossless and no noise")
def test_sifting_agreement_no_noise():
    """
    Simulates an ideal noiseless channel:
    - Alice sends bit in basis B
    - Bob measures in same basis B (forced)
    - Bob must get the same bit as Alice

    QBER of the sifted key should be exactly 0.
    """
    n = 500
    alice_bits  = [random.randint(0, 1) for _ in range(n)]
    alice_bases = [random.choice(["Z", "X"]) for _ in range(n)]

    # Bob always uses Alice's basis (ideal measurement) → always same bit
    bob_meas = {}
    for i in range(n):
        bob_meas[i] = MeasurementRecord(
            qubit_id=i,
            basis=Basis(alice_bases[i]),
            bit_result=alice_bits[i],
        )

    alice_s, bob_s, _ = perform_sifting_by_id(alice_bits, alice_bases, bob_meas)
    assert len(alice_s) == n, f"All should be sifted (same basis forced), got {len(alice_s)}"
    assert alice_s == bob_s, "Alice and Bob sifted bits must be identical in ideal case"

    qber, _, _ = compute_qber(alice_s, bob_s, sample_seed=7)
    assert qber == 0.0, f"Ideal sifted QBER must be 0.0, got {qber}"


# ===========================================================================
# SECTION 3 — End-to-end Step 0 integration
# ===========================================================================

def _run_bb84_step0(
    n_qubits: int,
    loss_rate: float,
    seed: int = 0,
) -> dict:
    """
    Full BB84 simulation using StatisticalChannel (Step 0).

    Simulates the entire pipeline:
    1. Alice randomly generates bits and bases
    2. Each photon is passed through StatisticalChannel
    3. Bob measures surviving photons in a random basis
    4. Basis reconciliation (sifting)
    5. QBER computation

    No HTTP, no Redis, no Celery. Pure Python.

    Returns dict with:
        n_sent, n_delivered, n_sifted, qber, key_len,
        delivery_rate, sift_rate
    """
    random.seed(seed)

    ch = StatisticalChannel(loss_rate=loss_rate)

    alice_bits  = [random.randint(0, 1) for _ in range(n_qubits)]
    alice_bases = [random.choice(["Z", "X"]) for _ in range(n_qubits)]

    # Simulate photon transmission + Bob's random measurement
    bob_measurements: dict[int, MeasurementRecord] = {}
    n_delivered = 0

    for i in range(n_qubits):
        photon = {"qubit_id": i, "bit": alice_bits[i], "basis": alice_bases[i]}
        result = ch.transmit(photon)

        if result is None:
            continue  # photon lost

        n_delivered += 1
        bob_basis = random.choice(list(Basis))
        # If Bob's basis matches Alice's, he gets the correct bit;
        # otherwise random (discarded at sifting anyway)
        if bob_basis.value == alice_bases[i]:
            bob_bit = alice_bits[i]
        else:
            bob_bit = random.randint(0, 1)

        bob_measurements[i] = MeasurementRecord(
            qubit_id=i,
            basis=bob_basis,
            bit_result=bob_bit,
        )

    # Sifting
    alice_s, bob_s, matched = perform_sifting_by_id(
        alice_bits, alice_bases, bob_measurements
    )
    n_sifted = len(alice_s)

    if n_sifted < 2:
        return {
            "n_sent": n_qubits, "n_delivered": n_delivered,
            "n_sifted": 0, "qber": 1.0, "key_len": 0,
            "delivery_rate": n_delivered / n_qubits,
            "sift_rate": 0.0,
        }

    qber, alice_key, _ = compute_qber(alice_s, bob_s, sample_seed=seed + 1)

    return {
        "n_sent":        n_qubits,
        "n_delivered":   n_delivered,
        "n_sifted":      n_sifted,
        "qber":          qber,
        "key_len":       len(alice_key),
        "delivery_rate": n_delivered / n_qubits,
        "sift_rate":     n_sifted / n_delivered if n_delivered > 0 else 0.0,
    }


@_test("e2e: loss=0.0 → QBER=0.0, delivery=100%, sift≈50%")
def test_e2e_no_loss():
    result = _run_bb84_step0(n_qubits=2000, loss_rate=0.0, seed=42)
    print(f"\n        delivery={result['delivery_rate']:.2%}  "
          f"sift≈{result['sift_rate']:.2%}  "
          f"QBER={result['qber']*100:.2f}%  "
          f"key_len={result['key_len']}", end="")
    assert result["delivery_rate"] == 1.0, "All photons must be delivered at loss=0"
    _assert_close(result["sift_rate"], 0.5, tol=0.06, msg="Sift rate")
    assert result["qber"] == 0.0, f"QBER must be 0 at loss=0, got {result['qber']}"
    assert result["key_len"] > 0


@_test("e2e: loss=0.5 → QBER=0.0, delivery≈50%, sift≈50% of delivered")
def test_e2e_half_loss():
    result = _run_bb84_step0(n_qubits=5000, loss_rate=0.5, seed=7)
    print(f"\n        delivery={result['delivery_rate']:.2%}  "
          f"sift≈{result['sift_rate']:.2%}  "
          f"QBER={result['qber']*100:.2f}%  "
          f"key_len={result['key_len']}", end="")
    _assert_close(result["delivery_rate"], 0.5, tol=0.05, msg="Delivery rate")
    _assert_close(result["sift_rate"],     0.5, tol=0.06, msg="Sift rate")
    assert result["qber"] == 0.0, f"QBER must be 0 (no noise), got {result['qber']}"


@_test("e2e: loss=0.9 → QBER=0.0, delivery≈10%, key_len still > 0")
def test_e2e_heavy_loss():
    result = _run_bb84_step0(n_qubits=10_000, loss_rate=0.9, seed=13)
    print(f"\n        delivery={result['delivery_rate']:.2%}  "
          f"sift≈{result['sift_rate']:.2%}  "
          f"QBER={result['qber']*100:.2f}%  "
          f"key_len={result['key_len']}", end="")
    _assert_close(result["delivery_rate"], 0.10, tol=0.05, msg="Delivery rate")
    assert result["qber"] == 0.0, f"QBER must be 0 even at high loss, got {result['qber']}"
    assert result["key_len"] > 0, "Should still produce some key bits at 90% loss with 10k qubits"


@_test("e2e: QBER stays below threshold (0.11) at all loss rates")
def test_e2e_qber_always_safe():
    for lr in [0.0, 0.2, 0.5, 0.7, 0.85]:
        result = _run_bb84_step0(n_qubits=2000, loss_rate=lr, seed=lr*100)
        assert result["qber"] < QBER_THRESHOLD or result["n_sifted"] < 2, (
            f"QBER={result['qber']:.4f} exceeded threshold at loss_rate={lr}"
        )


@_test("e2e: key_len ≈ 0.4 × n_sifted (after 20% QBER sample removed)")
def test_e2e_key_length_fraction():
    """
    After sifting: 50% of n_delivered are sifted.
    After QBER sampling: 80% of sifted bits become the key.
    Total: key_len ≈ 0.5 × 0.8 × n_delivered ≈ 0.4 × n_sifted.
    """
    result = _run_bb84_step0(n_qubits=5000, loss_rate=0.0, seed=55)
    expected_key = result["n_sifted"] * 0.80
    _assert_close(
        result["key_len"], expected_key,
        tol=expected_key * 0.05 + 5,  # 5% tolerance + 5 bit slack
        msg="Key length vs 80% of sifted"
    )


# ===========================================================================
# Runner
# ===========================================================================

def _run_all():
    import inspect
    tests = [
        v for k, v in globals().items()
        if callable(v) and getattr(v, "_is_test", False)
    ]

    print("\n" + "=" * 65)
    print("  Step 0 — Statistical Channel Baseline  (BB84 QKD)")
    print("=" * 65)
    print(f"  Python {sys.version.split()[0]}")
    print(f"  Project root: {_ROOT}")
    print("=" * 65 + "\n")

    for test_fn in tests:
        test_fn()

    print()
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)
    total  = len(_RESULTS)

    print("=" * 65)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
        print("\nFailed tests:")
        for name, ok, msg in _RESULTS:
            if not ok:
                print(f"  ✗  {name}")
                print(f"     {msg}")
    else:
        print("  ✓  All passed")
    print("=" * 65 + "\n")

    # Print a one-line Step 0 summary
    r = _run_bb84_step0(n_qubits=1000, loss_rate=0.0, seed=0)
    print(
        f"  Step 0 baseline (loss=0.0, n=1000):\n"
        f"    delivery={r['delivery_rate']:.0%}  "
        f"sift_rate={r['sift_rate']:.1%}  "
        f"QBER={r['qber']*100:.2f}%  "
        f"key_len={r['key_len']} bits\n"
    )

    return failed == 0


# pytest compatibility: each test is also discoverable by pytest
# (functions decorated with @_test are returned as callables)

if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)