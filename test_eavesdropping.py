"""
Test 3  Eavesdropping scenario
=================================
Tests the intercept-resend attack (Eve node) in two sub-scenarios:

A) SEQUENTIAL  one session at a time.
   For every (alice, bob, n_qubits, batch_size) combination, two sessions
   are run back-to-back:
     1. Clean run  (no Eve)  -> expect: status=done,    QBER≈0
     2. Eve run   (eve-1)   -> expect: status=aborted,  QBER≈0.25, QBER_TOO_HIGH
   Both rows share a 'pair_id' so they can be correlated in analysis.

B) CONCURRENT  3 sessions simultaneously (one per QKDL).
   Same pairs as sequential but all three fire at once.
   Two groups per n_qubits/batch_size combination:
     1. Clean group  (no Eve)
     2. Eve group    (all three pairs intercepted by eve-1)
   A single Eve node handles all three sessions  each QKDL independently
   calls POST /intercept/{session_id} on the Eve node, which registers
   per-QKDL. Concurrency is safe.

After each Eve session, the test fetches Eve's stats from
GET /session/{session_id}/stats on the Eve node to record:
  - n_intercepted   : how many qubits Eve saw
  - basis_match_rate: fraction where Eve's basis matched Alice's (~0.5 expected)
  - eve_registered  : whether Eve successfully registered as MITM

These extra columns make the CSV self-contained for thesis plots.

Usage:
    python tests/test_eavesdropping.py [--dry-run] [--skip-from N]
                                       [--sequential-only | --concurrent-only]
                                       [--output PATH]
"""
import argparse
import concurrent.futures
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

#Configuration 

ALICE_URLS = {
    "alice-1": os.getenv("ALICE_1_URL", "http://localhost:8001"),
    "alice-2": os.getenv("ALICE_2_URL", "http://localhost:8102"),
    "alice-3": os.getenv("ALICE_3_URL", "http://localhost:8103"),
}
EVE_URL  = os.getenv("EVE_URL",  "http://localhost:8010")
KME_URL  = os.getenv("KME_URL",  "http://localhost:8000")
EVE_LABEL = os.getenv("EVE_LABEL", "eve-1")

QKDL_URLS = [
    os.getenv("QKDL_1_URL", "http://localhost:8003"),
    os.getenv("QKDL_2_URL", "http://localhost:8013"),
    os.getenv("QKDL_3_URL", "http://localhost:8023"),
]

#Matched pairs only  eavesdropping is most interesting on intended pairs
PAIRS = [
    ("alice-1", "bob-1"),
    ("alice-2", "bob-2"),
    ("alice-3", "bob-3"),
]

#Larger n_qubits gives tighter QBER estimates (variance ~ 1/sqrt(n_sample))
#Skip very small n  QBER sample too small to be meaningful
N_QUBITS_LIST = [64, 128, 200, 512, 1024]
BATCH_SIZES   = [10, 20]

POLL_INTERVAL        = 2.0
INTER_SESSION_DELAY  = 4.0   #between sequential sessions
INTER_GROUP_DELAY    = 4.0   #between concurrent groups
MAX_WAIT_PER_SESSION = int(os.getenv("MAX_WAIT_S", "1200"))

#Theoretical QBER from intercept-resend: 0.25
#(Eve wrong basis 50% -> random bit -> 50% error on those -> 0.5*0.5=0.25)
THEORETICAL_EVE_QBER = 0.25

CSV_COLUMNS = [
    #Identity
    "run_index", "pair_id", "scenario",          #scenario: clean | eve
    "run_type",                                   #sequential | concurrent
    "group_index", "group_type",
    "alice_label", "bob_label",
    "batch_size", "n_qubits",
    #Session outcome
    "status", "latency_s",
    "session_id", "n_delivered", "n_sifted", "qber",
    "key_final", "error_message",
    "progress_pct", "phase_label",
    #Fixed params
    "loss_rate", "interceptor_label", "retry_enabled",
    "qkdl_url", "intercepted",
    #Eve-specific
    "eve_registered", "n_intercepted",
    "basis_match_rate", "theoretical_eve_qber",
    "eve_detection_correct",                      #True if aborted when eve present
    #Concurrent only
    "group_wall_time_s",
    "timestamp",
]


#Preflight 

def preflight_check(need_eve: bool = True) -> bool:
    print("\n[preflight] Checking services...")
    ok = True
    checks = [("KME", KME_URL)] + list(ALICE_URLS.items())
    if need_eve:
        checks.append(("Eve", EVE_URL))
    for name, url in checks:
        try:
            httpx.get(f"{url}/health", timeout=5.0).raise_for_status()
            print(f"  {name:<10} {url}  OK")
        except Exception as e:
            print(f"  {name:<10} {url}  FAIL  {e}")
            ok = False
    for url in QKDL_URLS:
        try:
            httpx.get(f"{url}/health", timeout=5.0).raise_for_status()
            print(f"  QKDL      {url}  OK")
        except Exception as e:
            print(f"  QKDL      {url}  WARN  {e}")
    if not ok:
        print("\n[preflight] FAILED.\n")
    else:
        print("[preflight] All services reachable.\n")
    return ok


#Eve stats helper 

def fetch_eve_stats(session_id: str) -> dict:
    """
    GET /session/{session_id}/stats from the Eve node.
    Returns {} if Eve node unreachable or session not found.
    """
    try:
        resp = httpx.get(
            f"{EVE_URL}/session/{session_id}/stats",
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


#Session helpers 

def start_session(alice_label: str, bob_label: str,
                  n_qubits: int, batch_size: int,
                  interceptor_label: str = None) -> dict:
    url  = ALICE_URLS[alice_label]
    params = {
        "receiver_label": bob_label,
        "n_qubits":       n_qubits,
        "batch_size":     batch_size,
        "loss_rate":      0.0,
        "retry_enabled":  False,
    }
    if interceptor_label:
        params["interceptor_label"] = interceptor_label
    resp = httpx.post(f"{url}/start", params=params, timeout=30.0)
    resp.raise_for_status()
    body = resp.json()
    body["_alice_label"] = alice_label
    body["_bob_label"]   = bob_label
    return body


def poll_until_done(session_id: str, max_wait: int) -> tuple[dict, bool]:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = httpx.get(f"{KME_URL}/sessions/{session_id}", timeout=10.0)
            r.raise_for_status()
            data = r.json()
            if data.get("status") in ("done", "aborted"):
                return data, False
        except Exception as e:
            print(f"    [poll {session_id[:8]}] {e}")
        time.sleep(POLL_INTERVAL)
    try:
        data = httpx.get(f"{KME_URL}/sessions/{session_id}", timeout=10.0).json()
    except Exception:
        data = {"status": "timeout", "session_id": session_id}
    return data, True


def cleanup_after_timeout(session_id: str, qkdl_url: str) -> None:
    try:
        httpx.delete(f"{KME_URL}/sessions/{session_id}", timeout=10.0)
    except Exception as e:
        print(f"    [cleanup] KME: {e}")
    if qkdl_url:
        try:
            httpx.post(f"{qkdl_url}/network/reset", timeout=10.0)
        except Exception as e:
            print(f"    [cleanup] QKDL: {e}")
    time.sleep(3.0)


def run_one_session_threaded(alice_label: str, bob_label: str,
                             n_qubits: int, batch_size: int,
                             interceptor_label: str = None) -> dict:
    """Used in concurrent groups. Runs start + poll in a thread."""
    try:
        start = start_session(
            alice_label, bob_label, n_qubits, batch_size, interceptor_label
        )
        session_id = start["session_id"]
        qkdl_url   = start.get("qkdl_url", "")
    except Exception as e:
        return {
            "status": "start_error", "error_message": str(e),
            "session_id": "", "_alice_label": alice_label,
            "_bob_label": bob_label, "_qkdl_url": "",
            "_timed_out": False,
        }
    result, timed_out = poll_until_done(session_id, MAX_WAIT_PER_SESSION)
    result["_alice_label"] = alice_label
    result["_bob_label"]   = bob_label
    result["_qkdl_url"]    = qkdl_url
    result["_timed_out"]   = timed_out
    return result


#Row builders 

def _base_row(result: dict, timed_out: bool,
              alice: str, bob: str,
              n_qubits: int, batch_size: int,
              scenario: str, interceptor_label: str) -> dict:
    """Common fields shared between sequential and concurrent rows."""
    status = "timeout" if timed_out else result.get("status", "unknown")
    qber   = result.get("qber")

    #Fetch Eve stats if this was an intercepted session
    eve_stats         = {}
    eve_registered    = ""
    n_intercepted     = ""
    basis_match_rate  = ""
    if interceptor_label and result.get("session_id"):
        #Small delay to let Eve's collect_measurements finish
        time.sleep(1.0)
        eve_stats        = fetch_eve_stats(result["session_id"])
        eve_registered   = eve_stats.get("registered", "")
        n_intercepted    = eve_stats.get("n_intercepted", "")
        basis_match_rate = eve_stats.get("basis_match_rate", "")

    #Was Eve correctly detected?
    if scenario == "eve":
        eve_detection_correct = (status == "aborted" and
                                 result.get("error_message") == "QBER_TOO_HIGH")
    else:
        eve_detection_correct = ""   #not applicable for clean runs

    return {
        "alice_label":          alice,
        "bob_label":            bob,
        "batch_size":           batch_size,
        "n_qubits":             n_qubits,
        "scenario":             scenario,
        "status":               status,
        "latency_s":            result.get("elapsed_s", ""),
        "session_id":           result.get("session_id", ""),
        "n_delivered":          result.get("n_delivered", ""),
        "n_sifted":             result.get("n_sifted", ""),
        "qber":                 qber if qber is not None else "",
        "key_final":            result.get("key_final", ""),
        "error_message":        result.get("error_message", ""),
        "progress_pct":         result.get("progress_pct", ""),
        "phase_label":          result.get("phase_label", ""),
        "loss_rate":            0.0,
        "interceptor_label":    interceptor_label or "",
        "retry_enabled":        False,
        "qkdl_url":             result.get("qkdl_url", ""),
        "intercepted":          result.get("intercepted", False),
        "eve_registered":       eve_registered,
        "n_intercepted":        n_intercepted,
        "basis_match_rate":     basis_match_rate,
        "theoretical_eve_qber": THEORETICAL_EVE_QBER if interceptor_label else "",
        "eve_detection_correct": eve_detection_correct,
        "timestamp":            datetime.utcnow().isoformat(),
    }


def append_rows(csv_path: Path, rows: list[dict], write_header: bool) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            #Fill any missing columns with empty string
            full_row = {col: row.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(full_row)


def print_session_line(prefix: str, scenario: str, alice: str, bob: str,
                       status: str, result: dict, n_qubits: int) -> None:
    qber  = result.get("qber")
    qber_s = f"QBER={qber*100:.1f}%  " if qber is not None else ""
    err   = result.get("error_message", "")
    lat   = result.get("elapsed_s", "")
    mark  = "✓" if (
        (scenario == "clean" and status == "done") or
        (scenario == "eve"   and status == "aborted" and err == "QBER_TOO_HIGH")
    ) else "✗"
    print(f"  {mark} [{prefix}] {alice}->{bob}  {scenario:<5}  "
          f"{status.upper():<8} lat={lat}s  "
          f"sifted={result.get('n_sifted','?')}/{n_qubits}  "
          f"{qber_s}"
          + (f"[{err}]" if err else ""))


#Sequential sub-scenario 

def run_sequential(csv_path: Path, write_header: bool,
                   skip_from: int = 0) -> tuple[bool, int, int, int]:
    """
    Run all sequential eavesdropping tests.
    Returns (write_header, passed, failed, run_index_end).
    """
    print(f"\n{''*65}")
    print(f"  SUB-SCENARIO A  Sequential")
    print(f"{''*65}\n")

    #Build ordered list of (pair_id, alice, bob, n, b, scenario)
    cases  = []
    idx    = 1
    pair_n = 0
    for alice, bob in PAIRS:
        for n in N_QUBITS_LIST:
            for b in BATCH_SIZES:
                pair_n += 1
                for scenario, iceptor in [("clean", None), ("eve", EVE_LABEL)]:
                    cases.append({
                        "run_index":   idx,
                        "pair_id":     pair_n,
                        "alice_label": alice,
                        "bob_label":   bob,
                        "n_qubits":    n,
                        "batch_size":  b,
                        "scenario":    scenario,
                        "interceptor": iceptor,
                    })
                    idx += 1

    total  = len(cases)
    passed = failed = 0

    if skip_from:
        cases = [c for c in cases if c["run_index"] >= skip_from]

    for case in cases:
        alice  = case["alice_label"]
        bob    = case["bob_label"]
        n      = case["n_qubits"]
        b      = case["batch_size"]
        sc     = case["scenario"]
        iceptor = case["interceptor"]
        ridx   = case["run_index"]

        print(f"[{ridx:>3}/{total}] {alice}->{bob}  n={n:<5} batch={b}  "
              f"scenario={sc:<5}  ", end="", flush=True)

        try:
            start_resp = start_session(alice, bob, n, b, iceptor)
            session_id = start_resp["session_id"]
            qkdl_url   = start_resp.get("qkdl_url", "")
        except Exception as e:
            print(f"START_ERROR  {e}")
            row = {
                "run_index": ridx, "pair_id": case["pair_id"],
                "run_type": "sequential", "group_index": "", "group_type": "",
                **_base_row(
                    {"status": "start_error", "error_message": str(e)},
                    False, alice, bob, n, b, sc, iceptor or "",
                ),
            }
            append_rows(csv_path, [row], write_header)
            write_header = False
            failed += 1
            time.sleep(INTER_SESSION_DELAY)
            continue

        result, timed_out = poll_until_done(session_id, MAX_WAIT_PER_SESSION)
        status = "timeout" if timed_out else result.get("status", "unknown")

        qber  = result.get("qber")
        qber_s = f"QBER={qber*100:.1f}%  " if qber is not None else ""
        err   = result.get("error_message", "")
        mark  = "✓" if (
            (sc == "clean" and status == "done") or
            (sc == "eve"   and status == "aborted" and err == "QBER_TOO_HIGH")
        ) else "✗"
        print(f"{mark} {status.upper():<8} lat={result.get('elapsed_s','')}s  "
              f"sifted={result.get('n_sifted','?')}/{n}  {qber_s}"
              + (f"[{err}]" if err else ""))

        if (sc == "clean" and status == "done") or \
           (sc == "eve"   and status == "aborted" and err == "QBER_TOO_HIGH"):
            passed += 1
        else:
            failed += 1

        if timed_out:
            cleanup_after_timeout(session_id, qkdl_url)

        base = _base_row(result, timed_out, alice, bob, n, b, sc, iceptor or "")
        row  = {
            "run_index":  ridx,
            "pair_id":    case["pair_id"],
            "run_type":   "sequential",
            "group_index": "",
            "group_type":  "",
            "group_wall_time_s": "",
            **base,
        }
        append_rows(csv_path, [row], write_header)
        write_header = False
        time.sleep(INTER_SESSION_DELAY)

    return write_header, passed, failed, total


#Concurrent sub-scenario 

def run_concurrent(csv_path: Path, write_header: bool,
                   skip_from: int = 0) -> tuple[bool, int, int, int]:
    """
    Run all concurrent eavesdropping groups.
    Returns (write_header, passed_groups, failed_groups, total_groups).
    """
    print(f"\n{''*65}")
    print(f"  SUB-SCENARIO B  Concurrent (3 sessions simultaneously)")
    print(f"{''*65}\n")

    groups = []
    idx    = 1
    for n in N_QUBITS_LIST:
        for b in BATCH_SIZES:
            for scenario, iceptor in [("clean", None), ("eve", EVE_LABEL)]:
                groups.append({
                    "group_index": idx,
                    "group_type":  scenario,
                    "n_qubits":    n,
                    "batch_size":  b,
                    "scenario":    scenario,
                    "interceptor": iceptor,
                    "pairs":       PAIRS,
                })
                idx += 1

    total_groups   = len(groups)
    passed_groups  = 0
    failed_groups  = 0

    if skip_from:
        groups = [g for g in groups if g["group_index"] >= skip_from]

    for gi, group in enumerate(groups, 1):
        n      = group["n_qubits"]
        b      = group["batch_size"]
        sc     = group["scenario"]
        iceptor = group["interceptor"]
        pairs  = group["pairs"]

        print(f"\n[Group {group['group_index']:>3}/{total_groups}] "
              f"scenario={sc:<5}  n={n:<5}  batch={b}")
        for a, bob in pairs:
            print(f"    {a} -> {bob}"
                  + (f"  (interceptor={iceptor})" if iceptor else ""))

        t0 = time.time()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            futures = {
                pool.submit(
                    run_one_session_threaded, a, bob, n, b, iceptor
                ): (a, bob)
                for a, bob in pairs
            }
            for future in concurrent.futures.as_completed(futures):
                a, bob = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = {
                        "status": "thread_error", "error_message": str(e),
                        "session_id": "", "_alice_label": a, "_bob_label": bob,
                        "_qkdl_url": "", "_timed_out": False,
                    }
                results.append(res)

        group_wall_s = round(time.time() - t0, 3)
        group_ok     = True
        rows         = []

        for res in results:
            timed_out  = res.pop("_timed_out", False)
            alice      = res.pop("_alice_label", "?")
            bob        = res.pop("_bob_label",   "?")
            qkdl_url   = res.pop("_qkdl_url",   "")
            res["qkdl_url"] = qkdl_url

            status = "timeout" if timed_out else res.get("status", "unknown")
            err    = res.get("error_message", "")
            ok     = (
                (sc == "clean" and status == "done") or
                (sc == "eve"   and status == "aborted" and err == "QBER_TOO_HIGH")
            )
            mark = "✓" if ok else "✗"
            qber = res.get("qber")
            qber_s = f"QBER={qber*100:.1f}%  " if qber is not None else ""
            print(f"  {mark} {alice}->{bob}  {sc:<5}  {status.upper():<8}  "
                  f"lat={res.get('elapsed_s','')}s  "
                  f"sifted={res.get('n_sifted','?')}/{n}  {qber_s}"
                  + (f"[{err}]" if err else ""))

            if not ok:
                group_ok = False
            if timed_out:
                cleanup_after_timeout(res.get("session_id", ""), qkdl_url)

            base = _base_row(res, timed_out, alice, bob, n, b, sc, iceptor or "")
            row  = {
                "run_index":         "",
                "pair_id":           "",
                "run_type":          "concurrent",
                "group_index":       group["group_index"],
                "group_type":        sc,
                "group_wall_time_s": group_wall_s,
                **base,
            }
            rows.append(row)

        print(f"  Group wall time: {group_wall_s}s  "
              f"{'ALL OK' if group_ok else 'SOME FAILED'}")
        if group_ok:
            passed_groups += 1
        else:
            failed_groups += 1

        append_rows(csv_path, rows, write_header)
        write_header = False

        if gi < len(groups):
            session_ids = [
                res.get("session_id", "") for res in results
                if res.get("session_id")
            ]
            _wait_sessions_terminal(session_ids, timeout=60)
            time.sleep(2.0)

    return write_header, passed_groups, failed_groups, total_groups


def _wait_sessions_terminal(session_ids: list[str], timeout: int = 60) -> None:
    """
    Poll KME until all sessions are done/aborted or timeout is reached.
    Prevents the next group firing while alice nodes still hold an active guard.
    """
    if not session_ids:
        return
    deadline = time.time() + timeout
    pending  = set(session_ids)
    while pending and time.time() < deadline:
        still_pending = set()
        for sid in pending:
            try:
                r = httpx.get(f"{KME_URL}/sessions/{sid}", timeout=5.0)
                if r.status_code == 200:
                    if r.json().get("status") not in ("done", "aborted"):
                        still_pending.add(sid)
                #404 = already cleaned up = terminal
            except Exception:
                still_pending.add(sid)
        pending = still_pending
        if pending:
            time.sleep(1.0)


#Main 

def main():
    parser = argparse.ArgumentParser(description="BB84 eavesdropping test suite")
    parser.add_argument("--dry-run",          action="store_true")
    parser.add_argument("--skip-from",        type=int, default=0,
                        help="Skip run_index/group_index < N (resume)")
    parser.add_argument("--sequential-only",  action="store_true")
    parser.add_argument("--concurrent-only",  action="store_true")
    parser.add_argument("--output", default="results/eavesdropping_results.csv")
    args = parser.parse_args()

    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    seq_sessions  = len(PAIRS) * len(N_QUBITS_LIST) * len(BATCH_SIZES) * 2
    conc_groups   = len(N_QUBITS_LIST) * len(BATCH_SIZES) * 2
    conc_sessions = conc_groups * len(PAIRS)

    print(f"\n{'='*65}")
    print(f"  BB84 Eavesdropping Test Suite")
    print(f"  Eve node: {EVE_LABEL} @ {EVE_URL}")
    print(f"  Theoretical QBER with Eve: {THEORETICAL_EVE_QBER*100:.0f}%")
    if not args.concurrent_only:
        print(f"  Sequential: {seq_sessions} sessions")
    if not args.sequential_only:
        print(f"  Concurrent: {conc_groups} groups ({conc_sessions} sessions)")
    print(f"  Output: {csv_path}")
    print(f"{'='*65}\n")

    if args.dry_run:
        print("Sequential cases:")
        idx = 1
        for a, bob in PAIRS:
            for n in N_QUBITS_LIST:
                for b in BATCH_SIZES:
                    for sc in ["clean", "eve"]:
                        print(f"  [{idx:>3}] {a}->{bob}  n={n:<5} batch={b}  {sc}")
                        idx += 1
        print(f"\nConcurrent groups:")
        idx = 1
        for n in N_QUBITS_LIST:
            for b in BATCH_SIZES:
                for sc in ["clean", "eve"]:
                    pairs_str = "  ".join(f"{a}↔{bob}" for a, bob in PAIRS)
                    print(f"  [{idx:>3}] n={n:<5} batch={b}  {sc:<5}  {pairs_str}")
                    idx += 1
        return

    if not preflight_check(need_eve=True):
        sys.exit(1)

    write_header  = not csv_path.exists()
    t_suite_start = time.time()
    total_passed  = total_failed = 0

    if not args.concurrent_only:
        write_header, passed, failed, total = run_sequential(
            csv_path, write_header, skip_from=args.skip_from
        )
        total_passed += passed
        total_failed += failed
        print(f"\n  Sequential: {passed}/{total} expected outcomes  "
              f"({failed} unexpected)\n")

    if not args.sequential_only:
        write_header, passed, failed, total = run_concurrent(
            csv_path, write_header, skip_from=args.skip_from
        )
        total_passed += passed
        total_failed += failed
        print(f"\n  Concurrent: {passed}/{total} groups fully correct  "
              f"({failed} with unexpected outcomes)\n")

    elapsed = round(time.time() - t_suite_start, 1)
    print(f"\n{'='*65}")
    print(f"  DONE  passed={total_passed}  unexpected={total_failed}")
    print(f"  Wall time: {elapsed}s ({elapsed/60:.1f} min)")
    print(f"  Results: {csv_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()