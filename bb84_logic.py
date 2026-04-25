import random
import hashlib
from typing import Optional
from models import Basis, QubitMeasurement

QBER_THRESHOLD = 0.11
SAMPLE_FRACTION = 0.20


def compute_qber(
    alice_sifted: list[int],
    bob_sifted: list[int],
    sample_fraction: float = SAMPLE_FRACTION,
    sample_seed: Optional[int] = None,
) -> tuple[float, list[int], list[int]]:
   
    n = min(len(alice_sifted), len(bob_sifted))
    if n == 0:
        return 1.0, [], []

    rng = random.Random(sample_seed)
    n_sample = max(1, int(n * sample_fraction))
    sample_indices = set(rng.sample(range(n), n_sample))

    errors = sum(
        1 for i in sample_indices
        if alice_sifted[i] != bob_sifted[i]
    )
    qber = errors / n_sample

    alice_key = [b for i, b in enumerate(alice_sifted) if i not in sample_indices]
    bob_key   = [b for i, b in enumerate(bob_sifted)   if i not in sample_indices]

    return qber, alice_key, bob_key


def perform_sifting(
    alice_bits: list[int],
    alice_bases: list[Basis],
    measurements: list[QubitMeasurement],
) -> tuple[list[int], list[int]]:

    alice_sifted = []
    bob_sifted   = []

    alice_bases_map = {i: basis for i, basis in enumerate(alice_bases)}

    for m in measurements:
        qid = m.qubit_id
        if qid not in alice_bases_map:
            continue
        if alice_bases_map[qid] == m.basis:
            alice_sifted.append(alice_bits[qid])
            bob_sifted.append(m.bit_res)

    return alice_sifted, bob_sifted
