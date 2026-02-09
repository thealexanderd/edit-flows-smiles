"""Alignment utilities for Edit Flows with epsilon-augmented space."""

from typing import Callable, List, Optional, Tuple
import random

EPS = "<EPS>"


def strip_epsilon(z: List[str]) -> List[str]:
    """Remove epsilon tokens while preserving order."""
    return [tok for tok in z if tok != EPS]


def build_uniform_halfhalf_alignment(
    x0_tokens: List[str],
    x1_tokens: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Construct paper-style Uniform X0 half/half alignment.

    For an x0 sequence of length L0:
    - first floor(L0/2) tokens are aligned as deletions
    - remaining ceil(L0/2) tokens are aligned as substitutions
    - any remaining x1 tokens are aligned as insertions

    Args:
        x0_tokens: Source tokens sampled from empirical token marginal.
        x1_tokens: Target tokens.

    Returns:
        (a0, a1) aligned lists with EPS placeholders.
    """
    L0 = len(x0_tokens)
    del_count = L0 // 2
    sub_count = L0 - del_count

    if len(x1_tokens) < sub_count:
        raise ValueError(
            f"Need len(x1_tokens) >= {sub_count} for half/half substitutions, got {len(x1_tokens)}"
        )

    a0: List[str] = []
    a1: List[str] = []

    # Delete half of x0.
    for i in range(del_count):
        a0.append(x0_tokens[i])
        a1.append(EPS)

    # Substitute the remaining half against the start of x1.
    j = 0
    for i in range(del_count, L0):
        a0.append(x0_tokens[i])
        a1.append(x1_tokens[j])
        j += 1

    # Insert the remaining x1 suffix.
    while j < len(x1_tokens):
        a0.append(EPS)
        a1.append(x1_tokens[j])
        j += 1

    return a0, a1


def make_alignment_fixed_N(
    a0: List[str],
    a1: List[str],
    N: int,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], List[str]]:
    """
    Pad aligned sequences (a0, a1) with (EPS, EPS) pairs to length N.

    Args:
        a0: Aligned sequence for x0 (length L)
        a1: Aligned sequence for x1 (length L)
        N: Desired aligned length (>= L)
        rng: Optional RNG for random insertion positions

    Returns:
        (z0, z1) of length N
    """
    if len(a0) != len(a1):
        raise ValueError("Aligned sequences must have the same length")
    if len(a0) > N:
        raise ValueError(f"Aligned length {len(a0)} exceeds N={N}")

    z0 = list(a0)
    z1 = list(a1)
    if len(z0) == N:
        return z0, z1

    pad_needed = N - len(z0)
    if rng is None:
        rng = random.Random()

    for _ in range(pad_needed):
        insert_pos = rng.randint(0, len(z0))
        z0.insert(insert_pos, EPS)
        z1.insert(insert_pos, EPS)

    return z0, z1


def sample_z_t(
    z0: List[str],
    z1: List[str],
    t: float,
    kappa: Callable[[float], float],
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Sample z_t by per-position mixture between z0 and z1.

    Args:
        z0: Aligned sequence at time 0
        z1: Aligned sequence at time 1
        t: Scalar in [0,1]
        kappa: Schedule function
        rng: Optional RNG

    Returns:
        z_t list
    """
    if len(z0) != len(z1):
        raise ValueError("z0 and z1 must have the same length")
    if rng is None:
        rng = random.Random()

    p = kappa(t)
    z_t = []
    for a, b in zip(z0, z1):
        if rng.random() < p:
            z_t.append(b)
        else:
            z_t.append(a)
    return z_t


def build_index_maps(z_t: List[str]) -> Tuple[List[Optional[int]], List[int]]:
    """
    Build maps from aligned indices to x_t indices and prefix counts.

    Returns:
        map_j_to_i: list of length N with x_t index or None
        prefix_counts: list of length N+1 with count of non-eps before j
    """
    map_j_to_i: List[Optional[int]] = [None] * len(z_t)
    prefix_counts: List[int] = [0] * (len(z_t) + 1)

    count = 0
    for j, tok in enumerate(z_t):
        prefix_counts[j] = count
        if tok != EPS:
            map_j_to_i[j] = count
            count += 1
    prefix_counts[len(z_t)] = count

    return map_j_to_i, prefix_counts


def extract_targets(
    z_t: List[str],
    z1: List[str],
) -> Tuple[List[int], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Extract delete/sub/insert targets mapped into x_t indices/gaps.

    Returns:
        delete_positions: list of x_t indices
        sub_pairs: list of (x_t index, target token)
        ins_pairs: list of (gap index, target token)
    """
    if len(z_t) != len(z1):
        raise ValueError("z_t and z1 must have the same length")

    map_j_to_i, prefix_counts = build_index_maps(z_t)

    delete_positions: List[int] = []
    sub_pairs: List[Tuple[int, str]] = []
    ins_pairs: List[Tuple[int, str]] = []

    for j, (tok_t, tok_1) in enumerate(zip(z_t, z1)):
        if tok_t != EPS and tok_1 == EPS:
            i = map_j_to_i[j]
            if i is not None:
                delete_positions.append(i)
        elif tok_t == EPS and tok_1 != EPS:
            gap = prefix_counts[j]
            ins_pairs.append((gap, tok_1))
        elif tok_t != EPS and tok_1 != EPS and tok_t != tok_1:
            i = map_j_to_i[j]
            if i is not None:
                sub_pairs.append((i, tok_1))

    return delete_positions, sub_pairs, ins_pairs
