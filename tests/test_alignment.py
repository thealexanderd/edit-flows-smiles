"""Tests for epsilon-alignment utilities."""

import random

from smiles_editflow.alignment import (
    EPS,
    strip_epsilon,
    make_alignment_fixed_N,
    sample_z_t,
    extract_targets,
    build_uniform_halfhalf_alignment,
)
from smiles_editflow.edit_distance import align_with_epsilon, levenshtein_script
from smiles_editflow.tokenizer import add_special, build_vocab, BOS, EOS, PAD, UNK, tokenize
from smiles_editflow.chemistry import canonicalize_smiles, randomized_smiles
from smiles_editflow.train_step import prepare_training_batch, kappa, sample_t_open_interval


def test_strip_epsilon():
    z = ["C", EPS, "O", EPS, "N"]
    assert strip_epsilon(z) == ["C", "O", "N"]


def test_alignment_strip_roundtrip():
    x0 = ["C", "C"]
    x1 = ["C", "O", "C"]
    a0, a1 = align_with_epsilon(x0, x1)
    assert strip_epsilon(a0) == x0
    assert strip_epsilon(a1) == x1


def test_target_count_matches_edit_distance():
    x0 = ["C", "C", "N"]
    x1 = ["C", "O", "O", "N"]
    a0, a1 = align_with_epsilon(x0, x1)
    z0, z1 = make_alignment_fixed_N(a0, a1, len(a0))
    z_t = z0

    del_pos, sub_pairs, ins_pairs = extract_targets(z_t, z1)
    target_count = len(del_pos) + len(sub_pairs) + len(ins_pairs)

    x0_full = add_special(x0)
    x1_full = add_special(x1)
    dist = len(levenshtein_script(x0_full, x1_full))

    assert target_count == dist


def test_sample_z_t_endpoints():
    x0 = ["C", "N"]
    x1 = ["O"]
    a0, a1 = align_with_epsilon(x0, x1)
    z0, z1 = make_alignment_fixed_N(a0, a1, len(a0))

    kappa = lambda t: t ** 3
    z_t0 = sample_z_t(z0, z1, 0.0, kappa)
    z_t1 = sample_z_t(z0, z1, 1.0, kappa)

    assert z_t0 == z0
    assert z_t1 == z1


def test_extract_targets_mapping():
    z_t = ["C", EPS, "O", "N", EPS, "S"]
    z1 = ["C", "F", EPS, "N", "P", "S"]

    del_pos, sub_pairs, ins_pairs = extract_targets(z_t, z1)

    assert del_pos == [1]
    assert sub_pairs == []
    assert ins_pairs == [(1, "F"), (3, "P")]


def test_prepare_training_batch_bos_shift_inserts():
    smiles_list = ["CC"]
    token_to_id, id_to_token = build_vocab(smiles_list)
    vocab_set = set(token_to_id.keys())

    seed = 123
    aligned_length = 8
    t_min = 0.5
    t_max = 0.5

    batch = prepare_training_batch(
        smiles_list,
        token_to_id,
        id_to_token,
        vocab_set,
        t_min=t_min,
        t_max=t_max,
        seed=seed,
        aligned_length=aligned_length,
        x0_mode="empty",
        x0_max_len=3,
        kappa_power=3,
    )

    rng = random.Random(seed)
    canonical = canonicalize_smiles(smiles_list[0])
    randomized = randomized_smiles(canonical) or canonical
    tokens = tokenize(randomized)
    t = sample_t_open_interval(rng, t_min, t_max)

    a0, a1 = align_with_epsilon([], tokens)
    z0, z1 = make_alignment_fixed_N(a0, a1, aligned_length, rng)
    z_t = sample_z_t(z0, z1, t, lambda u: kappa(u, 3), rng)

    del_pos, sub_pairs, ins_pairs = extract_targets(z_t, z1)
    expected_del = [i + 1 for i in del_pos]
    expected_sub = [(i + 1, token_to_id.get(tok, token_to_id.get(UNK))) for i, tok in sub_pairs]
    expected_ins = [(g + 1, token_to_id.get(tok, token_to_id.get(UNK))) for g, tok in ins_pairs]

    assert batch["del_targets"][0] == expected_del
    assert batch["sub_targets"][0] == expected_sub
    assert batch["ins_targets"][0] == expected_ins


def test_prepare_training_batch_bos_shift_deletes():
    smiles_list = ["C"]
    token_to_id, id_to_token = build_vocab(smiles_list)
    vocab_set = set(token_to_id.keys())

    seed = 0
    aligned_length = 8
    t_min = 0.5
    t_max = 0.5

    batch = prepare_training_batch(
        smiles_list,
        token_to_id,
        id_to_token,
        vocab_set,
        t_min=t_min,
        t_max=t_max,
        seed=seed,
        aligned_length=aligned_length,
        x0_mode="uniform",
        x0_max_len=3,
        kappa_power=3,
    )

    rng = random.Random(seed)
    canonical = canonicalize_smiles(smiles_list[0])
    randomized = randomized_smiles(canonical) or canonical
    tokens = tokenize(randomized)
    t = sample_t_open_interval(rng, t_min, t_max)

    vocab_tokens = [
        tok for tok in vocab_set
        if tok not in {BOS, EOS, PAD, UNK}
    ]
    L0 = rng.randint(0, 3)
    if vocab_tokens:
        x0_tokens = [rng.choice(vocab_tokens) for _ in range(L0)]
    else:
        x0_tokens = []

    a0, a1 = align_with_epsilon(x0_tokens, tokens)
    z0, z1 = make_alignment_fixed_N(a0, a1, aligned_length, rng)
    z_t = sample_z_t(z0, z1, t, lambda u: kappa(u, 3), rng)

    del_pos, sub_pairs, ins_pairs = extract_targets(z_t, z1)
    expected_del = [i + 1 for i in del_pos]
    expected_sub = [(i + 1, token_to_id.get(tok, token_to_id.get(UNK))) for i, tok in sub_pairs]
    expected_ins = [(g + 1, token_to_id.get(tok, token_to_id.get(UNK))) for g, tok in ins_pairs]

    assert batch["del_targets"][0] == expected_del
    assert batch["sub_targets"][0] == expected_sub
    assert batch["ins_targets"][0] == expected_ins


def test_uniform_halfhalf_alignment_counts_even():
    x0 = [f"X{i}" for i in range(10)]
    x1 = [f"T{i}" for i in range(13)]

    a0, a1 = build_uniform_halfhalf_alignment(x0, x1)

    delete_count = sum(1 for u, v in zip(a0, a1) if u != EPS and v == EPS)
    sub_count = sum(1 for u, v in zip(a0, a1) if u != EPS and v != EPS)
    ins_count = sum(1 for u, v in zip(a0, a1) if u == EPS and v != EPS)

    assert len(a0) == len(a1)
    assert delete_count == 5
    assert sub_count == 5
    assert ins_count == len(x1) - 5


def test_uniform_halfhalf_alignment_counts_odd():
    x0 = [f"X{i}" for i in range(9)]
    x1 = [f"T{i}" for i in range(12)]

    a0, a1 = build_uniform_halfhalf_alignment(x0, x1)

    delete_count = sum(1 for u, v in zip(a0, a1) if u != EPS and v == EPS)
    sub_count = sum(1 for u, v in zip(a0, a1) if u != EPS and v != EPS)
    ins_count = sum(1 for u, v in zip(a0, a1) if u == EPS and v != EPS)

    assert len(a0) == len(a1)
    assert delete_count == 4
    assert sub_count == 5
    assert ins_count == len(x1) - 5


def test_uniform_halfhalf_strip_roundtrip():
    x0 = ["N", "N", "N", "N", "N"]
    x1 = ["C", "C", "O", "C"]

    a0, a1 = build_uniform_halfhalf_alignment(x0, x1)

    assert strip_epsilon(a0) == x0
    assert strip_epsilon(a1) == x1


def test_prepare_batch_uniform_halfhalf_uses_mode():
    smiles_list = ["CCO"]
    token_to_id, id_to_token = build_vocab(smiles_list)
    vocab_set = set(token_to_id.keys())

    batch = prepare_training_batch(
        smiles_list,
        token_to_id,
        id_to_token,
        vocab_set,
        t_min=0.0,
        t_max=0.0,
        seed=0,
        aligned_length=16,
        x0_mode="uniform_halfhalf",
        x0_max_len=4,
        kappa_power=3,
        emp_tokens=["N"],
        emp_weights=[1.0],
    )

    assert batch is not None
    assert len(batch["del_targets"][0]) > 0
    assert len(batch["sub_targets"][0]) > 0


def test_prepare_batch_uniform_halfhalf_length_clamp():
    smiles_list = ["C"]
    token_to_id, id_to_token = build_vocab(smiles_list)
    vocab_set = set(token_to_id.keys())

    batch = prepare_training_batch(
        smiles_list,
        token_to_id,
        id_to_token,
        vocab_set,
        t_min=0.0,
        t_max=0.0,
        seed=0,
        aligned_length=16,
        x0_mode="uniform_halfhalf",
        x0_max_len=10,
        kappa_power=3,
        emp_tokens=["N"],
        emp_weights=[1.0],
    )

    assert batch is not None
    # x1 has length 1 => clamp enforces L0 <= 2, so at t=0 we should have exactly 2 non-insert edits.
    assert len(batch["del_targets"][0]) + len(batch["sub_targets"][0]) == 2
    assert len(batch["ins_targets"][0]) == 0
