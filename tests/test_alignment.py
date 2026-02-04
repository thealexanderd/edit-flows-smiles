"""Tests for epsilon-alignment utilities."""

from smiles_editflow.alignment import EPS, strip_epsilon, make_alignment_fixed_N, sample_z_t, extract_targets
from smiles_editflow.edit_distance import align_with_epsilon, levenshtein_script
from smiles_editflow.tokenizer import add_special


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
