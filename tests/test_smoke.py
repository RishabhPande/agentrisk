"""Smoke tests for AgentRisk-Sim: the generator runs, its schema and labels
are stable, it is deterministic per seed, and a small model trained on era-1
fraud separates held-out era-1 fraud better than chance.

These are fast sanity checks (small synthetic batches, a 30-tree model), not
the benchmark. The full experiment protocol lives in run.py.
"""
import numpy as np
import pandas as pd
import pytest

import sim

SMALL = {c: 40 for c in sim.ALL}


def test_generate_schema_and_labels():
    df = sim.generate(SMALL, seed=0)
    expected_cols = set(sim.B_FEATS + sim.D_FEATS + sim.T_FEATS + sim.A_FEATS + ["cls", "y"])
    missing = expected_cols - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    assert len(df) == sum(SMALL.values())
    for cls, sub in df.groupby("cls"):
        want = int(cls not in sim.LEGIT)
        assert (sub["y"] == want).all(), f"wrong labels for class {cls}"
    assert set(df["cls"].unique()) == set(sim.ALL)


def test_determinism_per_seed():
    a = sim.generate(SMALL, seed=7)
    b = sim.generate(SMALL, seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = sim.generate(SMALL, seed=8)
    assert not a.equals(c)


def test_nans_only_where_the_generator_intends():
    # Autofill humans can lack keystroke timing, mobile sessions have no mouse
    # path, and some agent telemetry modes report no timing: iki_mean_ms,
    # iki_cv and mouse_straightness legitimately contain NaNs (LightGBM handles
    # them natively). Every other model feature must stay NaN-free.
    df = sim.generate(SMALL, seed=1)
    feats = sim.B_FEATS + sim.D_FEATS + sim.T_FEATS + sim.A_FEATS
    nan_cols = {c for c in feats if df[c].isna().any()}
    assert nan_cols <= {"iki_mean_ms", "iki_cv", "mouse_straightness"}, nan_cols
    # and the intentional ones do occur on this batch, so the guard stays honest
    assert df["mouse_straightness"].isna().any()


def test_era1_separability_sanity():
    lgb = pytest.importorskip("lightgbm")
    from sklearn.metrics import roc_auc_score

    train = sim.generate({"H": 400, "FH": 150, "FB": 150}, seed=2)
    test = sim.generate({"H": 400, "FH": 150, "FB": 150}, seed=99)
    feats = sim.B_FEATS + sim.D_FEATS + sim.T_FEATS
    model = lgb.LGBMClassifier(n_estimators=30, learning_rate=0.1, num_leaves=15,
                               min_child_samples=10, verbose=-1, random_state=0)
    model.fit(train[feats], train["y"])
    auc = roc_auc_score(test["y"], model.predict_proba(test[feats])[:, 1])
    assert auc > 0.7, f"era-1 sanity AUC unexpectedly low: {auc:.3f}"
