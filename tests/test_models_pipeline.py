import json
import time

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate as ev
from src.models.calibrate import calibrate_run, crossfit_calibrate
from src.models.features import load_yaml
from src.models.splits import build_splits, train_categories
from src.models.train import (
    assert_no_weights,
    build_lr,
    lr_frame,
    resolve_features,
    train_model,
)

DATA = load_yaml("configs/data.yaml")
FEAT = load_yaml("configs/features.yaml")
TUNE = load_yaml("configs/models/tuning.yaml")


def model_cfg(name):
    return load_yaml(f"configs/models/{name}.yaml")


def test_tuning_budget_and_no_weight_keys_in_configs():
    assert TUNE["n_trials"] == 20 and TUNE["seed"] == 42
    for name in ("simple", "b0", "lr", "g0"):
        assert_no_weights(model_cfg(name))
    assert_no_weights(TUNE)


@pytest.mark.parametrize("key", ["scale_pos_weight", "is_unbalance", "class_weight"])
def test_weight_guard(key):
    with pytest.raises(ValueError):
        assert_no_weights({"lgb_fixed": {key: 3}})


def test_resolve_features_dist_modes():
    plus = resolve_features(model_cfg("g0"), FEAT, "plus")
    minus = resolve_features(model_cfg("g0"), FEAT, "minus")
    assert "dist_km" in plus["numeric"] and "dist_km" not in minus["numeric"]
    lr = resolve_features(model_cfg("lr"), FEAT, "plus")
    assert {"hour", "dow", "month"} <= set(lr["categorical"])


def test_freeze_guard_blocks_final_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "freeze_tag_exists", lambda tag: False)
    with pytest.raises(RuntimeError):
        ev.evaluate_run(tmp_path, DATA, TUNE, final_eval=True)


def test_crossfit_uses_other_fold_only():
    n = 4000
    dates = pd.Series(pd.date_range("2024-01-01", periods=n // 10, freq="D").repeat(10))
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < p).astype(int)
    out = crossfit_calibrate(p, y, dates, 7, 2)
    assert out.shape == p.shape and np.isfinite(out).all()
    assert abs(out.mean() - y.mean()) < 0.05


def test_smoke_all_models_end_to_end(synth_base, tmp_path):
    t0 = time.time()
    tune = {**TUNE, "n_trials": 2}
    parts = build_splits(synth_base, DATA["split_years"], str(tmp_path / "splits"))
    data_cfg = {**DATA, "splits_dir": str(tmp_path / "splits")}
    models_dir = str(tmp_path / "models")
    runs = []
    for name in ("simple", "b0", "lr", "g0"):
        cfg = model_cfg(name)
        for dist in cfg["dist_modes"]:
            run = train_model(
                cfg, tune, FEAT, dist, parts["train"], parts["valid"], models_dir, "r1"
            )
            runs.append(run)
            calibrate_run(run, data_cfg, tune)
            metrics, _ = ev.evaluate_run(run, data_cfg, tune)
            assert set(metrics["stage"]) == {"raw", "crossfit"}
            assert metrics["brier"].between(0, 1).all()
            allrows = metrics[metrics["group"] == "all"]
            assert allrows["citl"].notna().all()
            assert allrows[allrows["split"] == "valid"]["selection_set"].all()
            assert not allrows[allrows["split"] == "calib"]["selection_set"].any()
            assert (metrics["cohort"] == "center").all()
            assert (metrics["train_years"] == "2019-2022").all()
            expect = "D무관" if name in ("simple", "b0") else f"d{dist}"
            assert (metrics["dist_mode"] == expect).all()
            alert = json.loads((run / "threshold.json").read_text("utf-8"))["actual_alert_frac"]
            assert 0 < alert <= 1
            assert (run / "threshold.json").exists()
    assert len(runs) == 6
    g0 = next(r for r in runs if r.parent.parent.name == "g0" and r.parent.name == "dplus")
    params = json.loads((g0 / "params.json").read_text("utf-8"))
    assert len(params["trials"]) == 2 and params["seed"] == 42
    assert (g0 / "model.txt").exists()
    table = pd.concat([ev.evaluate_run(r, data_cfg, tune)[0] for r in runs])
    out = tmp_path / "m.csv"
    ev.append_results(table, out)
    ev.append_results(table, out)
    assert len(pd.read_csv(out)) == len(table)
    assert time.time() - t0 < 180


def test_lr_unseen_category_is_not_nan_category():
    train = pd.DataFrame(
        {
            "center_id": ["A"] * 6 + ["B"] * 6 + [None] * 2,
            "out_of_jur": [0.0] * 13 + [None],
            "y_late": [0, 1] * 7,
        }
    )
    spec = {"numeric": ["out_of_jur"], "categorical": ["center_id"]}
    cats = train_categories(train, ["center_id"])
    frame = lr_frame(train, spec, cats)
    assert (frame["center_id"] == "__nan__").sum() == 2
    assert frame["out_of_jur_missing"].sum() == 1
    new = pd.DataFrame({"center_id": ["Z", None, "A"], "out_of_jur": [0.0, 0.0, 0.0]})
    new_frame = lr_frame(new, spec, cats)
    assert new_frame["center_id"].tolist() == ["__unseen__", "__nan__", "A"]
    model = build_lr(spec, 1.0, 42).fit(frame, train["y_late"])
    onehot = model.named_steps["prep"].transform(new_frame)
    cat_block = onehot[:, 2:].toarray() if hasattr(onehot, "toarray") else onehot[:, 2:]
    assert cat_block[0].sum() == 0
    assert cat_block[1].sum() == 1


def test_dminus_rejected_for_distance_only_models(synth_base, tmp_path):
    parts = build_splits(synth_base, DATA["split_years"], str(tmp_path / "s"))
    with pytest.raises(ValueError):
        train_model(
            model_cfg("b0"), TUNE, FEAT, "minus", parts["train"], parts["valid"], str(tmp_path)
        )
