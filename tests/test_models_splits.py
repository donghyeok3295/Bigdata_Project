import pandas as pd
import pytest

from src.models.features import load_yaml
from src.models.splits import (
    build_splits,
    fix_categories,
    load_split,
    read_variant,
    train_categories,
    variant_name,
)

YEARS = load_yaml("configs/data.yaml")["split_years"]


def test_splits_do_not_overlap_and_cover(synth_base, tmp_path):
    parts = build_splits(synth_base, YEARS, str(tmp_path))
    ids = [set(p["call_id"]) for p in parts.values()]
    assert sum(len(i) for i in ids) == len(synth_base)
    assert len(set.union(*ids)) == len(synth_base)
    assert parts["train"]["year"].max() <= 2022
    assert parts["valid"]["year"].unique().tolist() == [2023]
    assert parts["calib"]["year"].unique().tolist() == [2024]


def test_build_splits_rejects_test_year(synth_base, tmp_path):
    bad = synth_base.copy()
    bad.loc[0, "year"] = 2025
    with pytest.raises(ValueError):
        build_splits(bad, YEARS, str(tmp_path))


def test_load_split_test_guard(synth_base, tmp_path):
    build_splits(synth_base, YEARS, str(tmp_path))
    with pytest.raises(ValueError):
        load_split("test", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_split("test", str(tmp_path), final_eval=True)
    assert len(load_split("valid", str(tmp_path))) > 0


def test_unseen_category_becomes_nan():
    train = pd.DataFrame({"c": ["a", "b", "a"]})
    valid = pd.DataFrame({"c": ["a", "z"]})
    cats = train_categories(train, ["c"])
    out = fix_categories(valid, cats)
    assert cats == {"c": ["a", "b"]}
    assert out["c"].isna().tolist() == [False, True]


def test_calib_months_variant_and_variant_file(synth_base, tmp_path):
    parts = build_splits(synth_base, YEARS, str(tmp_path), calib_months=(2, 12))
    assert parts["calib"]["month"].min() == 2
    info = read_variant(str(tmp_path))
    assert info["calib_months"] == [2, 12] and info["cohort"] == "center"
    assert variant_name(YEARS, (2, 12)) == "train2019-2022_calib2-12"
    assert read_variant(str(tmp_path / "none"))["name"] == "unknown"
