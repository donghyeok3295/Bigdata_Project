from pathlib import Path

import numpy as np
import pandas as pd

from src.models.features import (
    add_out_of_jur,
    add_time_features,
    build_base_table,
    clean_dist,
    filter_safety_center,
    load_holidays,
    load_yaml,
    make_label,
    unit_type_of,
)

KEYWORDS = load_yaml("configs/features.yaml")["unit_type_keywords"]


def test_label_boundary_matches_gt_not_ge():
    resp = pd.Series([7.99, 8.0, 8.01])
    assert make_label(resp, 8).tolist() == [0, 0, 1]
    server_ge = (resp >= 8).astype(int)
    assert server_ge.tolist() != make_label(resp, 8).tolist()


def test_clean_dist_nan_for_errors_keeps_out_of_range():
    out = clean_dist(pd.Series([-1, 0, 6.5, 100, 101, 2024, None]), 100)
    assert out.isna().tolist() == [True, False, False, False, True, True, True]
    assert out.iloc[2] == 6.5


def test_unit_type():
    names = pd.Series(["개포119안전센터", "서울현장대응단", "중부119구조대", "항공대", None])
    assert unit_type_of(names, KEYWORDS).tolist() == [
        "center",
        "field_unit",
        "rescue",
        "other",
        "other",
    ]


def test_out_of_jur_only_for_center_and_mapped():
    ref = pd.DataFrame({"center_id": ["A 안전센터"], "center_sgg": ["강남구"]})
    df = pd.DataFrame(
        {
            "center_id": ["A 안전센터", "A 안전센터", "B 안전센터", "A 안전센터"],
            "sgg": ["강남구", "서초구", "강남구", "서초구"],
            "unit_type": ["center", "center", "center", "other"],
        }
    )
    out = add_out_of_jur(df, ref)["out_of_jur"]
    assert out.iloc[:2].tolist() == [0.0, 1.0]
    assert out.iloc[2:].isna().all()


def test_time_features_dow_monday_zero_and_midnight():
    df = pd.DataFrame({"t": ["2024-01-01 00:00:00", "2024-12-31 23:59:59", "2024-10-01 12:00:00"]})
    holidays = {pd.Timestamp("2024-10-01").date()}
    out = add_time_features(df, "t", holidays)
    assert out["dow"].tolist() == [0, 1, 1]
    assert out["hour"].tolist() == [0, 23, 12]
    assert out["month"].tolist() == [1, 12, 10]
    assert out["is_holiday"].tolist() == [0, 0, 1]


def test_holidays_file_is_draft_and_covers_years():
    text = Path("configs/holidays_kr.yaml").read_text(encoding="utf-8")
    assert "팀 대조 전" in text.splitlines()[0]
    years = {d.year for d in load_holidays("configs/holidays_kr.yaml")}
    assert years == set(range(2019, 2026))


def test_build_base_table(tmp_path):
    ref = tmp_path / "ref.parquet"
    pd.DataFrame({"center_id": ["A 119안전센터"], "center_sgg": ["강남구"]}).to_parquet(ref)
    hol = tmp_path / "h.yaml"
    hol.write_text('holidays:\n  2024:\n    - "2024-10-01"\n', encoding="utf-8")
    data_cfg = {
        **load_yaml("configs/data.yaml"),
        "holidays": str(hol),
        "ref_center_sgg": str(ref),
    }
    raw = pd.DataFrame(
        {
            "call_id": ["1", "2", "3"],
            "dclr_str": ["2024-10-01 08:00:00", "2024-10-02 09:00:00", "2024-10-02 10:00:00"],
            "center_id": ["A 119안전센터", "A 119안전센터", "X 현장대응단"],
            "cntr_nm": ["119안전센터", "119안전센터", "현장대응단"],
            "sgg": ["서초구", "강남구", "강남구"],
            "dist_raw": [7.0, -1.0, 2.0],
            "resp_min": [8.0, 9.0, 4.0],
        }
    )
    out = build_base_table(raw, data_cfg, load_yaml("configs/features.yaml"))
    assert len(out) == 3
    assert out["y_late"].tolist() == [0, 1, 0]
    assert out["out_of_range"].tolist() == [1, 0, 0]
    assert np.isnan(out["dist_km"].iloc[1])
    assert out["out_of_jur"].iloc[:2].tolist() == [1.0, 0.0]
    assert out["out_of_jur"].isna().tolist() == [False, False, True]
    assert out["is_holiday"].tolist() == [1, 0, 0]
    assert out["year"].tolist() == [2024, 2024, 2024]


def test_filter_safety_center_counts_by_year():
    df = pd.DataFrame(
        {
            "year": [2023, 2023, 2023, 2024, 2024],
            "unit_type": ["center", "field_unit", "other", "center", "rescue"],
        }
    )
    kept, excl = filter_safety_center(df)
    assert len(kept) == 2 and (kept["unit_type"] == "center").all()
    row23 = excl[excl["year"] == 2023].iloc[0]
    assert (row23["n_total"], row23["n_center"], row23["n_excluded"]) == (3, 1, 2)
    assert row23["n_field_unit"] == 1 and row23["n_other"] == 1
    assert excl[excl["year"] == 2024].iloc[0]["n_rescue"] == 1
