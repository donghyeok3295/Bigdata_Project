import pandas as pd
import pytest

from src.weather.asos import COLUMNS, attach_rain, clean_hourly, latest_block_end, read_asos

CFG = {
    "station": 108,
    "winter_months": [11, 12, 1, 2, 3],
    "winter_interval_h": 3,
    "publish_delay_min": 60,
}


def make(rows):
    return pd.DataFrame(
        [[108, "서울", ts, 10.0, None, rn, qc, snow, None] for ts, rn, qc, snow in rows],
        columns=COLUMNS,
    )


def full_year(year, rows):
    """연도 전체 시간 격자를 채우되 rows에 준 시각만 값을 덮어쓴다."""
    grid = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    given = {pd.Timestamp(ts): (rn, qc, snow) for ts, rn, qc, snow in rows}
    return make([(t, *given.get(t, (None, None, None))) for t in grid])


def at(out, ts):
    return out.set_index("ts").loc[pd.Timestamp(ts)]


def test_blank_and_qc9_are_dry_not_missing():
    raw = make(
        [
            ("2024-07-01 00:00", None, None, None),
            ("2024-07-01 01:00", None, 9, None),
            ("2024-07-01 02:00", 3.5, None, None),
            ("2024-07-01 03:00", 0.0, None, None),
        ]
    )
    out = clean_hourly(raw, CFG)
    for ts, mm in [
        ("2024-07-01 00:00", 0.0),
        ("2024-07-01 01:00", 0.0),
        ("2024-07-01 02:00", 3.5),
        ("2024-07-01 03:00", 0.0),
    ]:
        assert at(out, ts).rain_mm == mm
        assert at(out, ts).rain_missing == 0


def test_qc1_and_absent_hours_are_missing():
    raw = make([("2024-07-01 00:00", 1.0, 1, None), ("2024-12-31 00:00", None, None, None)])
    out = clean_hourly(raw, CFG)
    assert at(out, "2024-07-01 00:00").rain_missing == 1
    assert at(out, "2024-07-01 00:00").rain_mm == 0.0
    assert len(out) == 366 * 24
    assert at(out, "2024-12-31 05:00").rain_missing == 1


def test_summer_3h_block_is_sum_of_three_hours():
    out = clean_hourly(
        full_year(
            2024,
            [
                ("2024-07-01 04:00", 1.0, None, None),
                ("2024-07-01 05:00", 2.0, None, None),
                ("2024-07-01 06:00", 0.5, None, None),
                ("2024-07-01 07:00", 9.0, None, None),
            ],
        ),
        CFG,
    )
    row = at(out, "2024-07-01 06:00")
    assert row.rain_3h_mm == 3.5
    assert row.rain_3h_missing == 0
    assert pd.isna(at(out, "2024-07-01 07:00").rain_3h_mm)
    assert pd.isna(at(out, "2024-07-01 07:00").rain_3h_missing)
    assert at(out, "2024-07-01 09:00").rain_3h_mm == 9.0


def test_winter_3h_block_uses_recorded_value():
    out = clean_hourly(
        full_year(
            2024,
            [
                ("2024-01-15 12:00", 4.0, None, None),
                ("2024-01-15 13:00", None, 9, None),
                ("2024-01-15 14:00", None, 9, None),
                ("2024-01-15 15:00", None, None, None),
            ],
        ),
        CFG,
    )
    assert at(out, "2024-01-15 12:00").rain_3h_mm == 4.0
    assert at(out, "2024-01-15 15:00").rain_3h_mm == 0.0
    assert at(out, "2024-01-15 15:00").rain_3h_missing == 0
    assert at(out, "2024-01-15 13:00").rain_missing == 0


def test_block_missing_when_qc1_inside_block():
    out = clean_hourly(
        full_year(
            2024, [("2024-07-01 05:00", 2.0, 1, None), ("2024-07-01 06:00", 1.0, None, None)]
        ),
        CFG,
    )
    assert at(out, "2024-07-01 06:00").rain_3h_missing == 1
    assert pd.isna(at(out, "2024-07-01 06:00").rain_3h_mm)


def test_daily_total_matches_hourly_and_block_sums():
    rows = [
        ("2024-07-02 01:00", 1.0, None, None),
        ("2024-07-02 05:00", 2.5, None, None),
        ("2024-07-02 13:00", 0.5, None, None),
        ("2024-07-02 15:00", 4.0, None, None),
        ("2024-07-02 23:00", 1.5, None, None),
        ("2024-07-03 00:00", 0.5, None, None),
    ]
    out = clean_hourly(full_year(2024, rows), CFG).set_index("ts")
    day = out.loc["2024-07-02 01:00":"2024-07-03 00:00"]
    assert day.rain_mm.sum() == 10.0
    assert day.rain_3h_mm.sum() == 10.0


def test_snow_blank_is_zero_and_other_stations_dropped():
    raw = make([("2024-02-01 00:00", None, None, 2.4), ("2024-02-01 03:00", None, None, None)])
    other = raw.iloc[[0]].copy()
    other["stn"] = 999
    out = clean_hourly(pd.concat([raw, other]), CFG)
    assert at(out, "2024-02-01 00:00").snow_cm == 2.4
    assert at(out, "2024-02-01 03:00").snow_cm == 0.0


def test_season_boundary_follows_observation_day():
    out = clean_hourly(
        full_year(
            2024,
            [
                ("2024-04-01 00:00", 1.5, None, None),
                ("2024-10-31 22:00", 1.0, None, None),
                ("2024-10-31 23:00", 2.0, None, None),
                ("2024-11-01 00:00", 0.5, None, None),
            ],
        ),
        CFG,
    )
    assert at(out, "2024-04-01 00:00").rain_3h_mm == 1.5
    assert at(out, "2024-04-01 00:00").rain_3h_missing == 0
    assert at(out, "2024-11-01 00:00").rain_3h_mm == 3.5
    assert out.rain_3h_missing.sum() == 0


def test_zero_value_and_blank_are_both_dry():
    out = clean_hourly(
        full_year(2024, [("2024-07-01 03:00", 0.0, None, None)]),
        CFG,
    )
    assert at(out, "2024-07-01 03:00").rain_3h_mm == 0.0
    assert at(out, "2024-07-01 06:00").rain_3h_mm == 0.0


def test_absent_row_makes_block_missing():
    raw = full_year(2024, [])
    raw = raw[~raw["ts"].isin([pd.Timestamp("2024-01-15 12:00"), pd.Timestamp("2024-07-01 05:00")])]
    out = clean_hourly(raw, CFG)
    assert at(out, "2024-01-15 12:00").rain_3h_missing == 1
    assert at(out, "2024-07-01 06:00").rain_3h_missing == 1
    assert at(out, "2024-07-01 09:00").rain_3h_missing == 0


def test_year_boundary_and_leap_day():
    raw = pd.concat(
        [
            full_year(2023, [("2023-12-31 23:00", 2.0, None, None)]),
            full_year(2024, [("2024-01-01 00:00", 1.0, None, None)]),
        ]
    )
    out = clean_hourly(raw, CFG)
    assert len(out) == (365 + 366) * 24
    assert at(out, "2024-01-01 00:00").rain_3h_mm == 1.0
    assert at(out, "2024-02-29 03:00").rain_3h_missing == 0


def test_missing_flag_dtype_is_nullable_int():
    out = clean_hourly(full_year(2024, []), CFG)
    assert str(out.rain_3h_missing.dtype) == "Int8"
    assert out.rain_3h_missing.isna().sum() == (out.ts.dt.hour % 3 != 0).sum()


def test_latest_block_end_respects_publish_delay():
    assert latest_block_end(pd.Timestamp("2024-07-01 06:30"), CFG) == pd.Timestamp(
        "2024-07-01 03:00"
    )
    assert latest_block_end(pd.Timestamp("2024-07-01 07:00"), CFG) == pd.Timestamp(
        "2024-07-01 06:00"
    )
    assert latest_block_end(pd.Timestamp("2024-07-01 09:00"), CFG) == pd.Timestamp(
        "2024-07-01 06:00"
    )


def test_read_asos_rejects_unexpected_header(tmp_path):
    bad = pd.DataFrame([[108, "서울", "2024-01-01 00:00", 1, None, None, None, None, None]])
    bad.columns = [
        "지점",
        "지점명",
        "일시",
        "강수량(mm)",
        "기온 QC플래그",
        "기온(°C)",
        "강수량 QC플래그",
        "적설(cm)",
        "3시간신적설(cm)",
    ]
    bad.to_csv(tmp_path / "OBS_ASOS_TIM_2024.csv", index=False, encoding="cp949")
    with pytest.raises(AssertionError):
        read_asos(str(tmp_path))


def test_attach_rain_uses_only_blocks_ended_before_report_time():
    weather = clean_hourly(
        full_year(
            2024,
            [
                ("2024-07-01 04:00", 1.0, None, None),
                ("2024-07-01 05:00", 2.0, None, None),
                ("2024-07-01 06:00", 0.5, None, None),
                ("2024-07-01 07:00", 9.0, None, None),
            ],
        ),
        CFG,
    )
    calls = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "dclr": pd.to_datetime(["2024-07-01 06:30", "2024-07-01 07:00", "2024-07-01 08:59"]),
        }
    )
    out = attach_rain(calls, weather, "dclr", CFG)
    assert len(out) == 3
    assert out.rain_3h_mm.tolist() == [0.0, 3.5, 3.5]
    assert out.rain_3h_missing.tolist() == [0, 0, 0]
    assert (out["dclr"] - out["rain_block_end"] >= pd.Timedelta(minutes=60)).all()
