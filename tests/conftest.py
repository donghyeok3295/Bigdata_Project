import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_base() -> pd.DataFrame:
    """합성 기본 표 1만 행(2019~2024). 지연은 거리·시간에 약하게 의존한다."""
    rng = np.random.default_rng(0)
    n = 10_000
    days = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    date = pd.Series(days[rng.integers(0, len(days), n)]).dt.strftime("%Y-%m-%d")
    centers = [f"가{i}소방서 나{i}119안전센터" for i in range(10)]
    center = rng.choice(centers, n)
    sgg = rng.choice(["강남구", "서초구", "종로구", "중구", "마포구"], n)
    hour = rng.integers(0, 24, n)
    dist = rng.integers(0, 9, n).astype(float)
    dist[rng.random(n) < 0.05] = np.nan
    z = -1.5 + 0.25 * np.nan_to_num(dist, nan=3.0) + 0.3 * (hour >= 22)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype("int8")
    ts = pd.to_datetime(date)
    return pd.DataFrame(
        {
            "call_id": np.arange(n).astype(str),
            "date": date,
            "year": ts.dt.year.astype("int16"),
            "center_id": center,
            "sgg": sgg,
            "unit_type": "center",
            "out_of_jur": rng.integers(0, 2, n).astype(float),
            "hour": hour.astype("int8"),
            "dow": ts.dt.dayofweek.astype("int8"),
            "month": ts.dt.month.astype("int8"),
            "is_holiday": 0,
            "dist_km": dist,
            "out_of_range": 0,
            "resp_min": np.where(y == 1, 9.0, 5.0),
            "y_late": y,
        }
    )
