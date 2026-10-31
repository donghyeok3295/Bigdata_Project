import pandas as pd
import pytest

from src.models.baselines import SimpleRateModel


def test_fallback_order_cell_center_global():
    rows = (
        [("A", 8, 1)] * 60  # A 8시 셀 충분: 지연율 1.0
        + [("A", 9, 0)] * 10  # A 9시 셀 부족 -> A 센터 평균
        + [("B", 8, 0)] * 10  # B 센터 전체 부족 -> 전체 평균
    )
    train = pd.DataFrame(rows, columns=["center_id", "hour", "y_late"])
    model = SimpleRateModel(min_n=50).fit(train)
    center_a = 60 / 70
    test = pd.DataFrame({"center_id": ["A", "A", "B", "C"], "hour": [8, 9, 8, 8]})
    out = model.predict(test).tolist()
    assert out[0] == 1.0
    assert out[1] == pytest.approx(center_a)
    assert out[2] == pytest.approx(model.global_rate)
    assert out[3] == pytest.approx(model.global_rate)
    assert model.global_rate == pytest.approx(60 / 80)
