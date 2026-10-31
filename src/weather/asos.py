import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

COLUMNS = ["stn", "name", "ts", "ta", "ta_qc", "rn", "rn_qc", "snow", "snow3"]
HEADER_PREFIX = [
    "지점",
    "지점명",
    "일시",
    "기온",
    "기온 QC",
    "강수량",
    "강수량 QC",
    "적설",
    "3시간신적설",
]


def load_config(path: str = "configs/data.yaml") -> dict:
    """data.yaml의 weather 설정을 읽는다."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["weather"]


def read_asos(raw_dir: str) -> pd.DataFrame:
    """ASOS 시간자료 CSV(cp949)를 모두 읽어 컬럼명을 통일해 합친다."""
    frames = []
    for path in sorted(Path(raw_dir).glob("OBS_ASOS_TIM_20*.csv")):
        df = pd.read_csv(path, encoding="cp949")
        for name, prefix in zip(df.columns, HEADER_PREFIX, strict=True):
            assert name.startswith(prefix), f"{path.name}: 예상 컬럼 {prefix}, 실제 {name}"
        df.columns = COLUMNS
        log.info("read %s rows=%d", path.name, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_hourly(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """정시 격자로 맞추고 rain_mm, rain_3h_mm과 결측 표시, snow_cm을 만든다.

    빈칸과 QC 9는 무강수다(일강수량 대조로 확인). 결측은 행이 없거나 QC 1인 시간뿐이다.
    rain_3h_mm은 3시간 정시마다 직전 3시간 누적이다. 겨울 정시 값은 이미 3시간 누적이다.
    """
    df = raw[raw["stn"] == cfg["station"]].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.drop_duplicates("ts").set_index("ts")

    grid = pd.date_range(
        f"{df.index.min().year}-01-01", f"{df.index.max().year}-12-31 23:00", freq="h"
    )
    df = df.reindex(grid)
    for col in ("rn", "rn_qc", "snow"):
        df[col] = df[col].astype(float)
    block = cfg["winter_interval_h"]

    missing = pd.Series(df["stn"].isna() | (df["rn_qc"] == 1), index=grid)
    rain_mm = df["rn"].where(~missing).fillna(0.0)
    # 기상청 하루는 01~24시라 계절은 00시를 전날로 보고 정한다
    day_month = (grid - pd.Timedelta(minutes=1)).month
    winter = pd.Series(day_month.isin(cfg["winter_months"]), index=grid)
    block_end = pd.Series(grid.hour % block == 0, index=grid)

    summer_sum = rain_mm.rolling(block, min_periods=1).sum()
    summer_bad = missing.astype(int).rolling(block, min_periods=1).sum() > 0
    block_missing = pd.Series(np.where(winter, missing, summer_bad), index=grid)
    block_mm = pd.Series(np.where(winter, rain_mm, summer_sum), index=grid)

    out = pd.DataFrame(
        {
            "ts": grid,
            "rain_mm": rain_mm.to_numpy(),
            "rain_missing": missing.astype("int8").to_numpy(),
            "rain_3h_mm": block_mm.where(block_end & ~block_missing, np.nan).to_numpy(),
            "rain_3h_missing": block_missing.astype("Int8").where(block_end).reset_index(drop=True),
            "snow_cm": df["snow"].fillna(0.0).to_numpy(),
        }
    )
    return out


def latest_block_end(t: pd.Timestamp | pd.Series, cfg: dict) -> pd.Timestamp | pd.Series:
    """신고 시각 t보다 공개 지연 이상 앞서 끝난 가장 최근 3시간 블록의 종료 시각."""
    shifted = t - pd.Timedelta(minutes=cfg["publish_delay_min"])
    freq = f"{cfg['winter_interval_h']}h"
    return shifted.dt.floor(freq) if isinstance(t, pd.Series) else shifted.floor(freq)


def attach_rain(calls: pd.DataFrame, weather: pd.DataFrame, ts_col: str, cfg: dict) -> pd.DataFrame:
    """신고 시각 기준으로 직전에 끝난 3시간 블록의 rain_3h_mm, rain_3h_missing을 붙인다."""
    out = calls.copy()
    out["rain_block_end"] = latest_block_end(out[ts_col], cfg)
    blocks = weather.loc[
        weather["rain_3h_missing"].notna(), ["ts", "rain_3h_mm", "rain_3h_missing"]
    ]
    blocks = blocks.rename(columns={"ts": "rain_block_end"})
    return out.merge(blocks, on="rain_block_end", how="left", validate="many_to_one")


def build(config_path: str = "configs/data.yaml") -> pd.DataFrame:
    """원본 CSV를 정제해 work/weather 아래 parquet로 저장한다."""
    cfg = load_config(config_path)
    raw = read_asos(cfg["raw_dir"])
    out = clean_hourly(raw, cfg)
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg["out"], index=False)
    log.info(
        "weather rows in=%d out=%d rain_missing=%d rain_3h_missing=%d snow_hours=%d",
        len(raw),
        len(out),
        out["rain_missing"].sum(),
        out["rain_3h_missing"].sum(),
        (out["snow_cm"] > 0).sum(),
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
