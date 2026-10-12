"""결측 처리 방법 10종을 적용하고 결과를 비교한다."""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent   # coursework/netflix
CSV = BASE / "data" / "netflix_titles.csv"
OUT = BASE / "output"
OUT.mkdir(parents=True, exist_ok=True)

MISS = ["director", "cast", "country", "date_added", "rating"]

df = pd.read_csv(CSV)
N = len(df)

# [정제] 문자열 앞뒤 공백 제거, 빈 문자열은 결측으로 통일 (교재 3.5.2)
_blank_before = int(df.select_dtypes(include="object").apply(
    lambda c: c.astype("string").str.strip().eq("").sum()).sum())
_ws_before = int(df.select_dtypes(include="object").apply(
    lambda c: (c.astype("string") != c.astype("string").str.strip()).sum()).sum())
for _c in df.select_dtypes(include="object").columns:
    df[_c] = df[_c].astype("string").str.strip().replace("", pd.NA)
print(f"[정제] 앞뒤 공백 포함 셀 {_ws_before}개 정리, 빈 문자열 {_blank_before}개를 결측 처리")
df["g1"] = df["listed_in"].str.split(",").str[0].str.strip()
df["is_doc"] = df["listed_in"].str.contains("Documentar", case=False, na=False)
df["lead"] = df["cast"].str.split(",").str[0].str.strip()

rows = []
def log(no, name, target, filled, note, adopt):
    rows.append({"No": no, "방법": name, "대상 컬럼": target,
                 "확보 건수": filled, "근거/위험": note, "채택": adopt})

base = int(df[MISS].notna().all(axis=1).sum())

# M0 행 제거
log("M0", "행 제거(listwise)", "전체", -(N - base),
    f"기준선: {base:,}행만 생존, {N - base:,}행 손실", "미채택")

# M1 상수 치환
log("M1", "Unknown 상수 치환", "전체", 0,
    "결측 표시만 바뀜, 정보량 증가 없음", "미채택")

# M2 전체 최빈값
rmode = df["rating"].mode().iloc[0]
log("M2", "전체 최빈값 대치", "rating", 7,
    f"최빈값 '{rmode}' 일괄 적용, 장르 특성 무시", "미채택")

# M3 그룹별 최빈값 (표본 5개 미만이면 type 최빈값으로 폴백)
g = df.dropna(subset=["rating"]).groupby(["type", "g1"])["rating"]
gmode = g.agg(n="count", mode=lambda s: s.mode().iloc[0])
tmode = df.dropna(subset=["rating"]).groupby("type")["rating"].agg(lambda s: s.mode().iloc[0])

def fill_rating(r):
    key = (r["type"], r["g1"])
    if key in gmode.index and gmode.loc[key, "n"] >= 5:
        return gmode.loc[key, "mode"]
    return tmode[r["type"]]

m3 = df[df["rating"].isna()].apply(fill_rating, axis=1)
log("M3", "그룹별 최빈값 대치", "rating", len(m3),
    "type+주장르 최빈값, 표본 5건 미만이면 type 최빈값", "채택")

# M4 country <- 같은 감독
dmap = df.dropna(subset=["director", "country"]).groupby("director")["country"].agg(lambda s: s.mode().iloc[0])
need = df["country"].isna() & df["director"].notna()
hit_dir = df.loc[need, "director"].isin(dmap.index)
idx_dir = df.loc[need].index[hit_dir]
log("M4", "같은 감독의 다른 작품에서 추론", "country", len(idx_dir),
    "2편 이상 감독 697명 중 34.0%가 2개국 이상 -> 오류 위험", "채택")

# M5 country <- 같은 주연배우 (M4 미해결분만)
cmap = df.dropna(subset=["lead", "country"]).groupby("lead")["country"].agg(lambda s: s.mode().iloc[0])
rest = df["country"].isna() & ~df.index.isin(idx_dir) & df["lead"].notna()
n_cast = int(df.loc[rest, "lead"].isin(cmap.index).sum())
log("M5", "같은 주연배우의 다른 작품에서 추론", "country", n_cast,
    "2편 이상 배우 1,016명 중 41.2%가 2개국 이상 -> 위험 더 큼", "미채택")

# M6 구조적 결측 식별 -> 범위 한정
mv_dir = int(df[df.type == "Movie"]["director"].isna().sum())
log("M6a", "구조적 결측 식별 후 범위 한정", "director", 2389 - mv_dir,
    f"TV 92.4% 구조적. 영화만 분석 시 결측 {mv_dir}건(3.0%)로 축소", "채택")
nd_cast = int(df[~df.is_doc]["cast"].isna().sum())
log("M6b", "구조적 결측 식별 후 범위 한정", "cast", 718 - nd_cast,
    f"다큐 48.3% vs 비다큐 4.8%. 비다큐만 시 결측 {nd_cast}건", "채택")

# M7 결측 플래그
log("M7", "결측 플래그 변수 생성", "director/cast", 0,
    "결측 여부를 변수화(TV 시리즈/다큐 여부의 대리지표)", "채택")

# M8 형식 변환
mv = df.type == "Movie"
n_min = int(df.loc[mv, "duration"].str.contains("min", na=False).sum())
n_sea = int(df.loc[~mv, "duration"].str.contains("Season", na=False).sum())
log("M8", "문자열을 숫자로 변환", "duration", n_min + n_sea,
    f"분(영화 {n_min:,}) / 시즌(TV {n_sea:,})으로 컬럼 분리", "채택")

# M9 근거 없는 추론
log("M9", "외부 근거 없는 추론", "date_added", 0,
    "넷플릭스 등록일은 데이터 내 단서 없음 -> 채우지 않음", "미채택")

cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(OUT / "method_comparison.csv", index=False, encoding="utf-8-sig")

# ── 최종 파이프라인 ────────────────────────────────────────
f = df.copy()
f.loc[f["rating"].isna(), "rating"] = m3
f["country_imputed"] = 0
f.loc[idx_dir, "country"] = df.loc[idx_dir, "director"].map(dmap)
f.loc[idx_dir, "country_imputed"] = 1
f["director_missing"] = df["director"].isna().astype(int)
f["cast_missing"] = df["cast"].isna().astype(int)
f["duration_min"] = f.loc[mv, "duration"].str.extract(r"(\d+)").astype(float)
f["duration_seasons"] = f.loc[~mv, "duration"].str.extract(r"(\d+)").astype(float)
f = f.drop(columns=["g1", "lead"])  # 중간 작업용 컬럼 제거
f.to_csv(OUT / "netflix_cleaned.csv", index=False, encoding="utf-8-sig")

after = int(f[MISS].notna().all(axis=1).sum())
mv_ok = int(f[f.type == "Movie"][MISS].notna().all(axis=1).sum())
core = ["country", "date_added", "rating"]
core_ok = int(f[core].notna().all(axis=1).sum())
nodoc_ok = int(f[(~f.is_doc)][["cast"] + core].notna().all(axis=1).sum())

rep = [cmp_df.to_string(index=False), "", "=" * 70,
       f"전체 {N:,}행",
       f"[기준선] 결측 행 전부 제거 시 생존       : {base:,}행 ({base / N * 100:.1f}%)",
       f"[처리 후] 5개 컬럼 모두 유효             : {after:,}행 ({after / N * 100:.1f}%)  (+{after - base:,}행)",
       f"[영화 한정] 5개 컬럼 모두 유효           : {mv_ok:,}행 / 영화 {int(mv.sum()):,}행 ({mv_ok / mv.sum() * 100:.1f}%)",
       f"[구조적 결측 제외] country/date/rating   : {core_ok:,}행 ({core_ok / N * 100:.1f}%)",
       f"[비다큐 + 위 3컬럼 + cast]               : {nodoc_ok:,}행",
       "",
       f"country 추론 보완: {len(idx_dir)}건 (country_imputed=1로 표시)",
       f"rating 대치: {len(m3)}건",
       ]
with open(OUT / "_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(rep))

# ── 차트 5. 처리 시나리오별 확보 행 수 ──────────────────────
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

labels = ["원본\n전체", "행 제거\n(기준선)", "5개 컬럼\n모두 보완",
          "영화 한정\n(감독 분석용)", "비다큐 한정\n(출연진 포함)", "구조적 결측\n컬럼 제외"]
vals = [N, base, after, mv_ok, nodoc_ok, core_ok]
colors = ["#8C8C8C", "#C44E52", "#C44E52", "#4C72B0", "#55A868", "#55A868"]

fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.bar(labels, vals, color=colors)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + N * 0.015,
            f"{v:,}행\n({v / N * 100:.1f}%)", ha="center", fontsize=9)
ax.axhline(base, color="#C44E52", ls="--", lw=1, alpha=0.7)
ax.text(5.45, base + N * 0.01, "기준선", color="#C44E52", fontsize=8, ha="right")
ax.set_xlabel("결측 처리 시나리오")
ax.set_ylabel("분석 가능한 행 수")
ax.set_title(f"그림 5. 결측 처리 시나리오별 분석 가능 데이터 (분모: 전체 {N:,}행)")
ax.set_ylim(0, N * 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "05_시나리오별_확보행수.png", dpi=150)
plt.close()
print("saved 05")

# ── 그림 5. 결측 처리 전후 비교 ──────────────────────────────
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

cols = ["director", "cast", "country", "date_added", "rating"]
bv = [int(df[c].isna().sum()) for c in cols]
av = [int(f[c].isna().sum()) for c in cols]

x = np.arange(len(cols)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x - w/2, bv, w, label="처리 전", color="#C44E52")
b2 = ax.bar(x + w/2, av, w, label="처리 후", color="#55A868")
for bars, vals in [(b1, bv), (b2, av)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + N*0.008,
                f"{v:,}건\n({v/N*100:.2f}%)", ha="center", fontsize=8)
ax.set_xticks(x, cols)
ax.set_xlabel("컬럼")
ax.set_ylabel("결측 건수")
ax.set_title(f"그림 6. 결측 처리 전후 비교 (분모: 전체 {N:,}행, 행 삭제 없음)")
ax.set_ylim(0, max(bv) * 1.2)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "06_정제전후_비교.png", dpi=150)
plt.close()
