"""넷플릭스 결측 진단 차트 생성. CSV 경로만 바꾸면 어느 버전이든 동작."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent   # coursework/netflix
CSV = BASE / "data" / "netflix_titles.csv"
OUT = BASE / "output"
OUT.mkdir(parents=True, exist_ok=True)


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(CSV)
N = len(df)
n_movie = (df["type"] == "Movie").sum()
n_tv = (df["type"] == "TV Show").sum()
MISS = ["director", "cast", "country", "date_added", "rating"]

# ── 차트 1. 콘텐츠 유형별 결측률 (건수 병기) ─────────────────────────
cols = ["director", "cast", "country"]
rate, cnt = {}, {}
for t in ["Movie", "TV Show"]:
    s = df[df["type"] == t]
    rate[t] = [s[c].isna().mean() * 100 for c in cols]
    cnt[t] = [s[c].isna().sum() for c in cols]

x = np.arange(len(cols))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - w / 2, rate["Movie"], w, label=f"Movie (n={n_movie:,})", color="#4C72B0")
b2 = ax.bar(x + w / 2, rate["TV Show"], w, label=f"TV Show (n={n_tv:,})", color="#C9A227")

for bars, t in [(b1, "Movie"), (b2, "TV Show")]:
    for bar, r, c in zip(bars, rate[t], cnt[t]):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 1.5,
                f"{r:.2f}%\n({c:,}건)", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x, cols)
ax.set_xlabel("컬럼")
ax.set_ylabel("해당 유형 내 결측률 (%)")
ax.set_title(f"그림 1. 콘텐츠 유형별 결측률 (분모: Movie {n_movie:,}행 / TV Show {n_tv:,}행)")
ax.set_ylim(0, 108)
ax.legend(title="콘텐츠 유형")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "01_결측률_유형별.png", dpi=150)
plt.close()

# ── 차트 2. 전체 컬럼별 결측 현황 ───────────────────────────────────
m = df[MISS].isna().sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(m.index, m.values, color="#4C72B0")
for bar, v in zip(bars, m.values):
    ax.text(v + N * 0.005, bar.get_y() + bar.get_height() / 2,
            f"{v:,}건 ({v / N * 100:.2f}%)", va="center", fontsize=9)
ax.set_xlabel(f"결측 건수 (분모: 전체 {N:,}행)")
ax.set_ylabel("컬럼")
ax.set_title(f"그림 2. 컬럼별 결측 현황 (전체 {N:,}행)")
ax.set_xlim(0, m.max() * 1.25)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "02_결측_컬럼별.png", dpi=150)
plt.close()

# ── 차트 3. 행당 결측 컬럼 수 분포 ──────────────────────────────────
nm = df[MISS].isna().sum(axis=1).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(nm.index.astype(str), nm.values, color=["#55A868"] + ["#C44E52"] * (len(nm) - 1))
for bar, v in zip(bars, nm.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + N * 0.008,
            f"{v:,}행\n({v / N * 100:.1f}%)", ha="center", fontsize=9)
ax.set_xlabel("한 행에서 결측인 컬럼 수 (개)")
ax.set_ylabel("행 수")
ax.set_title(f"그림 3. 행당 결측 컬럼 수 분포 (전체 {N:,}행)")
ax.set_ylim(0, nm.max() * 1.2)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "03_행당_결측수.png", dpi=150)
plt.close()

print(f"[{CSV.name}] {N:,}행 (Movie {n_movie:,} / TV Show {n_tv:,})")
for c in MISS:
    mv = df[df.type == "Movie"][c].isna()
    tv = df[df.type == "TV Show"][c].isna()
    print(f"  {c:11s} 전체 {df[c].isna().sum():>5,}건({df[c].isna().mean()*100:5.2f}%) | "
          f"Movie {mv.sum():>5,}({mv.mean()*100:5.2f}%) | TV {tv.sum():>5,}({tv.mean()*100:5.2f}%)")
print("저장:", OUT)

# ── 차트 4. 결측 조합 상위 패턴 ─────────────────────────────────────
pat = df[MISS].isna().apply(
    lambda r: " + ".join([c for c, v in zip(MISS, r) if v]), axis=1
)
has_miss = pat[pat != ""]
vc = has_miss.value_counts()
D = len(has_miss)  # 분모: 결측이 있는 행

top = vc.head(10).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top.index, top.values, color="#C44E52")
for bar, v in zip(bars, top.values):
    ax.text(v + D * 0.008, bar.get_y() + bar.get_height() / 2,
            f"{v:,}건 ({v / D * 100:.1f}%)", va="center", fontsize=9)
ax.set_xlabel(f"행 수 (분모: 결측이 있는 {D:,}행)")
ax.set_ylabel("결측 컬럼 조합")
ax.set_title(f"그림 4. 결측 조합 상위 패턴 (결측 보유 {D:,}행 기준)")
ax.set_xlim(0, top.max() * 1.25)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "04_결측조합_패턴.png", dpi=150)
plt.close()

print(f"\n[결측 조합 패턴] 분모 = 결측 보유 {D:,}행")
for k, v in vc.items():
    print(f"  {k:38s} {v:>5,}건 ({v / D * 100:5.1f}%)")
