"""6주차: 교안(B) 4장~6-3 실습을 10문제 Q&A로 정리한다 (팀 질문은 복합·세부 질문으로 수정).

공통 3문제(연습문제 1) + 팀 질문 7문제 + 추가 분석(6-2 네트워크, 6-3 텍스트).
입력은 5주차 정제본이고 원본 CSV는 변경하지 않는다.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "netflix_titles.csv"
CLEANED = BASE / "output" / "netflix_cleaned.csv"
OUT = BASE / "output" / "week6"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "figure.dpi": 110, "savefig.dpi": 140, "axes.titlesize": 12})
BLUE, ORANGE, GREY, PURPLE, RED, GREEN = "#2878b5", "#f2b134", "#9aa0a6", "#7b2cbf", "#c0392b", "#2a9d8f"
SEED = 42
MPAA = ["G", "PG", "PG-13", "R", "NC-17", "NR", "UR"]
TVR = ["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14", "TV-MA"]
log = []


def say(msg=""):
    print(msg)
    log.append(msg)


def save(fig, name, sup=None):
    if sup:
        fig.suptitle(sup, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, .95])
    else:
        fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def lab(ax, title, x=None, y=None):
    ax.set_title(title)
    if x:
        ax.set_xlabel(f"x축: {x}")
    if y:
        ax.set_ylabel(f"y축: {y}")


def note(ax, text):
    ax.text(.5, -.03, text, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#555")


def dump(obj, name):
    obj.to_csv(TAB / name, encoding="utf-8-sig")


def hbar(ax, s, color, title, xlabel, fmt="{:,}"):
    t = s[::-1]
    ax.barh(range(len(t)), t.values, color=color)
    ax.set_yticks(range(len(t)), t.index, fontsize=9)
    for i, v in enumerate(t.values):
        ax.text(v + max(t.values) * .012, i, fmt.format(v), va="center", fontsize=8)
    ax.set_xlim(0, max(t.values) * 1.18)
    ax.set(title=title, xlabel=xlabel)


raw = pd.read_csv(RAW)
df = pd.read_csv(CLEANED)
assert len(raw) == len(df) == 7787 and raw["show_id"].equals(df["show_id"])
N = len(df)
df["rating_imputed"] = raw["rating"].isna().astype(int)

# ══════════════════════════════════════════════════════════════════
say("=" * 70)
say("0. 전처리")
say("=" * 70)
FMT = "%B %d, %Y"
naive = pd.to_datetime(raw["date_added"], format=FMT, errors="coerce")
fixed = pd.to_datetime(df["date_added"], format=FMT, errors="coerce")
df["date_added_parsed"] = fixed
df["added_year"] = fixed.dt.year
df["added_month"] = fixed.dt.month
parts = df["duration"].str.extract(r"^(\d+)\s+(min|Season|Seasons)$")
df["duration_value"] = pd.to_numeric(parts[0], errors="coerce")
df["duration_unit"] = parts[1]
mv = df["type"].eq("Movie")
mixed_mean = pd.to_numeric(df["duration"].str.extract(r"^(\d+)")[0]).mean()
df["rating_system"] = np.where(df["rating"].isin(MPAA), "MPAA(영화 등급)",
                               np.where(df["rating"].isin(TVR), "TV 등급", "기타"))
say(f"날짜 파싱 성공: 공백 정리 전 {naive.notna().sum():,} → 후 {fixed.notna().sum():,}, 원문 결측 {int(df['date_added'].isna().sum())}")
say(f"관측 구간 {fixed.min():%Y-%m-%d} ~ {fixed.max():%Y-%m-%d}")
say(f"단위 무시 평균 {mixed_mean:.1f} / Movie {df.loc[mv, 'duration_value'].mean():.1f}분 / TV {df.loc[~mv, 'duration_value'].mean():.2f}시즌")
say(f"rating 체계: {df['rating_system'].value_counts().to_dict()}")

src_country = df["country"].where(df["country_imputed"].eq(0))


def to_long(series):
    s = series.dropna()
    return (pd.DataFrame({"show_id": df.loc[s.index, "show_id"], "value": s.str.split(",")})
            .explode("value").assign(value=lambda x: x["value"].str.strip())
            .loc[lambda x: x["value"].ne("")].drop_duplicates())


rel = {"director": to_long(df["director"]), "cast": to_long(df["cast"]),
       "country": to_long(src_country), "genre": to_long(df["listed_in"])}
for k, t in rel.items():
    t.to_csv(TAB / f"rel_{k}.csv", index=False, encoding="utf-8-sig")
    say(f"rel_{k}: 관계 {len(t):,} / 작품 {t['show_id'].nunique():,} / 고유값 {t['value'].nunique():,}")

genre_raw = df["listed_in"].value_counts().head(10)
genre_long = rel["genre"]["value"].value_counts().head(10)
say(f"장르 셀 문자열 고유값 {df['listed_in'].nunique()}개, 펼친 뒤 {rel['genre']['value'].nunique()}개")
first_country = src_country.dropna().str.split(",").str[0].str.strip().value_counts()
all_country = rel["country"]["value"].value_counts()
cmp_c = pd.DataFrame({"첫 국가만": first_country, "전체 펼침": all_country}).fillna(0).astype(int)
cmp_c["차이"] = cmp_c["전체 펼침"] - cmp_c["첫 국가만"]
cmp_c["증가율(%)"] = (cmp_c["차이"] / cmp_c["첫 국가만"] * 100).round(1)
top_c = cmp_c.sort_values("전체 펼침", ascending=False).head(10)
dump(top_c, "p_country_first_vs_long.csv")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
ax = axes[0]
ax.bar(["공백 정리 전", "공백 정리 후"], [naive.notna().sum(), fixed.notna().sum()], color=[GREY, BLUE], width=.5)
for i, v in enumerate([naive.notna().sum(), fixed.notna().sum()]):
    ax.text(i, v + 4, f"{v:,}", ha="center", fontsize=10)
ax.set_ylim(7600, 7810)
lab(ax, "(a) date_added 날짜 변환 성공 건수", "공백 정리 전 / 후", "날짜로 읽힌 작품 수(편)")
ax = axes[1]
vals = [mixed_mean, df.loc[mv, "duration_value"].mean(), df.loc[~mv, "duration_value"].mean()]
ax.bar(["단위 무시\n숫자만 평균", "분리 후\nMovie(분)", "분리 후\nTV(시즌)"], vals, color=[RED, BLUE, ORANGE], width=.55)
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=10)
lab(ax, "(b) duration 단위를 섞었을 때의 평균", "계산 방법", "평균값 (분 또는 시즌)")
ax = axes[2]
rs = df["rating_system"].value_counts()
ax.bar(rs.index, rs.values, color=["#8ecae6", ORANGE][:len(rs)][::-1] if rs.index[0] == "TV 등급" else ["#8ecae6", ORANGE], width=.5)
for i, v in enumerate(rs.values):
    ax.text(i, v + 60, f"{v:,}", ha="center", fontsize=10)
lab(ax, "(c) rating 열에 섞인 두 등급 체계", "등급 체계", "작품 수(편)")
save(fig, "p1_date_duration_rating.png", "0절 전처리 ① 날짜 · 길이 · 등급 정리 결과")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
yy = np.arange(10)
ax.barh(yy, genre_raw.values, color=GREY)
ax.set_yticks(yy, [t if len(t) < 42 else t[:40] + "…" for t in genre_raw.index], fontsize=8)
ax.invert_yaxis()
lab(ax, "(a) 펼치기 전: listed_in 문자열 그대로 센 상위 10", "작품 수(편)", "listed_in 문자열 (장르 조합)")
ax = axes[1]
ax.barh(yy, genre_long.values, color=BLUE)
ax.set_yticks(yy, genre_long.index, fontsize=9)
ax.invert_yaxis()
for i, v in enumerate(genre_long.values):
    ax.text(v + 20, i, f"{v:,}", va="center", fontsize=8)
lab(ax, "(b) 펼친 뒤: 장르 하나씩 센 상위 10", "해당 장르가 붙은 작품 수(편)", "장르")
save(fig, "p2_genre_explode.png", "0절 전처리 ② 장르 열 펼치기 전과 후")

fig, ax = plt.subplots(figsize=(11, 4.5))
yy = np.arange(len(top_c))
ax.barh(yy + .2, top_c["첫 국가만"], .38, color=GREY, label="첫 번째 국가만 사용 (교안 6-1.2 방식)")
ax.barh(yy - .2, top_c["전체 펼침"], .38, color=BLUE, label="전체 국가 펼침 (교안 4-2 방식)")
for i, (a, b, r) in enumerate(zip(top_c["첫 국가만"], top_c["전체 펼침"], top_c["증가율(%)"])):
    ax.text(b + 20, i - .2, f"+{b - a} ({r:.0f}%)", va="center", fontsize=8, color=BLUE)
ax.set_yticks(yy, top_c.index, fontsize=9)
ax.invert_yaxis()
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 3900)
lab(ax, "0절 전처리 ③ 국가 집계 방식 비교: 첫 번째 국가만 vs 전체 펼침 (상위 10개국)", "해당 국가가 포함된 작품 수(편)", "제작 국가")
save(fig, "p3_country_method.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q1. 가장 많이 출연한 배우 Top 10")
say("=" * 70)
cast_cnt = rel["cast"]["value"].value_counts()
top_act = cast_cnt.head(10)
ac = rel["cast"][rel["cast"]["value"].isin(top_act.index)].merge(rel["country"], on="show_id", how="left",
                                                                  suffixes=("", "_c"))
main_c = ac.groupby("value")["value_c"].agg(lambda s: s.value_counts().index[0] if s.notna().any() else "정보 없음")
share_c = ac.groupby("value")["value_c"].agg(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 0)
q1 = pd.DataFrame({"출연 작품 수": top_act, "가장 많은 제작 국가": main_c.reindex(top_act.index),
                   "그 국가 비율(%)": (share_c.reindex(top_act.index) * 100).round(0)})
dump(q1, "q1_actor_top10.csv")
say(f"cast 값이 있는 작품 {rel['cast']['show_id'].nunique():,}편, 고유 배우 {cast_cnt.size:,}명")
say(q1.to_string())
say(f"상위 10명 중 가장 많은 제작 국가가 India인 배우: {int((q1['가장 많은 제작 국가'] == 'India').sum())}명, Japan: {int((q1['가장 많은 제작 국가'] == 'Japan').sum())}명")
say(f"1편만 출연한 배우 {int((cast_cnt == 1).sum()):,}명 ({(cast_cnt == 1).mean():.1%})")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [3, 2]})
cmap = {"India": ORANGE, "Japan": RED}
t = top_act[::-1]
axes[0].barh(range(len(t)), t.values, color=[cmap.get(main_c[a], BLUE) for a in t.index])
axes[0].set_yticks(range(len(t)), t.index, fontsize=9)
for i, a in enumerate(t.index):
    axes[0].text(t[a] + .5, i, f"{t[a]}편  ({main_c[a]})", va="center", fontsize=8)
axes[0].set_xlim(0, 55)
lab(axes[0], f"(a) 출연 작품 수 상위 10명 (cast 있는 작품 {rel['cast']['show_id'].nunique():,}편 기준)", "출연 작품 수(편)", "배우 이름")
for k, c in [("India", ORANGE), ("Japan", RED)]:
    axes[0].scatter([], [], marker="s", c=c, label=f"주 제작국 {k}")
axes[0].legend(fontsize=8, loc="lower right")
dist = cast_cnt.value_counts().sort_index()
dist_b = pd.Series({"1편": dist.loc[1], "2편": dist.loc[2], "3~5편": dist.loc[3:5].sum(),
                    "6~10편": dist.loc[6:10].sum(), "11편 이상": dist.loc[11:].sum()})
axes[1].bar(dist_b.index, dist_b.values, color=GREY)
for i, v in enumerate(dist_b.values):
    axes[1].text(i, v + 300, f"{v:,}", ha="center", fontsize=9)
lab(axes[1], f"(b) 전체 배우 {cast_cnt.size:,}명의 출연 작품 수 분포", "출연 작품 수 구간", "배우 수(명)")
save(fig, "q1_actors.png", "Q1. 가장 많이 출연한 배우는 누구일까?")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q2. TV 쇼는 보통 몇 시즌인가")
say("=" * 70)
tv = df.loc[~mv & df["duration_unit"].isin(["Season", "Seasons"])]
sea = tv["duration_value"]
sv = sea.value_counts().sort_index()
q2 = pd.DataFrame({"작품 수": sv, "비율(%)": (sv / len(sea) * 100).round(1)})
q2["누적 비율(%)"] = q2["비율(%)"].cumsum().round(1)
dump(q2, "q2_tv_seasons.csv")
say(f"TV Show {len(sea):,}편, 평균 {sea.mean():.2f}, 중앙값 {sea.median():.0f}, 최댓값 {sea.max():.0f}")
say(q2.head(8).to_string())
longest = tv.nlargest(3, "duration_value")[["title", "duration_value"]]
say("최장: " + ", ".join(f"{r.title}({r.duration_value:.0f})" for r in longest.itertuples()))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
ax.bar(sv.index, sv.values, color=ORANGE)
for x, v in sv.items():
    if v >= 20:
        ax.text(x, v + 20, f"{v:,}\n({v / len(sea):.1%})", ha="center", fontsize=8)
ax.set_xticks(sv.index.tolist())
ax.set_ylim(0, sv.max() * 1.18)
lab(ax, f"(a) 시즌 수별 TV 쇼 작품 수 (N={len(sea):,})", "시즌 수", "TV 쇼 작품 수(편)")
ax = axes[1]
ax.plot(q2.index, q2["누적 비율(%)"], marker="o", color=PURPLE)
ax.axhline(90, color=GREY, ls="--", lw=1)
ax.text(10, 87, "90%", fontsize=8, color="#666")
ax.set_xticks(sv.index.tolist())
ax.set_ylim(0, 103)
lab(ax, "(b) 시즌 수별 누적 비율 (점선 = 90%)", "시즌 수 (해당 시즌 이하)", "전체 TV 쇼 중 누적 비율(%)")
save(fig, "q2_seasons.png", "Q2. TV 쇼는 보통 몇 시즌으로 구성될까?")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q3. 몇 월에 가장 많이 추가하는가 (월 평균 ±10% 기준)")
say("=" * 70)
added = df.dropna(subset=["added_month"]).copy()
added["added_month"] = added["added_month"].astype(int)
added["added_year"] = added["added_year"].astype(int)
full = added[added["added_year"] <= 2020]
m_full = full["added_month"].value_counts().sort_index()
avg = m_full.mean()
q3 = pd.DataFrame({"작품 수": m_full, "평균 대비(%)": ((m_full / avg - 1) * 100).round(1)})
q3["판정"] = np.select([q3["평균 대비(%)"] >= 10, q3["평균 대비(%)"] <= -10], ["많은 달", "적은 달"], "보통")
q3["Movie"] = full[full.type == "Movie"]["added_month"].value_counts().sort_index()
q3["TV Show"] = full[full.type == "TV Show"]["added_month"].value_counts().sort_index()
dump(q3, "q3_month.csv")
ym = pd.crosstab(added["added_year"], added["added_month"])
dump(ym, "q3_year_month.csv")
yr_top = ym.loc[2016:2020].idxmax(axis=1)
yr_many = {y: [m for m in range(1, 13) if ym.loc[y, m] >= ym.loc[y].mean() * 1.1] for y in range(2016, 2021)}
say(f"2008~2020년 {len(full):,}편, 월 평균 {avg:.1f}편 → 많은 달 기준 {avg * 1.1:.0f}편 이상, 적은 달 {avg * .9:.0f}편 이하")
say(q3.to_string())
say(f"2021년 1월 {int((added.added_year == 2021).sum())}편은 부분 연도라 제외")
say(f"2016~2020 연도별 1위 달: {yr_top.to_dict()}")
say("연도별 '많은 달'(그 해 평균+10%): " + str(yr_many))
cnt_many = pd.Series([m for v in yr_many.values() for m in v]).value_counts().sort_index()
say("5년 중 '많은 달'에 든 횟수: " + str(cnt_many.to_dict()))

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [2, 3]})
ax = axes[0]
col = q3["판정"].map({"많은 달": RED, "적은 달": GREY, "보통": BLUE})
ax.bar(q3.index, q3["작품 수"], color=col)
ax.axhline(avg, color="#333", lw=1)
ax.axhline(avg * 1.1, color=RED, ls="--", lw=1)
ax.axhline(avg * .9, color=GREY, ls="--", lw=1)
ax.text(12.6, avg * 1.1, "+10%", fontsize=8, color=RED, va="center")
ax.text(12.6, avg, f"평균 {avg:.0f}", fontsize=8, va="center")
ax.text(12.6, avg * .9, "-10%", fontsize=8, color="#666", va="center")
for m, v in q3["작품 수"].items():
    ax.text(m, v + 10, f"{v}", ha="center", fontsize=8)
ax.set_xticks(range(1, 13))
ax.set_xlim(.4, 13.6)
ax.set_ylim(0, 900)
lab(ax, f"(a) 월별 추가 작품 수 (2008~2020, N={len(full):,})\n빨강 = 많은 달(평균+10% 이상), 회색 = 적은 달", "추가된 월", "추가된 작품 수(편)")
ax = axes[1]
sns.heatmap(ym.loc[2015:2020], cmap="YlGnBu", annot=True, fmt="d", linewidths=.4, ax=ax, cbar_kws={"label": "칸 색 = 추가된 작품 수(편)"})
lab(ax, "(b) 연도 × 월 추가 작품 수 (2015~2020) — 해마다 같은 달이 많은지 확인", "추가된 월", "추가 연도")
save(fig, "q3_month.png", "Q3. 넷플릭스는 몇 월에 콘텐츠를 가장 많이 추가할까?")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q4. 성인·어린이 등급 비중: 유형 전체 vs 대응 장르")
say("=" * 70)
RGROUP = {**{r: "어린이" for r in ["G", "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G"]},
          **{r: "청소년" for r in ["PG", "TV-PG", "PG-13", "TV-14"]},
          **{r: "성인" for r in ["R", "NC-17", "TV-MA"]},
          **{r: "미분류" for r in ["NR", "UR"]}}
GORDER = ["어린이", "청소년", "성인", "미분류"]
GCOL = {"어린이": GREEN, "청소년": ORANGE, "성인": RED, "미분류": GREY}
rated = df[df["rating_imputed"].eq(0)].copy()
rated["rgroup"] = rated["rating"].map(RGROUP)
tg = pd.crosstab(rated["type"], rated["rgroup"])[GORDER]
tg_pct = tg.div(tg.sum(axis=1), axis=0) * 100
dump(tg_pct.round(1), "q4_type_rgroup_pct.csv")
say(f"유효 {len(rated):,}편 (Movie {int(tg.loc['Movie'].sum()):,}, TV {int(tg.loc['TV Show'].sum()):,})")
say(tg_pct.round(1).to_string())

PAIRS = [("Dramas", "TV Dramas"), ("Comedies", "TV Comedies"), ("Documentaries", "Docuseries"),
         ("International Movies", "International TV Shows"), ("Romantic Movies", "Romantic TV Shows"),
         ("Action & Adventure", "TV Action & Adventure"), ("Horror Movies", "TV Horror"),
         ("Children & Family Movies", "Kids' TV")]
gx = rel["genre"].merge(rated[["show_id", "rgroup"]], on="show_id")


def judge(diff):
    return "TV판 높음" if diff >= 1 else ("영화판 높음" if diff <= -1 else "같음")


prow = []
for mg, tvg in PAIRS:
    a_, b_ = gx[gx.value == mg], gx[gx.value == tvg]
    ad_, bd_ = (a_.rgroup == "성인").mean() * 100, (b_.rgroup == "성인").mean() * 100
    ak_, bk_ = (a_.rgroup == "어린이").mean() * 100, (b_.rgroup == "어린이").mean() * 100
    prow.append({"영화판": mg, "TV판": tvg, "n(영화/TV)": f"{len(a_)}/{len(b_)}",
                 "성인 영화": round(ad_, 1), "성인 TV": round(bd_, 1), "성인 판정": judge(bd_ - ad_),
                 "어린이 영화": round(ak_, 1), "어린이 TV": round(bk_, 1), "어린이 판정": judge(bk_ - ak_)})
pairs = pd.DataFrame(prow)
dump(pairs.set_index("영화판"), "q4_genre_pairs.csv")
say(pairs.to_string(index=False))
say("성인 판정: " + str(pairs["성인 판정"].value_counts().to_dict()))
say("어린이 판정: " + str(pairs["어린이 판정"].value_counts().to_dict()))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2, 3]})
ax = axes[0]
left = np.zeros(2)
for gname in GORDER:
    v = tg_pct.loc[["Movie", "TV Show"], gname].values
    ax.barh(["Movie", "TV Show"], v, left=left, color=GCOL[gname], label=gname)
    for i, (l_, w) in enumerate(zip(left, v)):
        if w >= 4:
            ax.text(l_ + w / 2, i, f"{w:.1f}", ha="center", va="center", fontsize=9, color="white")
    left += v
ax.set_xlim(0, 100)
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12))
lab(ax, "(a) 유형별 등급 그룹 구성 (100% 누적)", "유형 안에서 차지하는 비율(%)", "콘텐츠 유형")
ax = axes[1]
yy = np.arange(len(pairs))[::-1]
for y_, r in zip(yy, pairs.itertuples()):
    ax.plot([r._4, r._5], [y_ + .15] * 2, color=RED, lw=1.5, alpha=.5)
    ax.plot([r._7, r._8], [y_ - .15] * 2, color=GREEN, lw=1.5, alpha=.5)
ax.scatter(pairs["성인 영화"], yy + .15, marker="o", color=RED, s=45, label="성인 · 영화판")
ax.scatter(pairs["성인 TV"], yy + .15, marker="s", color=RED, s=45, facecolors="white", label="성인 · TV판")
ax.scatter(pairs["어린이 영화"], yy - .15, marker="o", color=GREEN, s=45, label="어린이 · 영화판")
ax.scatter(pairs["어린이 TV"], yy - .15, marker="s", color=GREEN, s=45, facecolors="white", label="어린이 · TV판")
ax.set_yticks(yy, [f"{a} / {b}" for a, b in PAIRS], fontsize=8)
ax.set_xlim(-2, 102)
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.45, -.12))
lab(ax, "(b) 대응 장르 8쌍의 성인·어린이 등급 비율: 영화판(●) vs TV판(□)", "장르 안에서 해당 등급 그룹이 차지하는 비율(%)", "대응 장르 쌍 (영화판 / TV판)")
save(fig, "q4_type_rating.png", "Q4. 성인·어린이 등급 비중: 유형 전체 vs 대응 장르끼리")

gen = rel["genre"]["value"].value_counts()
top10g = gen.head(10).index
gr = rel["genre"].merge(rated[["show_id", "rating"]], on="show_id")
ct = pd.crosstab(gr.loc[gr.value.isin(top10g), "value"], gr.loc[gr.value.isin(top10g), "rating"])
ct = ct.loc[top10g, [c for c in MPAA + TVR if c in ct.columns]]
row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
dump(row_pct.round(1), "q4_genre_rating_rowpct.csv")
fig, ax = plt.subplots(figsize=(12, 5.2))
sns.heatmap(row_pct, cmap="Purples", annot=True, fmt=".0f", linewidths=.5, ax=ax, cbar_kws={"label": "칸 색 = 그 장르 안에서 해당 등급 비율(%)"})
ax.axvline(len([c for c in MPAA if c in ct.columns]), color="#333", lw=1.5)
lab(ax, "Q4 보조. 상위 10개 장르 × 14개 시청 등급 (행 비율 %, 각 행의 합 = 100%)", "시청 등급 (세로선 왼쪽 = MPAA 영화 등급, 오른쪽 = TV 등급)", "장르")
save(fig, "q4_genre_rating_heatmap.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q5. 2016~2020 신작·구작 비중과 2016→2020 변화")
say("=" * 70)
Y5 = [2016, 2017, 2018, 2019, 2020]
a5 = added[added.added_year.isin(Y5)].copy()
a5["gap"] = a5["added_year"] - a5["release_year"]
a5["구분"] = np.select([a5["gap"] <= 1, a5["gap"] >= 10], ["신작", "구작"], "중간")
q5 = (a5.groupby(["type", "added_year"])["구분"].value_counts(normalize=True).unstack().fillna(0) * 100)[["신작", "중간", "구작"]]
dump(q5.round(1), "q5_gap_share.csv")
say(f"대상 {len(a5):,}편")
say(q5.round(1).to_string())
q5chg = []
for t_ in ["Movie", "TV Show"]:
    for k in ["신작", "구작"]:
        s = q5.loc[t_, k]
        d_ = s.loc[2020] - s.loc[2016]
        q5chg.append({"유형": t_, "구분": k, "2016": round(s.loc[2016], 1), "2020": round(s.loc[2020], 1),
                      "변화(%p)": round(d_, 1), "판정": "증가" if d_ >= 1 else ("감소" if d_ <= -1 else "유지"),
                      "정점": f"{s.idxmax()}년 {s.max():.1f}%", "최저": f"{s.idxmin()}년 {s.min():.1f}%"})
q5chg = pd.DataFrame(q5chg)
dump(q5chg.set_index(["유형", "구분"]), "q5_change.csv")
say(q5chg.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [2, 1.4, 1.4]})
ax = axes[0]
for t_, c in [("Movie", BLUE), ("TV Show", ORANGE)]:
    s = q5.loc[t_, "신작"]
    ax.plot(Y5, s.values, marker="o", color=c, lw=2.2, label=f"{t_} 신작")
    for x, v in zip(Y5, s.values):
        ax.text(x, v + 2, f"{v:.1f}", ha="center", fontsize=8, color=c)
    o = q5.loc[t_, "구작"]
    ax.plot(Y5, o.values, marker="s", ls="--", color=c, lw=1.4, alpha=.8, label=f"{t_} 구작(10년+)")
    for x, v in zip(Y5, o.values):
        ax.text(x, v - 5, f"{v:.1f}", ha="center", fontsize=7, color=c, alpha=.8)
ax.set_xticks(Y5)
ax.set_ylim(-6, 90)
ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.14))
lab(ax, "(a) 추가 연도별 신작·구작 비중 (실선 = 신작, 점선 = 구작)", "추가 연도", "그 해 추가된 작품 중 비율(%)")
for ax, t_ in zip(axes[1:], ["Movie", "TV Show"]):
    s = q5.loc[t_]
    bottom = np.zeros(len(Y5))
    for col_, c in zip(["신작", "중간", "구작"], [BLUE if t_ == "Movie" else ORANGE, "#c9d6df", GREY]):
        ax.bar([str(y) for y in Y5], s[col_].values, bottom=bottom, color=c, label=col_)
        bottom += s[col_].values
    ax.set_ylim(0, 100)
    kor = "영화" if t_ == "Movie" else "TV 쇼"
    lab(ax, f"({'b' if t_ == 'Movie' else 'c'}) {kor}: 신작·중간·구작 구성 (100%)", "추가 연도", f"그 해 추가된 {kor} 중 비율(%)")
    ax.legend(fontsize=7, loc="lower left")
save(fig, "q5_trend.png", "Q5. 2016~2020년 신작(0~1년)·구작(10년 이상) 비중: 영화 vs TV 쇼")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q6. 장르 비중(분모=작품 수) 2016 vs 2020")
say("=" * 70)
gy = rel["genre"].merge(a5[["show_id", "added_year"]], on="show_id")
gcnt = gy.groupby(["added_year", "value"])["show_id"].nunique().unstack(fill_value=0)
wden = a5.groupby("added_year").size()
gpct = gcnt.div(wden, axis=0) * 100
chg = pd.DataFrame({"2016 비중": gpct.loc[2016], "2020 비중": gpct.loc[2020],
                    "2016 건수": gcnt.loc[2016], "2020 건수": gcnt.loc[2020]})
chg["비중 변화(%p)"] = chg["2020 비중"] - chg["2016 비중"]
chg["건수 변화"] = chg["2020 건수"] - chg["2016 건수"]
chg["건수 방향"] = np.where(chg["건수 변화"] > 0, "증가", np.where(chg["건수 변화"] < 0, "감소", "같음"))
chg["비중 방향"] = np.select([chg["비중 변화(%p)"] >= 1, chg["비중 변화(%p)"] <= -1], ["증가", "감소"], "유지")
chg["같은 방향"] = np.where(chg["비중 방향"] == chg["건수 방향"], "예", "아니오")
chg = chg[(chg["2016 건수"] >= 15) | (chg["2020 건수"] >= 60)].sort_values("비중 변화(%p)")
dump(chg.round(1), "q6_genre_change.csv")
say(f"작품 수: 2016 {wden[2016]}, 2020 {wden[2020]} ({wden[2020] / wden[2016]:.1f}배), 비교 장르 {len(chg)}개")
say(chg.round(1).to_string())
say("비중 방향: " + str(chg["비중 방향"].value_counts().to_dict()) + " / 같은 방향: " + str(chg["같은 방향"].value_counts().to_dict()))
movers = pd.concat([chg.head(5), chg.tail(5)])
gtype = rel["genre"].merge(df[["show_id", "type"]], on="show_id").groupby("value")["type"].agg(lambda s: (s == "Movie").mean() * 100)
say("상위 변화 장르의 영화 비율: " + ", ".join(f"{g} {gtype[g]:.0f}%" for g in movers.index))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
ys = gpct.loc[2020, movers.index].sort_values()
lab_y, prev = {}, -99
for g_, v in ys.items():
    prev = max(v, prev + 1.1)
    lab_y[g_] = prev
for g_ in movers.index:
    c = RED if movers.loc[g_, "비중 변화(%p)"] < 0 else BLUE
    ax.plot(Y5, gpct.loc[Y5, g_].values, marker="o", color=c, alpha=.85, lw=1.6)
    ax.text(2020.12, lab_y[g_], f"{g_} ({movers.loc[g_, '비중 변화(%p)']:+.1f}%p)", fontsize=7.5, va="center", color=c)
ax.set_xticks(Y5)
ax.set_xlim(2015.8, 2021.9)
lab(ax, "(a) 비중 변화가 큰 10개 장르의 연도별 비중 (파랑 = 상승, 빨강 = 하락)", "추가 연도", "그 해 추가 작품 중 해당 장르 포함 비율(%)")
ax = axes[1]
yy = np.arange(len(movers))
ax.barh(yy - .2, movers["2016 건수"], .38, color=GREY, label="2016 건수")
ax.barh(yy + .2, movers["2020 건수"], .38, color=[RED if v < 0 else BLUE for v in movers["비중 변화(%p)"]], label="2020 건수")
for i, (a_, b_) in enumerate(zip(movers["2016 건수"], movers["2020 건수"])):
    ax.text(b_ + 6, i + .2, f"{a_}→{b_}", va="center", fontsize=7.5)
ax.set_yticks(yy, movers.index, fontsize=8)
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, 700)
lab(ax, "(b) 같은 10개 장르의 추가 건수 (회색 = 2016, 색 = 2020)", "추가된 작품 수(편)", "장르")
save(fig, "q6_genre_trend.png", "Q6. 2016년 대비 2020년 장르 비중 변화와 추가 건수")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q7. 미국 비중과 감소한 해의 최대 증가 국가")
say("=" * 70)
cy = rel["country"].merge(added[["show_id", "added_year"]], on="show_id")
cy = cy[cy.added_year.isin(Y5)]
cden = cy.groupby("added_year")["show_id"].nunique()
cshare = cy.groupby(["added_year", "value"]).size().unstack(fill_value=0).div(cden, axis=0) * 100
us = cshare["United States"]
q7 = pd.DataFrame({"국가 있는 작품": cden, "미국 비중(%)": us.round(1),
                   "전년 대비(%p)": us.diff().round(1)})
q7["판정"] = q7["전년 대비(%p)"].map(lambda d_: "" if pd.isna(d_) else ("증가" if d_ >= 1 else ("감소" if d_ <= -1 else "유지")))
tops = {}
for y in Y5[1:]:
    dd = (cshare.loc[y] - cshare.loc[y - 1]).drop("United States").sort_values(ascending=False)
    tops[y] = dd
    q7.loc[y, "증가 1위 비미국 국가"] = f"{dd.index[0]} ({dd.iloc[0]:+.1f}%p)"
    q7.loc[y, "2위"] = f"{dd.index[1]} ({dd.iloc[1]:+.1f}%p)"
dump(q7, "q7_us_share.csv")
say(q7.to_string())
ind_cnt = cy[cy.value == "India"].groupby("added_year").size()
us_cnt = cy[cy.value == "United States"].groupby("added_year").size()
say(f"건수 India {ind_cnt.to_dict()} / US {us_cnt.to_dict()}")
ind_ids = set(rel["country"].loc[rel["country"]["value"].eq("India"), "show_id"])
mv5 = a5[a5.type == "Movie"].assign(old=lambda x: x["gap"] >= 10, india=lambda x: x.show_id.isin(ind_ids))
link = mv5.groupby("added_year").apply(lambda s: pd.Series({
    "영화 구작 수": int(s.old.sum()), "구작 중 인도(%)": round(s[s.old].india.mean() * 100, 1),
    "인도 영화 중 구작(%)": round(s[s.india].old.mean() * 100, 1),
    "비인도 영화 중 구작(%)": round(s[~s.india].old.mean() * 100, 1)}), include_groups=False)
dump(link, "q7_link_india_old_movies.csv")
say("연결(영화 구작 × 인도):\n" + link.to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [2, 2, 2]})
ax = axes[1]
ax.plot(Y5, us.values, marker="o", color=BLUE, lw=2.4, label="United States")
ax.plot(Y5, cshare.loc[Y5, "India"].values, marker="o", color=ORANGE, lw=2.4, label="India")
for col_, c in [("United States", BLUE), ("India", ORANGE)]:
    for x, v in zip(Y5, cshare.loc[Y5, col_]):
        ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=c)
ax.set_xticks(Y5)
ax.set_ylim(0, 60)
ax.legend(fontsize=9)
lab(ax, "(b) 미국·인도 작품의 연도별 비중", "추가 연도", "그 해 추가 작품 중 해당 국가 포함 비율(%)")
ax = axes[0]
xs = np.arange(4)
d_us = us.diff().loc[2017:].values
d_top = [tops[y].iloc[0] for y in Y5[1:]]
ax.bar(xs - .2, d_us, .38, color=[RED if v <= -1 else (BLUE if v >= 1 else GREY) for v in d_us], label="미국 전년 대비")
ax.bar(xs + .2, d_top, .38, color=ORANGE, label="그 해 증가 1위 비미국 국가")
for i, y in enumerate(Y5[1:]):
    ax.text(i + .2, d_top[i] + .4, tops[y].index[0], ha="center", fontsize=7.5, rotation=0)
    ax.text(i - .2, d_us[i] + (.4 if d_us[i] >= 0 else -1.2), f"{d_us[i]:+.1f}", ha="center", fontsize=8)
ax.axhline(0, color="#333", lw=.8)
ax.axhspan(-1, 1, color="#eee", zorder=0)
ax.set_xticks(xs, [str(y) for y in Y5[1:]])
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(-10, 13)
lab(ax, "(a) 전년 대비 비중 변화 — 회색 띠 = ±1%p(유지)", "추가 연도", "전년 대비 비중 변화(%p)")
ax = axes[2]
ax.plot(Y5, link["구작 중 인도(%)"].values, marker="o", color=ORANGE, lw=2, label="영화 구작 중 인도 작품 비율")
ax.plot(Y5, q5.loc["Movie", "구작"].values, marker="s", color=BLUE, lw=2, ls="--", label="영화 중 구작 비율 (Q5)")
for x, v in zip(Y5, link["구작 중 인도(%)"]):
    ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=ORANGE)
ax.set_xticks(Y5)
ax.set_ylim(0, 55)
ax.legend(fontsize=8)
lab(ax, "(c) 연결: 영화 구작 증가와 인도 작품", "추가 연도", "비율(%)")
save(fig, "q7_country_trend.png", "Q7. 미국 비중이 줄어든 해에 비중이 가장 크게 오른 비미국 국가")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q8. 상위 10개국: 다수 유형 × 최다 장르로 묶기")
say("=" * 70)
cty = rel["country"]["value"].value_counts()
top10c = cty.head(10).index
ctype = rel["country"].merge(df[["show_id", "type"]], on="show_id")
tvp = (ctype.groupby("value")["type"].apply(lambda s: (s == "TV Show").mean() * 100)).loc[top10c]
cg_all = rel["country"][rel["country"].value.isin(top10c)].merge(rel["genre"], on="show_id", suffixes=("_c", "_g"))
gc = cg_all.groupby(["value_c", "value_g"]).size()
REGION_TAGS = {g for g in rel["genre"]["value"].unique()
               if g.startswith("International") or g in {"British TV Shows", "Korean TV Shows", "Spanish-Language TV Shows"}}
top_g = gc.groupby(level=0).idxmax().apply(lambda t: t[1])
top_n = gc.groupby(level=0).max()
gc_content = gc[~gc.index.get_level_values(1).isin(REGION_TAGS)]
top_cg = gc_content.groupby(level=0).idxmax().apply(lambda t: t[1])
q8 = pd.DataFrame({"작품 수": cty.loc[top10c], "TV 비율(%)": tvp.round(1),
                   "다수 유형": np.where(tvp > 50, "TV 쇼", "영화"),
                   "최다 장르": top_g.loc[top10c], "그 장르 작품 수": top_n.loc[top10c],
                   "참고: 지역 태그 제외 최다": top_cg.loc[top10c]})
q8["그룹"] = q8["다수 유형"] + " + " + q8["최다 장르"]
dump(q8, "q8_country_profile.csv")
say(q8.to_string())
say("지역 태그로 분류한 장르: " + ", ".join(sorted(REGION_TAGS)))
say("그룹: " + str(q8.groupby("그룹").apply(lambda s: list(s.index), include_groups=False).to_dict()))
uk_tv = rel["genre"].merge(df[["show_id", "type"]], on="show_id")
uk_ids = set(rel["country"].loc[rel["country"]["value"].eq("United Kingdom"), "show_id"])
ukm = uk_tv[uk_tv.show_id.isin(uk_ids)]
say(f"영국 TV {int((df.show_id.isin(uk_ids) & ~mv).sum())}편 중 British TV Shows {int(((ukm.value == 'British TV Shows')).sum())}편; "
    f"영국 영화 최다 장르 {ukm[ukm.type == 'Movie'].value.value_counts().head(3).to_dict()}")

cols = list(dict.fromkeys(list(top10g) + list(q8["최다 장르"])))
cg_sub = cg_all[cg_all.value_g.isin(cols)]
cgt = pd.crosstab(cg_sub["value_c"], cg_sub["value_g"]).reindex(index=top10c, columns=cols, fill_value=0)
cg_pct = cgt.div(cty.loc[top10c], axis=0) * 100
order_c = q8.sort_values(["다수 유형", "최다 장르", "TV 비율(%)"], ascending=[False, True, False]).index
fig, axes = plt.subplots(1, 2, figsize=(17, 5.8), gridspec_kw={"width_ratios": [4, 1.5]})
xlab = [g.replace(" Movies", "\nMovies").replace(" TV Shows", "\nTV Shows").replace(" & ", " &\n") for g in cols]
sns.heatmap(cg_pct.loc[order_c], cmap="Greens", annot=True, fmt=".0f", linewidths=.5, ax=axes[0],
            xticklabels=xlab, cbar_kws={"label": "칸 색 = 그 나라 작품 중 해당 장르 포함 비율(%)"})
for i, c_ in enumerate(order_c):
    j = cols.index(q8.loc[c_, "최다 장르"])
    axes[0].add_patch(plt.Rectangle((j, i), 1, 1, fill=False, ec=RED, lw=2.2))
lab(axes[0], "(a) 국가 × 장르 포함 비율 (빨간 테두리 = 그 나라의 최다 장르)", "장르", "제작 국가")
axes[0].tick_params(axis="x", labelsize=7.5, rotation=0)
ax = axes[1]
yy = np.arange(len(order_c))
tv_o = q8.loc[order_c, "TV 비율(%)"]
ax.barh(yy, 100 - tv_o, color=BLUE, label="Movie")
ax.barh(yy, tv_o, left=100 - tv_o, color=ORANGE, label="TV Show")
ax.axvline(50, color="#333", ls="--", lw=1)
for i, c_ in enumerate(order_c):
    ax.text(102, i, q8.loc[c_, "그룹"].replace(" + ", "\n+ "), va="center", fontsize=6.8)
ax.set_yticks(yy, order_c, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 160)
ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.35, -.14))
lab(ax, "(b) 영화·TV 쇼 비율과 그룹 (점선 = 50%)", "그 나라 작품 중 비율(%)", "제작 국가")
save(fig, "q8_country_genre.png", "Q8. 상위 10개국을 다수 유형 × 최다 장르로 묶으면")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q9. 성인 vs 어린이 등급 영화 러닝타임 (평균·중앙값), C&F 제외")
say("=" * 70)
mr = rated.loc[rated["type"].eq("Movie"), ["show_id", "rating", "rgroup", "duration_value", "rating_system"]].copy()
kid_ids = set(rel["genre"].loc[rel["genre"]["value"].eq("Children & Family Movies"), "show_id"])
mr["is_kids"] = mr["show_id"].isin(kid_ids)
rows = []
for lab_, s in [("전체", mr), ("C&F 제외", mr[~mr.is_kids])]:
    for gname in ["어린이", "성인"]:
        v = s.loc[s.rgroup == gname, "duration_value"]
        rows.append({"범위": lab_, "그룹": gname, "n": len(v), "평균": round(v.mean(), 1), "중앙값": v.median()})
q9 = pd.DataFrame(rows)
q9["성인-어린이 평균차"] = q9.groupby("범위")["평균"].transform(lambda s: s.iloc[1] - s.iloc[0]).round(1)
q9["성인-어린이 중앙값차"] = q9.groupby("범위")["중앙값"].transform(lambda s: s.iloc[1] - s.iloc[0])
dump(q9.set_index(["범위", "그룹"]), "q9_group_runtime.csv")
say(q9.to_string(index=False))
order = [r for r in MPAA + TVR if r in mr["rating"].unique()]
grp = mr.groupby("rating")["duration_value"].agg(n="count", mean="mean", median="median").loc[order]
grp["C&F 비율(%)"] = (mr.groupby("rating")["is_kids"].mean().loc[order] * 100).round(1)
grp = grp.join(mr[~mr.is_kids].groupby("rating")["duration_value"].agg(n_제외="count", mean_제외="mean", median_제외="median"))
dump(grp.round(1), "q9_rating_runtime.csv")
say(grp.round(1).to_string())
kc = mr[(mr.rgroup == "어린이")]
say("어린이 그룹 구성(전체→제외): " + str(kc.rating.value_counts().to_dict()) + " → " + str(kc[~kc.is_kids].rating.value_counts().to_dict()))

show = [r for r in order if grp.loc[r, "n"] >= 30]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
sns.boxplot(data=mr[mr.rating.isin(show)], x="rating", y="duration_value", order=show,
            hue="rating_system", palette={"MPAA(영화 등급)": "#8ecae6", "TV 등급": ORANGE},
            dodge=False, fliersize=1.8, ax=ax)
for i, r in enumerate(show):
    ax.text(i, -16, f"n={int(grp.loc[r, 'n']):,}", ha="center", fontsize=7.5, color="#555")
    ax.text(i, 318, RGROUP[r], ha="center", fontsize=7.5, color=GCOL[RGROUP[r]])
ax.axvline(len([r for r in show if r in MPAA]) - .5, color="#555", lw=1.2)
ax.set_ylim(-28, 330)
ax.legend(title="", loc="upper left", fontsize=8, bbox_to_anchor=(0, .93))
lab(ax, "(a) 시청 등급별 영화 상영시간 분포 (30편 이상 등급)", "시청 등급 (위 글자 = 등급 그룹, 세로선 왼쪽 MPAA / 오른쪽 TV 등급)", "영화 상영시간(분)")
ax = axes[1]
xs = np.arange(2)
w = .2
for j, (lab_, metric, c, h) in enumerate([("전체", "평균", GREEN, ""), ("전체", "중앙값", GREEN, "//"),
                                          ("C&F 제외", "평균", PURPLE, ""), ("C&F 제외", "중앙값", PURPLE, "//")]):
    vals = [q9[(q9.범위 == lab_) & (q9.그룹 == g_)][metric].iloc[0] for g_ in ["어린이", "성인"]]
    bars = ax.bar(xs + (j - 1.5) * w, vals, w, color=c, alpha=.85 if metric == "평균" else .5, hatch=h,
                  edgecolor="white", label=f"{lab_} {metric}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}" if metric == "평균" else f"{v:g}", ha="center", fontsize=7.5)
ns = [f"어린이 등급\n(n={q9.iloc[0]['n']} → {q9.iloc[2]['n']})", f"성인 등급\n(n={q9.iloc[1]['n']:,} → {q9.iloc[3]['n']:,})"]
ax.set_xticks(xs, ns, fontsize=9)
ax.set_ylim(0, 120)
ax.legend(fontsize=8, ncol=2, loc="upper left")
lab(ax, "(b) 어린이·성인 등급의 평균·중앙값 (초록 = 전체, 보라 = C&F 제외)", "등급 그룹 (작품 수: 전체 → C&F 제외)", "영화 상영시간(분)")
save(fig, "q9_rating_runtime.png", "Q9. 성인 등급 영화가 어린이 등급보다 긴가? (Children & Family 제외 비교)")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("Q10. 장르별 시즌 차이: 미국/비미국 나눠도 유지되는가")
say("=" * 70)
from scipy.stats import spearmanr

tvd = df.loc[~mv, ["show_id", "duration_value"]]
us_ids = set(rel["country"].loc[rel["country"]["value"].eq("United States"), "show_id"])
tvd = tvd[tvd.show_id.isin(set(rel["country"]["show_id"]))].assign(
    구분=lambda x: np.where(x.show_id.isin(us_ids), "미국", "비미국"))
cs = rel["country"].merge(tvd[["show_id", "duration_value"]], on="show_id")
cstat = cs.groupby("value")["duration_value"].agg(n="count", mean="mean", 일시즌=lambda s: (s == 1).mean() * 100)
cstat = cstat[cstat.n >= 50].sort_values("mean")
dump(cstat.round(2), "q10_country_seasons.csv")
gt = rel["genre"].merge(tvd, on="show_id")
gm = gt.groupby(["value", "구분"])["duration_value"].agg(["count", "mean"]).unstack()
gm.columns = [f"{a}_{b}" for a, b in gm.columns]
gm = gm[(gm["count_미국"] >= 30) & (gm["count_비미국"] >= 30)].copy()
gm["차이"] = gm["mean_미국"] - gm["mean_비미국"]
gm["순위_미국"] = gm["mean_미국"].rank(ascending=False).astype(int)
gm["순위_비미국"] = gm["mean_비미국"].rank(ascending=False).astype(int)
gm["순위_전체"] = gt[gt.value.isin(gm.index)].groupby("value")["duration_value"].mean().rank(ascending=False).astype(int)
gm = gm.sort_values("mean_미국", ascending=False)
rho = spearmanr(gm["mean_미국"], gm["mean_비미국"])
dump(gm.round(2), "q10_genre_us_seasons.csv")
say(cstat.round(2).to_string())
say(gm.round(2).to_string())
say(f"장르 {len(gm)}개, 미국 vs 비미국 장르 평균의 스피어만 상관 {rho.correlation:.2f} (p={rho.pvalue:.2f}) → 기준 0.7 {'이상(유지)' if rho.correlation >= .7 else '미만(유지되지 않음)'}")
say(f"같은 장르 안에서 미국이 더 긴 장르 {int((gm['차이'] > 0).sum())}/{len(gm)}, 차이 범위 {gm['차이'].min():.2f}~{gm['차이'].max():.2f}")
say(f"미국 {int((tvd.구분 == '미국').sum())}편 평균 {tvd[tvd.구분 == '미국'].duration_value.mean():.2f}, 비미국 {int((tvd.구분 == '비미국').sum())}편 {tvd[tvd.구분 == '비미국'].duration_value.mean():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1.6, 2.4, 2]})
hbar(axes[0], cstat["mean"].round(2), ORANGE, "", "", fmt="{:.2f}")
lab(axes[0], "(a) 국가별 TV 쇼 평균 시즌 수 (50편 이상 국가)", "평균 시즌 수", "제작 국가")
ax = axes[1]
gs = gm.sort_values("차이")
yy = np.arange(len(gs))
ax.barh(yy + .2, gs["mean_미국"], .38, color=BLUE, label="미국")
ax.barh(yy - .2, gs["mean_비미국"], .38, color=ORANGE, label="비미국")
for i, (a_, b_, na, nb) in enumerate(zip(gs["mean_미국"], gs["mean_비미국"], gs["count_미국"], gs["count_비미국"])):
    ax.text(a_ + .03, i + .2, f"{a_:.2f} (n={int(na)})", va="center", fontsize=7)
    ax.text(b_ + .03, i - .2, f"{b_:.2f} (n={int(nb)})", va="center", fontsize=7)
ax.set_yticks(yy, gs.index, fontsize=8)
ax.set_xlim(0, 4)
ax.legend(fontsize=8, loc="lower right")
lab(ax, "(b) 같은 장르 안에서 미국 vs 비미국", "평균 시즌 수", "TV 장르")
ax = axes[2]
ax.scatter(gm["mean_비미국"], gm["mean_미국"], s=50, color=PURPLE)
for g_, r in gm.iterrows():
    ax.annotate(g_.replace(" TV Shows", "").replace("TV ", ""), (r["mean_비미국"], r["mean_미국"]),
                textcoords="offset points", xytext=(4, 3), fontsize=7)
ax.plot([1, 3.3], [1, 3.3], color=RED, ls="--", lw=1, label="미국 = 비미국")
ax.legend(fontsize=8, loc="lower right")
ax.set(xlim=(1, 2.2), ylim=(1, 3.3))
lab(ax, f"(c) 장르별 평균 시즌 수 비교 (점 = 장르 하나, 스피어만 상관 {rho.correlation:.2f})", "비미국 작품의 평균 시즌 수", "미국 작품의 평균 시즌 수")
save(fig, "q10_genre_country_seasons.png", "Q10. 장르별 시즌 수 차이: 미국 / 비미국으로 나눠도 유지되는가")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("추가 분석 1. 감독–배우 네트워크 (6-2)")
say("=" * 70)
dc = rel["director"].merge(rel["cast"], on="show_id", suffixes=("_d", "_a"))
top30 = rel["director"]["value"].value_counts().head(30).index
edges = dc[dc.value_d.isin(top30)][["value_d", "value_a"]].drop_duplicates()
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
deg = pd.Series(nx.degree_centrality(G)).sort_values(ascending=False)
top15 = deg.head(15)
dump(top15.rename("연결 중심성"), "x1_degree_centrality_top15.csv")
say(f"노드 {G.number_of_nodes():,}, 간선 {G.number_of_edges():,}, 상위15 중 감독 {int(top15.index.isin(top30).sum())}")
say(top15.round(4).to_string())
comps = sorted(nx.connected_components(G), key=len, reverse=True)
say(f"연결 덩어리 {len(comps)}개, 가장 큰 덩어리 노드 {len(comps[0])}, 그 안의 감독 {len(set(top30) & comps[0])}명")

pos = nx.kamada_kawai_layout(G)
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
is_dir = [n in set(top30) for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[120 if d else 8 for d in is_dir],
                       node_color=[RED if d else "#8ecae6" for d in is_dir])
nx.draw_networkx_edges(G, pos, ax=ax, alpha=.12, edge_color="grey")
for n in deg.head(8).index:
    x, y_ = pos[n]
    ax.annotate(n, (x, y_), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="#bbb", alpha=.9),
                arrowprops=dict(arrowstyle="-", color="#999", lw=.8))
ax.set_title(f"(a) 감독–배우 협업 네트워크 (감독 30명, 노드 {G.number_of_nodes():,} / 간선 {G.number_of_edges():,})\n점 = 사람(빨강 감독, 하늘 배우), 선 = 같은 작품 참여")
ax.axis("off")
note(ax, "x축·y축 없음: 점의 위치는 연결이 많은 사람끼리 가깝게 놓은 배치이며 좌표 값에는 의미가 없음")
ax = axes[1]
t = top15[::-1]
ax.barh(range(len(t)), t.values, color=[RED if n in set(top30) else "#8ecae6" for n in t.index])
ax.set_yticks(range(len(t)), t.index, fontsize=8)
for i, v in enumerate(t.values):
    ax.text(v + .0004, i, f"{v:.4f}", va="center", fontsize=7.5)
lab(ax, "(b) 연결 중심성 상위 15명 (빨강 = 감독)", "연결 중심성 (연결된 사람 수 ÷ 전체 사람 수)", "감독 이름")
save(fig, "x1_network.png", "추가 분석 1. 감독–배우 협업 네트워크 (교안 6-2)")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("추가 분석 2. 설명문 텍스트 (6-3)")
say("=" * 70)
STOP = set(STOPWORDS) | set("""his her their its who when after into while about over two one new
them they where what which than then there these those through under very must series""".split())
tokens = df["description"].str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.split()
doc_sets = tokens.apply(lambda ws: {w for w in ws if w not in STOP and len(w) > 2})
word_doc = pd.Series([w for s in doc_sets for w in s]).value_counts()
dump(word_doc.head(50).rename("등장 문서 수"), "x2_word_docfreq_top50.csv")
say(f"설명문 길이 중앙값 {df['description'].str.len().median():.0f}자, 상위: " +
    ", ".join(f"{k}({v})" for k, v in word_doc.head(8).items()))


def genre_words(genre, k=12):
    ids = set(rel["genre"].loc[rel["genre"]["value"].eq(genre), "show_id"])
    m = df["show_id"].isin(ids)
    inn = pd.Series([w for s in doc_sets[m] for w in s]).value_counts()
    inn = inn[inn >= 25]
    lift = ((inn / m.sum()) / (word_doc.reindex(inn.index) / len(df))).sort_values(ascending=False).head(k)
    return pd.DataFrame({"장르 내 문서 수": inn[lift.index], "배수": lift.round(2)}), " ".join(df.loc[m, "description"]), int(m.sum())


dra, dra_txt, n_dra = genre_words("Dramas")
com, com_txt, n_com = genre_words("Comedies")
dump(dra, "x2_lift_dramas.csv")
dump(com, "x2_lift_comedies.csv")
say(f"Dramas {n_dra}: " + ", ".join(f"{i} {r['배수']}" for i, r in dra.head(8).iterrows()))
say(f"Comedies {n_com}: " + ", ".join(f"{i} {r['배수']}" for i, r in com.head(8).iterrows()))

wc_kw = dict(stopwords=STOP, background_color="white", random_state=SEED)
wc_all = WordCloud(max_words=100, width=900, height=450, colormap="viridis", **wc_kw).generate(" ".join(df["description"]))
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [3, 2]})
axes[0].imshow(wc_all, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title(f"(a) 전체 설명문 워드 클라우드 ({N:,}편)")
note(axes[0], "x축·y축 없음: 글자 크기 = 등장 빈도, 글자의 위치와 색에는 의미가 없음")
hbar(axes[1], word_doc.head(12), GREY, "", "")
lab(axes[1], "(b) 등장 문서 수 상위 12개 단어", "그 단어가 나온 설명문 수(편)", "단어")
save(fig, "x2_wordcloud_all.png", "추가 분석 2-1. 전체 설명문에서 자주 나온 단어 (교안 6-3)")

wc_d = WordCloud(max_words=80, width=700, height=400, colormap="GnBu_r", **wc_kw).generate(dra_txt)
wc_c = WordCloud(max_words=80, width=700, height=400, colormap="OrRd_r", **wc_kw).generate(com_txt)
fig, axes = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [2, 3]})
for row, (wc, t, name, c, n) in enumerate([(wc_d, dra, "Dramas", GREEN, n_dra), (wc_c, com, "Comedies", "#e76f51", n_com)]):
    axes[row][0].imshow(wc, interpolation="bilinear")
    axes[row][0].axis("off")
    axes[row][0].set_title(f"({'a' if row == 0 else 'c'}) {name} 워드 클라우드 ({n:,}편)")
    note(axes[row][0], "x축·y축 없음: 글자 크기 = 등장 빈도")
    ax = axes[row][1]
    tt2 = t[::-1]
    ax.barh(range(len(tt2)), tt2["배수"], color=c)
    ax.set_yticks(range(len(tt2)), tt2.index, fontsize=9)
    for i, (v, nn) in enumerate(zip(tt2["배수"], tt2["장르 내 문서 수"])):
        ax.text(v + .03, i, f"{v}  (n={nn})", va="center", fontsize=7.5)
    ax.axvline(1, color="#888", ls="--", lw=1)
    ax.set_xlim(0, 4.2)
    lab(ax, f"({'b' if row == 0 else 'd'}) {name} 특징어 — 전체 대비 등장 배수 (점선 = 1배)", "등장 배수 (장르 안 비율 ÷ 전체 비율)", "단어")
save(fig, "x2_genre_words.png", "추가 분석 2-2. 장르별 설명문 특징어: Dramas vs Comedies (교안 6-3)")

df.to_csv(OUT / "netflix_week6.csv", index=False, encoding="utf-8-sig")
(OUT / "_report_week6.txt").write_text("\n".join(log), encoding="utf-8")
say(f"\n차트 {len(list(OUT.glob('*.png')))}장, 집계표 {len(list(TAB.glob('*.csv')))}개")
