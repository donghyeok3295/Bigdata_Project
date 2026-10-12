"""6주차: 전처리(A 3.6) → 기초 EDA·시각화·심화 분석(B 4~7장) → 분석 카드(A 3.11.2).

입력은 5주차 정제본이고 원본 CSV는 변경하지 않습니다.
차트는 output/week6/, 집계표는 output/week6/tables/에 저장됩니다.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from wordcloud import STOPWORDS, WordCloud

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "netflix_titles.csv"
CLEANED = BASE / "output" / "netflix_cleaned.csv"
OUT = BASE / "output" / "week6"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

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


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def dump(obj, name):
    obj.to_csv(TAB / name, encoding="utf-8-sig")


raw = pd.read_csv(RAW)
df = pd.read_csv(CLEANED)
assert len(raw) == len(df) == 7787 and raw["show_id"].equals(df["show_id"])
N = len(df)
say(f"원본 {len(raw):,}행 × {raw.shape[1]}열, 5주차 정제본 {N:,}행 × {df.shape[1]}열입니다.")
say("5주차에 행을 삭제하지 않았으므로 두 파일의 행 수는 같습니다.")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("1. 전처리 (교재 3.6)")
say("=" * 70)

# rating 대치 플래그 — 5주차 누락분 보완
df["rating_imputed"] = raw["rating"].isna().astype(int)
say(f"\n5주차에 country에만 있던 추론 표시 플래그를 rating에도 추가했습니다. "
    f"country_imputed {int(df['country_imputed'].sum())}건, rating_imputed {int(df['rating_imputed'].sum())}건입니다.")

# ── 1-1 날짜 파생 (3.6.1) ────────────────────────────────────────
FMT = "%B %d, %Y"
naive = pd.to_datetime(raw["date_added"], format=FMT, errors="coerce")
fixed = pd.to_datetime(df["date_added"], format=FMT, errors="coerce")
df["date_added_parsed"] = fixed
df["added_year"] = fixed.dt.year
df["added_month"] = fixed.dt.month
say(f"\n[1-1] 날짜 파생")
say(f"  원본을 그대로 파싱하면 {naive.notna().sum():,}건이 성공합니다. "
    f"5주차에 앞뒤 공백을 지운 값으로 파싱하면 {fixed.notna().sum():,}건이 성공합니다. 차이는 {fixed.notna().sum() - naive.notna().sum()}건입니다.")
say(f"  실패했던 값은 ' August 4, 2017'처럼 앞에 공백이 붙어 있었습니다.")
say(f"  남은 파싱 실패 0건, 원문이 비어 있는 행 {int(df['date_added'].isna().sum())}개입니다. 이 10행은 연도별 비교에서 제외됩니다.")
say(f"  파싱 결과의 범위는 {fixed.min():%Y-%m-%d}부터 {fixed.max():%Y-%m-%d}까지입니다.")

# ── 1-2 duration 분리 (3.6.2) ────────────────────────────────────
parts = df["duration"].str.extract(r"^(\d+)\s+(min|Season|Seasons)$")
df["duration_value"] = pd.to_numeric(parts[0], errors="coerce")
df["duration_unit"] = parts[1]
mv = df["type"].eq("Movie")
mixed_mean = pd.to_numeric(df["duration"].str.extract(r"^(\d+)")[0]).mean()
say(f"\n[1-2] duration 값과 단위 분리")
say(f"  Movie {int(mv.sum()):,}편은 모두 'min', TV Show {int((~mv).sum()):,}편은 모두 'Season' 또는 'Seasons'입니다. 형식이 다른 행은 없습니다.")
say(f"  단위를 무시하고 숫자만 평균내면 {mixed_mean:.1f}이 나옵니다. "
    f"영화 {df.loc[mv, 'duration_value'].mean():.1f}분과 TV {df.loc[~mv, 'duration_value'].mean():.1f}시즌을 섞은 값이라 단위를 붙일 수 없습니다.")
say(f"  이후 길이 분석은 Movie와 TV Show를 따로 집계합니다.")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
ax = axes[0]
ax.bar(["공백 정리 전", "공백 정리 후"], [naive.notna().sum(), fixed.notna().sum()], color=[GREY, BLUE], width=.5)
for i, v in enumerate([naive.notna().sum(), fixed.notna().sum()]):
    ax.text(i, v + 4, f"{v:,}", ha="center", fontsize=10)
ax.set_ylim(7600, 7810)
ax.set(title="[그림 1] date_added 파싱 성공 건수", ylabel="건수")
ax = axes[1]
vals = [mixed_mean, df.loc[mv, "duration_value"].mean(), df.loc[~mv, "duration_value"].mean()]
ax.bar(["단위 무시\n숫자만 평균", "분리 후\nMovie(분)", "분리 후\nTV(시즌)"], vals, color=[RED, BLUE, ORANGE], width=.55)
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=10)
ax.set(title="[그림 2] 단위를 섞었을 때 나오는 값", ylabel="평균(단위 혼재)")
ax = axes[2]
sea = df.loc[~mv, "duration_value"].value_counts().sort_index()
ax.bar(sea.index, sea.values, color=ORANGE)
ax.set(title=f"[그림 3] 분리 후 TV Show 시즌 수 (N={int((~mv).sum()):,})", xlabel="시즌 수", ylabel="작품 수", xlim=(0, 17))
save(fig, "p1_date_duration.png")

# ── 1-3 다중값 열 장형 정규화 (3.6.3) ────────────────────────────
src_country = df["country"].where(df["country_imputed"].eq(0))


def to_long(series):
    s = series.dropna()
    return (pd.DataFrame({"show_id": df.loc[s.index, "show_id"], "value": s.str.split(",")})
            .explode("value").assign(value=lambda x: x["value"].str.strip())
            .loc[lambda x: x["value"].ne("")].drop_duplicates())


rel = {"director": to_long(df["director"]), "cast": to_long(df["cast"]),
       "country": to_long(src_country), "genre": to_long(df["listed_in"])}
BOOK = {"director": (6114, 5398), "cast": (55953, 7069), "country": (9063, 7280), "genre": (17071, 7787)}
rows = []
for k, t in rel.items():
    t.to_csv(TAB / f"rel_{k}.csv", index=False, encoding="utf-8-sig")
    rows.append({"관계": k, "관계 수": len(t), "작품 수": t["show_id"].nunique(), "고유값": t["value"].nunique(),
                 "교재 명시값": f"{BOOK[k][0]:,} / {BOOK[k][1]:,}",
                 "일치": "O" if (len(t), t["show_id"].nunique()) == BOOK[k] else "X"})
p3 = pd.DataFrame(rows)
dump(p3.set_index("관계"), "p3_long_tables.csv")
say(f"\n[1-3] 다중값 열을 장형으로 펼치기")
say(p3.to_string(index=False))
say("  한 작품이 여러 값에 속하므로 관계 수가 작품 수보다 큽니다. 오류가 아닙니다.")
say("  국가는 5주차에 감독으로 추론한 68건을 빼고 원문만 펼쳤습니다.")

genre_raw = df["listed_in"].value_counts().head(10)
genre_long = rel["genre"]["value"].value_counts().head(10)
first_country = src_country.dropna().str.split(",").str[0].str.strip().value_counts()
all_country = rel["country"]["value"].value_counts()
cmp_c = pd.DataFrame({"첫 국가만": first_country, "전체 펼침": all_country}).fillna(0).astype(int)
cmp_c["차이"] = cmp_c["전체 펼침"] - cmp_c["첫 국가만"]
cmp_c["증가율(%)"] = (cmp_c["차이"] / cmp_c["첫 국가만"] * 100).round(1)
top_c = cmp_c.sort_values("전체 펼침", ascending=False).head(12)
dump(top_c, "p3_country_first_vs_long.csv")
say(f"\n  장르를 셀 문자열 그대로 세면 1위가 '{genre_raw.index[0]}' {genre_raw.iloc[0]}건입니다. "
    f"3위는 '{genre_raw.index[2]}'처럼 조합입니다. 고유값이 {df['listed_in'].nunique()}개로 흩어집니다.")
say(f"  펼친 뒤에는 1위가 '{genre_long.index[0]}' {genre_long.iloc[0]:,}건이고 장르는 {rel['genre']['value'].nunique()}개입니다.")
say(f"\n  국가를 첫 번째 값만 쓰는 방식과 전부 펼친 방식의 차이입니다. 상위 12개국입니다.")
say(top_c.to_string())
multi = rel["country"].groupby("show_id").size()
say(f"  2개국 이상 공동 제작은 {int((multi > 1).sum()):,}편으로 전체의 {(multi > 1).mean():.1%}입니다. 최다는 {int(multi.max())}개국입니다.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axes[0]
yy = np.arange(10)
ax.barh(yy, genre_raw.values, color=GREY)
ax.set_yticks(yy, [t if len(t) < 42 else t[:40] + "…" for t in genre_raw.index], fontsize=8)
ax.invert_yaxis()
ax.set(title="[그림 4] 펼치기 전 — listed_in 문자열을 그대로 집계", xlabel="건수")
ax = axes[1]
ax.barh(yy, genre_long.values, color=BLUE)
ax.set_yticks(yy, genre_long.index, fontsize=9)
ax.invert_yaxis()
for i, v in enumerate(genre_long.values):
    ax.text(v + 20, i, f"{v:,}", va="center", fontsize=8)
ax.set(title="[그림 5] 펼친 뒤 — 콘텐츠–장르 관계로 집계", xlabel="관계 수")
save(fig, "p2_genre_explode.png")

fig, ax = plt.subplots(figsize=(11, 4.6))
t = top_c.head(10)
yy = np.arange(len(t))
ax.barh(yy + .2, t["첫 국가만"], .38, color=GREY, label="첫 번째 국가만 사용")
ax.barh(yy - .2, t["전체 펼침"], .38, color=BLUE, label="전체 국가 펼침")
for i, (a, b, r) in enumerate(zip(t["첫 국가만"], t["전체 펼침"], t["증가율(%)"])):
    ax.text(b + 20, i - .2, f"+{b - a} ({r:.0f}%)", va="center", fontsize=8, color=BLUE)
ax.set_yticks(yy, t.index, fontsize=9)
ax.invert_yaxis()
ax.legend(fontsize=9, loc="lower right")
ax.set(title="[그림 6] 국가 집계 방식에 따른 차이", xlabel="콘텐츠 수", xlim=(0, 3900))
save(fig, "p3_country_method.png")

# ── 1-4 등급 체계 분리 ───────────────────────────────────────────
df["rating_system"] = np.where(df["rating"].isin(MPAA), "MPAA(영화 등급)",
                               np.where(df["rating"].isin(TVR), "TV 등급", "기타"))
sc = df["rating_system"].value_counts()
say(f"\n[1-4] rating 열에 두 체계가 섞여 있습니다.")
say(f"  MPAA 영화 등급 {sc.get('MPAA(영화 등급)', 0):,}편, TV 등급 {sc.get('TV 등급', 0):,}편입니다. 그 밖의 값은 {sc.get('기타', 0)}개입니다.")
say("  부여 주체와 기준이 다르므로 한 축에 연령 순으로 늘어놓지 않습니다. 이후 등급 차트는 두 체계를 선으로 나눠 그립니다.")

# ── 1-5 텍스트 전처리 규칙 (3.6.4) ───────────────────────────────
desc_len = df["description"].str.len()
say(f"\n[1-5] 설명문 전처리 규칙")
say(f"  description은 결측 0건입니다. 길이는 중앙값 {desc_len.median():.0f}자, 범위 {desc_len.min()}~{desc_len.max()}자입니다.")
say(f"  완전히 같은 설명문이 {int(df['description'].duplicated(keep=False).sum())}건 있습니다.")
say("  적용 규칙은 소문자 변환, 영문 외 문자 제거, 불용어 제거, 3자 이상만 남기기입니다.")

STOP = set(STOPWORDS) | set("""his her their its who when after into while about over two one new
them they where what which than then there these those through under very must series""".split())
tokens = df["description"].str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.split()
doc_sets = tokens.apply(lambda ws: {w for w in ws if w not in STOP and len(w) > 2})
word_doc = pd.Series([w for s in doc_sets for w in s]).value_counts()
dump(word_doc.head(50).rename("등장 문서 수"), "t_word_docfreq_top50.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 3]})
ax = axes[0]
ax.hist(desc_len, bins=30, color=PURPLE)
ax.axvline(desc_len.median(), color=RED, ls="--", label=f"중앙값 {desc_len.median():.0f}자")
ax.legend()
ax.set(title="[그림 7] 설명문 길이 분포", xlabel="글자 수", ylabel="작품 수")
ax = axes[1]
t = word_doc.head(14)[::-1]
ax.barh(range(len(t)), t.values, color=GREY)
ax.set_yticks(range(len(t)), t.index, fontsize=9)
for i, v in enumerate(t.values):
    ax.text(v + 6, i, f"{v}", va="center", fontsize=8)
ax.set(title="[그림 8] 불용어 제거 후 상위 단어 (등장 문서 수)", xlabel="문서 수")
save(fig, "p4_text.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("2. 기초 EDA (교안 4장)")
say("=" * 70)

tt = df["type"].value_counts()
say(f"\n[2-1] 콘텐츠 유형")
say(f"  Movie {tt['Movie']:,}편({tt['Movie'] / N:.1%}), TV Show {tt['TV Show']:,}편({tt['TV Show'] / N:.1%})입니다. 분모는 전체 {N:,}편입니다.")

rt = pd.crosstab(df["rating"], df["type"])
rt = rt.loc[[r for r in MPAA + TVR if r in rt.index]]
dump(rt, "e1_rating_by_type.csv")
say(f"\n[2-2] 시청 등급")
say(f"  등급이 비어 있던 7건은 5주차에 채운 값이므로 이 표에는 원문 그대로 포함했습니다. 분모는 {N:,}편입니다.")
say(f"  가장 많은 등급은 TV-MA {int(rt.loc['TV-MA'].sum()):,}편, 다음은 TV-14 {int(rt.loc['TV-14'].sum()):,}편입니다.")
say(f"  R·PG-13·PG·G는 사실상 Movie에만 붙습니다. TV Show의 R 등급은 {int(rt.loc['R', 'TV Show'])}편뿐입니다.")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 3]})
ax = axes[0]
ax.bar(tt.index, tt.values, color=[BLUE, ORANGE], width=.55)
for i, v in enumerate(tt.values):
    ax.text(i, v + 80, f"{v:,}\n({v / N:.1%})", ha="center", fontsize=10)
ax.set(title="[그림 9] 콘텐츠 유형별 작품 수", ylabel="작품 수", ylim=(0, 6300))
ax = axes[1]
xx = np.arange(len(rt))
ax.bar(xx, rt["Movie"], .62, color=BLUE, label="Movie")
ax.bar(xx, rt["TV Show"], .62, bottom=rt["Movie"], color=ORANGE, label="TV Show")
for i, v in enumerate(rt.sum(axis=1)):
    ax.text(i, v + 25, f"{v:,}", ha="center", fontsize=8)
ax.axvline(len([r for r in MPAA if r in rt.index]) - .5, color="#555", lw=1.2)
ax.text(2.5, 2700, "MPAA 영화 등급", ha="center", fontsize=10, color="#555")
ax.text(9.5, 2700, "TV 등급", ha="center", fontsize=10, color="#555")
ax.set_xticks(xx, rt.index, fontsize=9)
ax.legend()
ax.set(title="[그림 10] 시청 등급별 작품 수 (두 체계를 선으로 구분)", ylabel="작품 수")
save(fig, "e1_type_rating.png")

# Top 10 네 가지
say(f"\n[2-3] 장르·국가·배우·감독 상위 10")
tops = {}
for key, label, uniq in [("genre", "장르", rel["genre"]["show_id"].nunique()),
                         ("country", "국가", rel["country"]["show_id"].nunique()),
                         ("cast", "배우", rel["cast"]["show_id"].nunique()),
                         ("director", "감독", rel["director"]["show_id"].nunique())]:
    s = rel[key]["value"].value_counts()
    tops[key] = (label, s, uniq)
    dump(s.head(20).rename("작품 수"), f"e2_top20_{key}.csv")
    say(f"  {label}: 값이 있는 작품 {uniq:,}편 기준입니다. 상위 3은 " +
        ", ".join(f"{k} {v:,}" for k, v in s.head(3).items()) + "입니다.")
say("  값이 여러 개인 작품은 각 값에 한 번씩 포함됩니다. 그래서 합계가 작품 수보다 큽니다.")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, key, c in zip(axes.ravel(), ["genre", "country", "cast", "director"], [BLUE, GREEN, ORANGE, PURPLE]):
    label, s, uniq = tops[key]
    t = s.head(10)[::-1]
    ax.barh(range(len(t)), t.values, color=c)
    ax.set_yticks(range(len(t)), t.index, fontsize=9)
    for i, v in enumerate(t.values):
        ax.text(v + max(t.values) * .012, i, f"{v:,}", va="center", fontsize=8)
    ax.set(title=f"{label} 상위 10 (값이 있는 작품 {uniq:,}편 기준)", xlabel="작품 수")
    ax.set_xlim(0, max(t.values) * 1.2)
fig.suptitle("[그림 11] 장르·국가·배우·감독 상위 10 — 여러 값을 가진 작품은 각 값에 포함됩니다", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .96])
fig.savefig(OUT / "e2_top10.png")
plt.close(fig)

# 길이·시즌·제작연도
mdur = df.loc[mv, "duration_value"]
tsea = df.loc[~mv, "duration_value"]
stat = pd.DataFrame({
    "Movie(분)": [len(mdur), mdur.mean(), mdur.median(), mdur.quantile(.25), mdur.quantile(.75), mdur.min(), mdur.max()],
    "TV Show(시즌)": [len(tsea), tsea.mean(), tsea.median(), tsea.quantile(.25), tsea.quantile(.75), tsea.min(), tsea.max()],
}, index=["유효 행", "평균", "중앙값", "25%", "75%", "최소", "최대"]).round(2)
dump(stat, "e3_duration_stats.csv")
say(f"\n[2-4] 길이와 시즌 수")
say(stat.to_string())
say("  단위가 다르므로 두 열의 평균을 직접 비교하지 않습니다.")
say(f"  영화 최단은 {df.loc[mv].nsmallest(1, 'duration_value')['title'].iloc[0]} {mdur.min():.0f}분, "
    f"최장은 {df.loc[mv].nlargest(1, 'duration_value')['title'].iloc[0]} {mdur.max():.0f}분입니다. 값을 수정하지 않았습니다.")

ry = df["release_year"]
say(f"\n[2-5] 제작 연도")
say(f"  범위는 {ry.min()}년부터 {ry.max()}년까지입니다. 중앙값은 {ry.median():.0f}년이고 2015년 이후 제작이 {(ry >= 2015).mean():.1%}입니다.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.hist(mdur, bins=np.arange(0, 320, 10), color=BLUE)
ax.axvline(mdur.median(), color=RED, ls="--", label=f"중앙값 {mdur.median():.0f}분")
ax.legend()
ax.set(title=f"[그림 12] Movie 상영시간 (N={len(mdur):,})", xlabel="분", ylabel="작품 수")
ax = axes[1]
vc = tsea.value_counts().sort_index()
ax.bar(vc.index, vc.values, color=ORANGE)
ax.set(title=f"[그림 13] TV Show 시즌 수 (N={len(tsea):,})", xlabel="시즌", ylabel="작품 수", xlim=(0, 17))
ax = axes[2]
ax.hist(ry, bins=50, color=PURPLE)
ax.set(title=f"[그림 14] 제작 연도 분포 (N={N:,})", xlabel="release_year", ylabel="작품 수")
save(fig, "e3_duration_release.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("3. 시각화 (교안 5장)")
say("=" * 70)

cutoff = df["date_added_parsed"].max()
annual = (df.dropna(subset=["added_year"]).groupby(["added_year", "type"]).size()
          .unstack(fill_value=0).astype(int))
annual["합계"] = annual.sum(axis=1)
annual.index = annual.index.astype(int)
dump(annual, "v1_annual_added.csv")
say(f"\n[3-1] 연도별 추가량")
say(f"  분모는 date_added가 있는 {int(annual['합계'].sum()):,}편입니다. 비어 있는 10행은 제외됩니다.")
say(annual.to_string())
say(f"  마지막 등록일은 {cutoff:%Y-%m-%d}입니다. 2021년은 1월 16일까지만 집계된 부분 연도이므로 다른 연도와 총량을 비교하지 않습니다.")

ym = pd.crosstab(df["added_year"].dropna().astype(int), df["added_month"].dropna().astype(int))
dump(ym, "v2_year_month.csv")
mcnt = df["added_month"].value_counts().sort_index()
say(f"\n[3-2] 월별 추가량")
say(f"  가장 많은 달은 {int(mcnt.idxmax())}월 {int(mcnt.max()):,}편, 가장 적은 달은 {int(mcnt.idxmin())}월 {int(mcnt.min()):,}편입니다.")
say(f"  하반기(7~12월) 비중은 {mcnt.loc[7:].sum() / mcnt.sum():.1%}입니다. 2021년은 1월만 있어 월별 비교를 왜곡할 수 있습니다.")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
full = annual.loc[annual.index < cutoff.year]
for col, c in [("Movie", BLUE), ("TV Show", ORANGE)]:
    ax.plot(full.index, full[col], marker="o", color=c, label=col, lw=2)
    ax.plot([full.index[-1], cutoff.year], [full[col].iloc[-1], annual.loc[cutoff.year, col]],
            ls=":", color=c, marker="x", lw=1.5)
ax.axvspan(cutoff.year - .5, cutoff.year + .5, color="#e0e0e0")
ax.text(cutoff.year, 1250, f"{cutoff.year}\n부분 연도\n(~{cutoff:%m/%d})", ha="center", fontsize=8)
ax.legend()
ax.set(title=f"[그림 15] 연도별 추가량 (N={int(annual['합계'].sum()):,})", xlabel="추가 연도", ylabel="작품 수")
ax = axes[1]
tot = annual["합계"]
ax.bar(tot.index, tot.values, color=[GREY if y == cutoff.year else PURPLE for y in tot.index])
ax.set_yscale("log")
for y, v in tot.items():
    ax.text(y, v * 1.15, f"{v:,}", ha="center", fontsize=7.5, rotation=90)
ax.set(title="[그림 16] 같은 값을 로그축으로", xlabel="추가 연도", ylabel="작품 수(로그)", ylim=(.7, 6500))
save(fig, "v1_added_year.png")

fig, ax = plt.subplots(figsize=(11, 4.6))
sns.heatmap(ym, cmap="YlGnBu", annot=True, fmt="d", linewidths=.5, ax=ax, cbar_kws={"label": "작품 수"})
ax.set(title="[그림 17] 연도 × 월별 추가량 (date_added 있는 7,777편)", xlabel="월", ylabel="추가 연도")
save(fig, "v2_year_month.png")

# 산점도 + 박스플롯
lag = df["added_year"] - df["release_year"]
lag_stat = df.assign(lag=lag).dropna(subset=["lag"]).groupby("type")["lag"].agg(
    유효행="count", 중앙값="median", 평균="mean").round(2)
dump(lag_stat, "v3_lag.csv")
say(f"\n[3-3] 제작 연도와 추가 연도의 관계")
say(lag_stat.to_string())
say(f"  추가 연도에서 제작 연도를 뺀 값입니다. 음수인 {int((lag < 0).sum())}건은 삭제하지 않고 품질 검토 대상으로 남겼습니다.")
say("  두 연도의 차이는 제작 시점과 목록 추가 시점의 간격이며 실제 제작 완료일이나 서비스 시작일을 뜻하지 않습니다.")

mr = df.loc[mv & df["duration_unit"].eq("min") & df["rating_imputed"].eq(0),
            ["show_id", "rating", "duration_value", "rating_system"]].dropna(subset=["rating"])
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
ax = axes[0]
sub = df.dropna(subset=["added_year"])
for t, c in [("Movie", BLUE), ("TV Show", ORANGE)]:
    s = sub[sub.type == t]
    ax.scatter(s["release_year"], s["added_year"], s=8, alpha=.25, color=c, label=t)
ax.plot([1940, 2021], [1940, 2021], color=RED, ls="--", lw=1, label="제작=추가")
ax.legend(fontsize=9)
ax.set(title=f"[그림 18] 제작 연도와 추가 연도 (N={len(sub):,})", xlabel="release_year", ylabel="added_year", xlim=(1940, 2023))
ax = axes[1]
top5 = df["rating"].value_counts().head(5).index.tolist()
order5 = [r for r in MPAA + TVR if r in top5]
sns.boxplot(data=mr[mr.rating.isin(order5)], x="rating", y="duration_value", order=order5,
            hue="rating_system", palette={"MPAA(영화 등급)": "#8ecae6", "TV 등급": ORANGE},
            dodge=False, fliersize=2, ax=ax)
for i, r in enumerate(order5):
    ax.text(i, -14, f"n={int((mr.rating == r).sum()):,}", ha="center", fontsize=8, color="#555")
ax.set_ylim(-26, 330)
ax.legend(title="", fontsize=8, loc="upper right")
ax.set(title="[그림 19] 상위 5개 등급의 영화 상영시간", xlabel="시청 등급", ylabel="분")
save(fig, "v3_scatter_box.png")

# 연습문제 2 — 가상 투자액
inv = pd.DataFrame({"year": [2015, 2016, 2017, 2018, 2019, 2020],
                    "investment": [4.5, 6.0, 8.9, 12.0, 15.0, 17.0]})
inv["added"] = inv["year"].map(annual["합계"])
r_pear = inv["added"].corr(inv["investment"])
r_spear = inv["added"].corr(inv["investment"], method="spearman")
dump(inv.set_index("year"), "v4_investment_virtual.csv")
say(f"\n[3-4] 추가량과 투자액 (교안 연습문제 2)")
say(f"  투자액은 교안이 제시한 가상 데이터입니다. 실제 공시 자료가 아닙니다.")
say(f"  2015~2020년 6개 연도에서 피어슨 상관 {r_pear:.3f}, 스피어만 상관 {r_spear:.3f}입니다.")
say("  가상 값으로 계산한 수치이므로 이 상관계수를 분석 결과로 해석하지 않습니다. 이중 축 차트를 그리는 방법을 확인하는 용도입니다.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
ax.bar(inv["year"], inv["added"], color=BLUE, width=.55, label="추가량(실제)")
ax.set(xlabel="연도", ylabel="추가된 작품 수")
ax2 = ax.twinx()
ax2.plot(inv["year"], inv["investment"], marker="o", color=RED, lw=2, label="투자액(가상)")
ax2.set_ylabel("투자액(십억 달러, 가상)")
ax.set_title("[그림 20] 추가량(실제) × 투자액(가상) 이중 축 — 가상 값이 섞여 있습니다")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
ax = axes[1]
ax.scatter(inv["investment"], inv["added"], s=70, color=PURPLE)
for _, r in inv.iterrows():
    ax.annotate(int(r["year"]), (r["investment"], r["added"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set(title=f"[그림 21] 산점도 (피어슨 {r_pear:.3f})", xlabel="투자액(가상)", ylabel="추가량(실제)")
save(fig, "v4_investment.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("4. 관계 분석 (교안 6-1)")
say("=" * 70)

gr = rel["genre"].merge(df[["show_id", "rating", "rating_imputed"]], on="show_id")
gr = gr[gr["rating_imputed"].eq(0)].dropna(subset=["rating"])
top10g = gr["value"].value_counts().head(10).index
ct = pd.crosstab(gr.loc[gr.value.isin(top10g), "value"], gr.loc[gr.value.isin(top10g), "rating"])
ct = ct[[c for c in MPAA + TVR if c in ct.columns]]
row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
dump(ct, "r1_genre_rating_count.csv")
dump(row_pct.round(1), "r1_genre_rating_rowpct.csv")
say(f"\n[4-1] 장르와 시청 등급")
say(f"  콘텐츠–장르 관계 {len(gr):,}건 중 상위 10개 장르만 그렸습니다. 5주차에 채운 rating 7건은 제외했습니다.")
say("  행 비율의 분모는 그 장르의 관계 수입니다. 한 작품이 여러 장르에 속하면 각 장르에 포함됩니다.")
say(row_pct.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
sns.heatmap(ct, cmap="Blues", annot=True, fmt="d", linewidths=.5, ax=axes[0], cbar_kws={"label": "관계 수"})
axes[0].set(title="[그림 22] 장르 × 등급 — 관계 수", xlabel="시청 등급", ylabel="장르")
sns.heatmap(row_pct, cmap="Purples", annot=True, fmt=".0f", linewidths=.5, ax=axes[1], cbar_kws={"label": "행 비율(%)"})
axes[1].set(title="[그림 23] 같은 표를 행 비율(%)로 — 분모는 그 장르의 관계 수", xlabel="시청 등급", ylabel="")
save(fig, "r1_genre_rating.png")

cg = rel["country"].merge(rel["genre"], on="show_id", suffixes=("_c", "_g"))
top10c = rel["country"]["value"].value_counts().head(10).index
cgt = pd.crosstab(cg.loc[cg.value_c.isin(top10c) & cg.value_g.isin(top10g), "value_c"],
                  cg.loc[cg.value_c.isin(top10c) & cg.value_g.isin(top10g), "value_g"])
cg_pct = cgt.div(cgt.sum(axis=1), axis=0) * 100
dump(cgt, "r2_country_genre_count.csv")
dump(cg_pct.round(1), "r2_country_genre_rowpct.csv")
say(f"\n[4-2] 국가와 장르")
say("  상위 10개국과 상위 10개 장르의 교차표입니다. 공동 제작이나 복수 장르 작품은 각 칸에 한 번씩 포함됩니다.")
say("  행 비율의 분모는 그 국가가 상위 10개 장르와 맺은 관계 수입니다.")
say(cg_pct.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
sns.heatmap(cgt, cmap="rocket_r", annot=True, fmt="d", linewidths=.5, ax=axes[0], cbar_kws={"label": "관계 수"})
axes[0].set(title="[그림 24] 국가 × 장르 — 관계 수", xlabel="장르", ylabel="국가")
sns.heatmap(cg_pct, cmap="Greens", annot=True, fmt=".0f", linewidths=.5, ax=axes[1], cbar_kws={"label": "행 비율(%)"})
axes[1].set(title="[그림 25] 같은 표를 행 비율(%)로 — 나라 크기 차이를 지움", xlabel="장르", ylabel="")
save(fig, "r2_country_genre.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("5. 네트워크 분석 (교안 6-2)")
say("=" * 70)

dc = rel["director"].merge(rel["cast"], on="show_id", suffixes=("_d", "_a"))
top30 = rel["director"]["value"].value_counts().head(30).index
edges = dc[dc.value_d.isin(top30)][["value_d", "value_a"]].drop_duplicates()
G = nx.Graph()
G.add_edges_from(edges.itertuples(index=False, name=None))
deg = pd.Series(nx.degree_centrality(G)).sort_values(ascending=False)
top15 = deg.head(15)
dump(top15.rename("연결 중심성"), "n1_degree_centrality_top15.csv")
say(f"\n[5-1] 감독과 배우의 협업 네트워크")
say(f"  작품 수 상위 30명의 감독과 그 작품에 출연한 배우만 대상으로 했습니다.")
say(f"  노드 {G.number_of_nodes():,}개, 간선 {G.number_of_edges():,}개입니다. 간선 하나는 감독과 배우가 같은 작품에 참여했다는 뜻입니다.")
say(f"  감독이나 배우가 비어 있는 작품은 제외됩니다.")
say(f"  연결 중심성 상위 15명 중 감독은 {int(top15.index.isin(top30).sum())}명입니다.")
say(top15.round(4).to_string())
say("  감독은 한 작품의 출연진 수만큼 연결이 생기고, 배우는 여러 감독과 일해야 연결이 늘어납니다.")
say("  그래서 이 순위는 영향력 순위가 아니라 '상위 30명 감독 중 가장 다양한 배우와 일한 감독' 순위로 읽습니다.")

pos = nx.kamada_kawai_layout(G)
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
is_dir = [n in set(top30) for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[120 if d else 8 for d in is_dir],
                       node_color=[RED if d else "#8ecae6" for d in is_dir])
nx.draw_networkx_edges(G, pos, ax=ax, alpha=.12, edge_color="grey")
for n in deg.head(8).index:
    x, yy_ = pos[n]
    ax.annotate(n, (x, yy_), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="#bbb", alpha=.9),
                arrowprops=dict(arrowstyle="-", color="#999", lw=.8))
ax.set_title(f"[그림 26] 감독–배우 협업 네트워크 (감독 30명, 노드 {G.number_of_nodes():,}개 / 간선 {G.number_of_edges():,}개)")
ax.axis("off")
ax = axes[1]
t = top15[::-1]
ax.barh(range(len(t)), t.values, color=[RED if n in set(top30) else "#8ecae6" for n in t.index])
ax.set_yticks(range(len(t)), t.index, fontsize=8)
for i, v in enumerate(t.values):
    ax.text(v + .0004, i, f"{v:.4f}", va="center", fontsize=7.5)
ax.set(title="[그림 27] 연결 중심성 상위 15\n(빨강=감독, 하늘=배우)", xlabel="연결 중심성")
save(fig, "n1_network.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("6. 텍스트 마이닝 (교안 6-3)")
say("=" * 70)

all_text = " ".join(df["description"])
wc_all = WordCloud(stopwords=STOP, background_color="white", max_words=100,
                   width=900, height=450, random_state=SEED, colormap="viridis").generate(all_text)
say(f"\n[6-1] 전체 설명문")
say(f"  전체 글자 수 {len(all_text):,}자입니다. 영문 3자 이상 단어를 세고 기본 불용어를 제외했습니다.")
say(f"  등장 문서 수 상위 6은 " + ", ".join(f"{k}({v})" for k, v in word_doc.head(6).items()) + "입니다.")
say(f"  워드 클라우드는 글자 크기로 빈도를 나타낸 그림입니다. 정확한 순위는 tables/t_word_docfreq_top50.csv를 기준으로 봅니다.")


def genre_words(genre, k=12):
    ids = set(rel["genre"].loc[rel["genre"]["value"].eq(genre), "show_id"])
    m = df["show_id"].isin(ids)
    inn = pd.Series([w for s in doc_sets[m] for w in s]).value_counts()
    inn = inn[inn >= 25]
    lift = ((inn / m.sum()) / (word_doc.reindex(inn.index) / len(df))).sort_values(ascending=False).head(k)
    txt = " ".join(df.loc[m, "description"])
    return pd.DataFrame({"장르 내 문서 수": inn[lift.index], "배수": lift.round(2)}), txt, int(m.sum())


dra, dra_txt, n_dra = genre_words("Dramas")
com, com_txt, n_com = genre_words("Comedies")
dump(dra, "t_lift_dramas.csv")
dump(com, "t_lift_comedies.csv")
say(f"\n[6-2] 장르별 설명문 비교")
say(f"  Dramas {n_dra:,}편, Comedies {n_com:,}편의 설명문을 각각 모았습니다.")
say(f"  단순 빈도로는 두 장르 모두 life·family·love가 상위라 구분되지 않습니다.")
say(f"  배수는 '그 장르에서 이 단어가 나오는 문서 비율'을 '전체에서 나오는 비율'로 나눈 값입니다. 1이면 전체와 같습니다.")
say(f"  노이즈를 줄이려고 장르 안에서 25개 문서 이상 등장한 단어만 남겼습니다.")
say(f"  Dramas: " + ", ".join(f"{i} {r['배수']}배" for i, r in dra.head(6).iterrows()))
say(f"  Comedies: " + ", ".join(f"{i} {r['배수']}배" for i, r in com.head(6).iterrows()))

wc_d = WordCloud(stopwords=STOP, background_color="white", max_words=80, width=700, height=400,
                 random_state=SEED, colormap="GnBu_r").generate(dra_txt)
wc_c = WordCloud(stopwords=STOP, background_color="white", max_words=80, width=700, height=400,
                 random_state=SEED, colormap="OrRd_r").generate(com_txt)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [3, 2]})
axes[0].imshow(wc_all, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title(f"[그림 28] 전체 설명문 워드 클라우드 ({N:,}편)")
t = word_doc.head(12)[::-1]
axes[1].barh(range(len(t)), t.values, color=GREY)
axes[1].set_yticks(range(len(t)), t.index, fontsize=9)
for i, v in enumerate(t.values):
    axes[1].text(v + 6, i, f"{v}", va="center", fontsize=8)
axes[1].set(title="[그림 29] 같은 내용을 등장 문서 수로", xlabel="문서 수")
save(fig, "t1_wordcloud_all.png")

fig, axes = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [2, 3]})
for row, (wc, t, name, c, n) in enumerate([(wc_d, dra, "Dramas", GREEN, n_dra),
                                           (wc_c, com, "Comedies", "#e76f51", n_com)]):
    axes[row][0].imshow(wc, interpolation="bilinear")
    axes[row][0].axis("off")
    axes[row][0].set_title(f"{name} 워드 클라우드 ({n:,}편)")
    ax = axes[row][1]
    tt2 = t[::-1]
    ax.barh(range(len(tt2)), tt2["배수"], color=c)
    ax.set_yticks(range(len(tt2)), tt2.index, fontsize=9)
    for i, (v, nn) in enumerate(zip(tt2["배수"], tt2["장르 내 문서 수"])):
        ax.text(v + .03, i, f"{v}  (n={nn})", va="center", fontsize=7.5)
    ax.axvline(1, color="#888", ls="--", lw=1)
    ax.set(title=f"{name} 특징어 — 전체 대비 등장 배수", xlabel="배수 (1 = 전체와 같음)")
fig.suptitle("[그림 30] 장르별 설명문 비교 — 워드 클라우드와 배수", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .96])
fig.savefig(OUT / "t2_genre_words.png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("7. 추천 (교안 6-4, 6-5)")
say("=" * 70)

soup = (df["title"].fillna("") + " " + df["listed_in"].fillna("") + " " + df["description"].fillna(""))
tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
M = tfidf.fit_transform(soup)
say(f"\n[7-1] 콘텐츠 기반 추천")
say(f"  제목, 장르, 설명을 합쳐 TF-IDF 행렬을 만들었습니다. 크기는 {M.shape[0]:,} × {M.shape[1]:,}입니다.")
say(f"  작품 사이의 코사인 유사도를 계산했습니다. 시청 기록이 아니라 텍스트가 얼마나 닮았는지를 재는 값입니다.")

seeds = ["Stranger Things", "The Irishman", "Our Planet", "Narcos"]
say(f"  기준 작품은 장르가 서로 다른 4편을 골랐습니다: " + ", ".join(seeds) + ".")
recs = []
for s in seeds:
    i = df.index[df["title"] == s][0]
    sim = cosine_similarity(M[i], M).ravel()
    sim[i] = -1
    for j in sim.argsort()[::-1][:3]:
        recs.append({"기준 작품": s, "유사 작품": df.at[j, "title"], "코사인 유사도": round(float(sim[j]), 3)})
rec_df = pd.DataFrame(recs)
dump(rec_df.set_index("기준 작품"), "rec1_content_based.csv")
say(rec_df.to_string(index=False))
say("  유사도는 텍스트가 닮은 정도입니다. 시청자의 실제 선호나 추천 만족도를 측정한 값이 아닙니다.")

rng = np.random.default_rng(SEED)
pop = df.sample(40, random_state=SEED)["title"].tolist()
fake = pd.DataFrame(rng.integers(1, 6, size=(60, 40)), columns=pop)
mask = rng.random((60, 40)) < .7
fake = fake.mask(mask)
item_sim = pd.DataFrame(cosine_similarity(fake.fillna(0).T.values), index=pop, columns=pop)
np.fill_diagonal(item_sim.values, 0)
cf = pd.DataFrame({"기준 작품": pop[:3],
                   "가장 유사한 작품": [item_sim[p].idxmax() for p in pop[:3]],
                   "유사도": [round(float(item_sim[p].max()), 3) for p in pop[:3]]})
dump(cf.set_index("기준 작품"), "rec2_collaborative_virtual.csv")
say(f"\n[7-2] 아이템 기반 협업 필터링")
say(f"  교안 6-5는 가상의 사용자 평점을 만들어 씁니다. 이 CSV에는 평점 열이 없습니다.")
say(f"  사용자 60명 × 작품 40개의 평점 행렬을 난수로 만들었습니다(seed={SEED}). 약 70%는 미평가로 두었습니다.")
say(cf.to_string(index=False))
say("  전부 난수로 만든 값입니다. 구현 절차를 확인하는 용도이며 결과 수치는 해석하지 않습니다.")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
ax = axes[0]
samp = cosine_similarity(M[:1500]).ravel()
samp = samp[samp < .999]
ax.hist(samp, bins=60, color=BLUE)
ax.set_yscale("log")
ax.set(title="[그림 31] 작품 간 코사인 유사도 분포 (앞 1,500편 표본)", xlabel="코사인 유사도", ylabel="쌍의 수(로그)")
ax = axes[1]
t = rec_df.iloc[::-1].reset_index(drop=True)
ax.barh(range(len(t)), t["코사인 유사도"],
        color=[GREEN if i % 3 == 2 else "#8ecae6" for i in range(len(t))])
ax.set_yticks(range(len(t)), [f"{r['유사 작품'][:30]}" for _, r in t.iterrows()], fontsize=8)
for i, v in enumerate(t["코사인 유사도"]):
    ax.text(v + .004, i, f"{v}", va="center", fontsize=7.5)
for i in range(0, len(t), 3):
    ax.axhline(i - .5, color="#bbb", lw=.8)
    ax.text(.002, i + 1, f"기준: {t.at[i + 2, '기준 작품']}", fontsize=8, color="#444",
            va="center", ha="left", bbox=dict(boxstyle="round,pad=.2", fc="white", ec="#ccc"))
ax.set_xlim(0, t["코사인 유사도"].max() * 1.22)
ax.set(title="[그림 32] 기준 작품 4편과 가장 닮은 작품 3편씩 (진한 막대 = 1위)", xlabel="코사인 유사도")
save(fig, "rec1_similarity.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("8. 회귀 모델 (교안 7장)")
say("=" * 70)

y = df["release_year"].astype(float)
Xtr, Xte, ytr, yte = train_test_split(soup, y, test_size=.2, random_state=SEED)
vec = TfidfVectorizer(stop_words="english", max_features=20000)
Xtr_v, Xte_v = vec.fit_transform(Xtr), vec.transform(Xte)
base_pred = np.full(len(yte), ytr.median())
base_mae = mean_absolute_error(yte, base_pred)
sweep = []
for a in [0.1, 1, 3, 10, 30, 100]:
    p = Ridge(alpha=a, random_state=SEED).fit(Xtr_v, ytr).predict(Xte_v)
    sweep.append({"alpha": a, "MAE(년)": round(mean_absolute_error(yte, p), 3), "R2": round(r2_score(yte, p), 3)})
sweep = pd.DataFrame(sweep).set_index("alpha")
best_a = sweep["MAE(년)"].idxmin()
model = Ridge(alpha=best_a, random_state=SEED).fit(Xtr_v, ytr)
pred = model.predict(Xte_v)
perf = pd.DataFrame({
    "MAE(년)": [base_mae, mean_absolute_error(yte, pred)],
    "R2": [r2_score(yte, base_pred), r2_score(yte, pred)],
}, index=["기준선(학습 중앙값)", f"TF-IDF + Ridge(alpha={best_a})"]).round(3)
dump(perf, "m1_regression.csv")
dump(sweep, "m2_alpha_sweep.csv")
say(f"\n[8-1] 설명 텍스트로 제작 연도 예측")
say(f"  입력은 제목·장르·설명을 합친 텍스트입니다. 학습 {len(ytr):,}편, 평가 {len(yte):,}편으로 나눴습니다(8:2, seed={SEED}).")
say(f"  기준선은 학습 자료의 제작 연도 중앙값 {ytr.median():.0f}년을 모든 작품에 그대로 답한 경우입니다.")
say(f"\n  규제 강도(alpha)를 6개 값으로 바꿔 본 결과입니다.")
say(sweep.to_string())
say(f"  MAE가 가장 작은 값은 alpha={best_a}의 {sweep.loc[best_a, 'MAE(년)']}년입니다. 기준선 {base_mae:.3f}년보다 여전히 큽니다.")
say(perf.to_string())
err = pred - yte.values
say(f"\n  두 지표가 반대 방향을 가리킵니다. R2는 {sweep.loc[best_a, 'R2']}로 0보다 크지만 MAE는 기준선보다 나쁩니다.")
say(f"  제작 연도가 최근에 쏠려 있어서 생기는 일입니다. 2015년 이후 제작이 {(y >= 2015).mean():.1%}이므로 "
    f"중앙값 하나로 답해도 대부분 몇 년 안에 들어맞습니다.")
say(f"  모델은 오래된 작품을 거의 맞히지 못합니다. 오차 중앙값은 {np.median(np.abs(err)):.2f}년이고 "
    f"5년 이내로 맞춘 비율은 {(np.abs(err) <= 5).mean():.1%}입니다.")
say(f"  1990년 이전 제작 {int((yte < 1990).sum())}편에서는 예측이 평균 {(pred[yte.values < 1990] - yte.values[yte.values < 1990]).mean():+.1f}년 치우칩니다.")
say("  이 값은 같은 CSV를 무작위로 나눈 평가 자료에 대한 결과입니다. 실제 서비스 환경의 성능을 뜻하지 않습니다.")
say("  설명문에 연도를 암시하는 단어가 들어 있을 수 있어 입력과 정답이 완전히 분리되지는 않습니다.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(sweep.index, sweep["MAE(년)"], marker="o", color=BLUE, lw=2, label="TF-IDF + Ridge")
ax.axhline(base_mae, color=RED, ls="--", lw=1.5, label=f"기준선 {base_mae:.2f}년")
ax.set_xscale("log")
ax.set_xticks(sweep.index.tolist())
ax.set_xticklabels([f"{a:g}" for a in sweep.index])
ax.minorticks_off()
for a, r in sweep.iterrows():
    ax.annotate(f"{r['MAE(년)']:.2f}", (a, r["MAE(년)"]), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7.5)
ax.legend(fontsize=9)
ax.set(title="[그림 33] 규제 강도별 MAE — 어떤 값에서도 기준선 미달", xlabel="alpha (로그축)", ylabel="MAE(년)")
ax = axes[1]
ax.scatter(yte, pred, s=6, alpha=.2, color=BLUE)
ax.plot([1940, 2021], [1940, 2021], color=RED, ls="--", lw=1)
ax.set(title="[그림 34] 실제 제작 연도와 예측값", xlabel="실제", ylabel="예측", xlim=(1940, 2023), ylim=(1940, 2023))
ax = axes[2]
ax.hist(err, bins=50, color=PURPLE)
ax.axvline(0, color=RED, ls="--", lw=1)
ax.set(title=f"[그림 35] 예측 오차 분포 (예측-실제, N={len(yte):,})", xlabel="오차(년)", ylabel="작품 수")
save(fig, "m1_regression.png")

# ══════════════════════════════════════════════════════════════════
say("\n" + "=" * 70)
say("9. 분석 카드용 집계 (교재 3.11.2)")
say("=" * 70)

# 카드 ① 장르 상위 15
gtop = rel["genre"]["value"].value_counts()
g15 = gtop.head(15)
REGION = ["International Movies", "International TV Shows", "Independent Movies", "Movies", "TV Shows"]
card1 = pd.DataFrame({"관계 수": g15, "작품 비율(%)": (g15 / N * 100).round(1),
                      "성격": ["지역·형식 태그" if g in REGION else "내용 장르" for g in g15.index]})
dump(card1, "card1_genre_top15.csv")
tagcnt = rel["genre"].groupby("show_id").size()
say(f"\n[카드①] 장르 상위 15")
say(card1.to_string())
say(f"  전체 장르 {gtop.size}개, 관계 {len(rel['genre']):,}건입니다. 작품당 평균 {len(rel['genre']) / N:.2f}개 장르가 붙어 있습니다.")
say(f"  태그 1개 작품 {int((tagcnt == 1).sum()):,}편, 2개 {int((tagcnt == 2).sum()):,}편, 3개 {int((tagcnt == 3).sum()):,}편입니다.")
say(f"  상위 15개 관계 수 합계는 {int(g15.sum()):,}건으로 전체의 {g15.sum() / len(rel['genre']):.0%}입니다.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
t = card1[::-1]
ax.barh(range(len(t)), t["관계 수"], color=[ORANGE if s == "지역·형식 태그" else BLUE for s in t["성격"]])
for i, (v, p) in enumerate(zip(t["관계 수"], t["작품 비율(%)"])):
    ax.text(v + 25, i, f"{v:,}  ({p}%)", va="center", fontsize=8)
ax.set_yticks(range(len(t)), t.index, fontsize=9)
ax.scatter([], [], marker="s", c=BLUE, label="내용 장르")
ax.scatter([], [], marker="s", c=ORANGE, label="지역·형식 태그")
ax.legend(loc="lower right", fontsize=9)
ax.set(title=f"[그림 36] 장르 상위 15 (관계 {len(rel['genre']):,}건 / 작품 {N:,}편)", xlabel="관계 수", xlim=(0, 2900))
ax = axes[1]
vc = tagcnt.value_counts().sort_index()
ax.bar(vc.index, vc.values, color=PURPLE)
for i, v in vc.items():
    ax.text(i, v + 60, f"{v:,}", ha="center", fontsize=9)
ax.set_xticks(vc.index.tolist())
ax.set(title="[그림 37] 한 작품에 붙은 장르 태그 수", xlabel="태그 수", ylabel="작품 수")
save(fig, "card1_genre_top15.png")

# 카드 ② 등급별 러닝타임 + 장르 통제
kid_ids = set(rel["genre"].loc[rel["genre"]["value"].eq("Children & Family Movies"), "show_id"])
mr["is_kids"] = mr["show_id"].isin(kid_ids)
grp = (mr.groupby(["rating_system", "rating"])["duration_value"]
       .agg(valid_n="count", median="median", q1=lambda s: s.quantile(.25),
            q3=lambda s: s.quantile(.75), mean="mean", sd=lambda s: s.std(ddof=0)))
order = [r for r in MPAA + TVR if r in grp.index.get_level_values("rating")]
grp = grp.reset_index().set_index("rating").loc[order]
grp["kids_share(%)"] = (mr.groupby("rating")["is_kids"].mean().loc[order] * 100).round(1)
grp = grp.join(mr[~mr["is_kids"]].groupby("rating")["duration_value"].agg(n_excl="count", median_excl="median"))
dump(grp, "card2_rating_duration.csv")
say(f"\n[카드②] 등급별 영화 상영시간")
say(f"  대상은 Movie {int(mv.sum()):,}편에서 5주차에 채운 rating {int((mv & df['rating_imputed'].eq(1)).sum())}건을 뺀 {len(mr):,}편입니다.")
say(grp.round(1).to_string())
say("  kids_share는 그 등급에서 Children & Family Movies 태그가 붙은 비율입니다.")
say("  median_excl은 그 장르를 뺀 뒤의 중앙값입니다.")

show = [r for r in order if grp.loc[r, "valid_n"] >= 30]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
sns.boxplot(data=mr[mr.rating.isin(show)], x="rating", y="duration_value", order=show,
            hue="rating_system", palette={"MPAA(영화 등급)": "#8ecae6", "TV 등급": ORANGE},
            dodge=False, fliersize=1.8, ax=ax)
for i, r in enumerate(show):
    ax.text(i, -16, f"n={int(grp.loc[r, 'valid_n']):,}", ha="center", fontsize=7.5, color="#555")
ax.axvline(len([r for r in show if r in MPAA]) - .5, color="#555", lw=1.2)
ax.set_ylim(-28, 330)
ax.legend(title="", loc="upper left", fontsize=9)
ax.set(title="[그림 38] 등급별 영화 상영시간 (n≥30 등급만)", xlabel="시청 등급", ylabel="분")
ax = axes[1]
ko = [r for r in show if r in TVR]
xx = np.arange(len(ko))
ax.plot(xx, grp.loc[ko, "median"], "o-", color=ORANGE, lw=2, label="전체")
ax.plot(xx, grp.loc[ko, "median_excl"], "s--", color=PURPLE, lw=2, label="Children & Family 장르 제외")
for i, r in enumerate(ko):
    ax.text(i, 120, f"{grp.loc[r, 'kids_share(%)']:.0f}%", ha="center", fontsize=8, color="#777")
ax.set_xticks(xx, ko, fontsize=9)
ax.set_ylim(10, 132)
ax.legend(fontsize=9, loc="lower right")
ax.set(title="[그림 39] 어린이·가족 장르를 뺐을 때의 중앙값\n(위 회색 숫자 = 그 등급에서 해당 장르 비율)",
       xlabel="TV 등급", ylabel="중앙값(분)")
save(fig, "card2_rating_duration.png")

u40 = mr[mr.duration_value < 40]
say(f"  40분 미만 영화는 {len(u40)}편입니다. 이 중 어린이 등급이 "
    f"{int(u40.rating.isin(['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'G']).sum())}편, "
    f"Children & Family 장르가 {int(u40.is_kids.sum())}편입니다.")

# 카드 ③ 연도별 등록 (차트는 v1 재사용, 별도 저장)
say(f"\n[카드③] 연도별 등록 건수 — 위 3-1의 집계를 그대로 사용합니다.")
say(f"  2015년 이전 8년 합계는 {int(annual.loc[:2015, '합계'].sum())}편으로 전체의 "
    f"{annual.loc[:2015, '합계'].sum() / annual['합계'].sum():.1%}입니다.")
say(f"  2016~2020년 합계는 {int(annual.loc[2016:2020, '합계'].sum()):,}편으로 전체의 "
    f"{annual.loc[2016:2020, '합계'].sum() / annual['합계'].sum():.1%}입니다.")
say(f"  2021년 117편은 16일 동안의 값이라 하루 {117 / 16:.1f}편입니다. 2020년은 하루 {2009 / 366:.1f}편입니다.")

df.to_csv(OUT / "netflix_week6.csv", index=False, encoding="utf-8-sig")
(OUT / "_report_week6.txt").write_text("\n".join(log), encoding="utf-8")
say("\n" + "=" * 70)
say(f"차트 {len(list(OUT.glob('*.png')))}장, 집계표 {len(list(TAB.glob('*.csv')))}개를 저장했습니다.")
