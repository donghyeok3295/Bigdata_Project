"""타이타닉 과제 (중간발표용): 기초통계 -> 결측 -> 영향 요소 -> 질문 10개 -> 질문 가치 검증 -> 질문 유형별 모델 선택.

원본 train.csv/test.csv는 수정하지 않는다. test.csv에는 정답이 없어 기초통계·결측 확인·티켓 일행 수 계산에만 쓰고 모델 평가에는 쓰지 않는다.
구간은 기초통계(분위수)로 경계를 정해 구간마다 인원이 비슷하게 나눈다(나이 5분위, 요금 4분위). 이산 변수는 값 그대로 보고 n<30 집단은 해석하지 않는다.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "output"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
if "--eda-only" not in sys.argv:
    for old in OUT.glob("*.png"):
        old.unlink()

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "figure.dpi": 110, "savefig.dpi": 140, "axes.titlesize": 12})
TEAL, ORANGE, BLUE, GREY, RED, PURPLE = "#277f8a", "#e08a4f", "#4361a6", "#9aa5b4", "#c0392b", "#7b5ea7"
SEED = 42
MIN_N = 30
log = []


def say(msg=""):
    print(msg)
    log.append(str(msg))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate(df, by):
    g = df.groupby(by, observed=True)["Survived"].agg(n="size", surv="sum")
    g["rate"] = g["surv"] / g["n"] * 100
    ci = [wilson(k, n) for k, n in zip(g["surv"], g["n"])]
    g["lo"] = [c[0] * 100 for c in ci]
    g["hi"] = [c[1] * 100 for c in ci]
    return g


def bar_rate(ax, g, labels=None, color=TEAL, width=.7, x=None, offset=0, label=None):
    x = np.arange(len(g)) if x is None else x
    bars = ax.bar(x + offset, g["rate"], width, color=color, label=label,
                  yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]], error_kw={"lw": 1, "capsize": 2, "ecolor": "#444"})
    for b, n in zip(bars, g["n"]):
        if n < MIN_N:
            b.set_alpha(.35)
    for xi, (r, k, n) in zip(x + offset, zip(g["rate"], g["surv"], g["n"])):
        ax.text(xi, min(r + 9, 104), f"{r:.0f}%\n{int(k)}/{int(n)}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("생존율 (%)")
    if labels is not None:
        ax.set_xticks(np.arange(len(g)))
        ax.set_xticklabels(labels)


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = chi2_contingency(t, correction=False)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))


# ---------------------------------------------------------------- 1. 데이터 확인·기초통계
train = pd.read_csv(BASE / "data" / "train.csv")
test = pd.read_csv(BASE / "data" / "test.csv")
say("=== 1. 항목 확인·기초통계 ===")
say(f"train {train.shape}, test {test.shape}, 합계 {len(train) + len(test)}명")
say(f"사망 {int((train.Survived == 0).sum())} / 생존 {int(train.Survived.sum())} ({train.Survived.mean() * 100:.1f}%)")
say(f"중복 행 {train.duplicated().sum()}, 중복 PassengerId {pd.concat([train, test]).PassengerId.duplicated().sum()}")

# 나이 구간: 기초통계(분위수)로 경계를 정해 구간마다 인원이 비슷하도록 한다
known_age = train.Age.dropna()
AGE_BINS = [0, *np.unique(np.round(known_age.quantile([.2, .4, .6, .8]).values)).astype(int).tolist(), 81]
AGE_LABELS = [f"{a}-{b - 1}" for a, b in zip(AGE_BINS[:-1], AGE_BINS[1:])]
CHILD = f"미성년({AGE_LABELS[0]}세)"
age_n = pd.cut(known_age, AGE_BINS, right=False, labels=AGE_LABELS).value_counts(sort=False)
say("나이 분위수 경계 " + str(AGE_BINS) + " 구간별 인원 " + age_n.to_dict().__str__())
say("참고: 10세 등간격이면 구간별 인원 " + str(pd.cut(known_age, range(0, 90, 10), right=False).value_counts(sort=False).tolist()))

rows = []
for c in ["Age", "Fare", "SibSp", "Parch"]:
    for s, d in [("train", train), ("test", test)]:
        x = d[c]
        rows.append({"변수": c, "데이터": s, "n": int(x.notna().sum()), "결측": int(x.isna().sum()),
                     "평균": x.mean(), "표준편차": x.std(), "최소": x.min(), "Q1": x.quantile(.25),
                     "중앙값": x.median(), "Q3": x.quantile(.75), "최대": x.max(), "왜도": x.skew()})
bs = pd.DataFrame(rows)
bs.to_csv(TAB / "s1_basic_numeric.csv", index=False, encoding="utf-8-sig")
say(bs.round(2).to_string())
cat_tabs = []
for c in ["Pclass", "Sex", "Embarked"]:
    t = pd.DataFrame({"train(%)": train[c].value_counts(normalize=True) * 100,
                      "test(%)": test[c].value_counts(normalize=True) * 100})
    say(f"{c} 구성비\n" + t.round(1).to_string())
    cat_tabs.append(t.assign(변수=c))
pd.concat(cat_tabs).to_csv(TAB / "s1_basic_categorical.csv", encoding="utf-8-sig")
say("요금 등간격 4구간 인원: " + str(pd.cut(train.Fare, 4).value_counts(sort=False).tolist())
    + " -> 등간격은 한 구간에 몰려 쓸 수 없어 요금만 같은 인원 4분위 사용")


def prop_bars(a, col, title):
    tr = train[col].value_counts(normalize=True) * 100
    te = test[col].value_counts(normalize=True) * 100
    idx = sorted(set(tr.index) | set(te.index))
    tr, te = tr.reindex(idx, fill_value=0), te.reindex(idx, fill_value=0)
    x = np.arange(len(idx))
    a.bar(x - .2, tr, .4, color=TEAL, label=f"train (n={train[col].notna().sum()})")
    a.bar(x + .2, te, .4, color=ORANGE, label=f"test (n={test[col].notna().sum()})")
    a.set_xticks(x)
    a.set_xticklabels([str(i) for i in idx])
    a.set_ylabel("비율 (%)")
    a.set_title(title)
    a.legend(fontsize=8)


fig, ax = plt.subplots(2, 4, figsize=(18, 8))
for s, d, col in [("train", train, TEAL), ("test", test, ORANGE)]:
    a = d.Age.dropna()
    ax[0, 0].hist(a, bins=range(0, 85, 5), weights=np.ones(len(a)) / len(a) * 100, alpha=.55, color=col, label=f"{s} (n={len(a)})")
    f = np.log1p(d.Fare.dropna())
    ax[0, 1].hist(f, bins=np.linspace(0, 6.5, 14), weights=np.ones(len(f)) / len(f) * 100, alpha=.55, color=col, label=s)
for e in AGE_BINS[1:-1]:
    ax[0, 0].axvline(e, color="#333", ls=":", lw=1)
ax[0, 0].set_title("Age 분포 (5세 간격, 점선 = 분위수 구간 경계)")
ax[0, 0].set_xlabel("나이 (세)")
ax[0, 0].set_ylabel("비율 (%)")
ax[0, 0].legend(fontsize=8)
ticks = [0, 5, 10, 20, 50, 100, 250, 500]
ax[0, 1].set_xticks(np.log1p(ticks))
ax[0, 1].set_xticklabels(ticks)
ax[0, 1].set_title("Fare 분포 (로그 눈금 등간격, 왜도 4.8)")
ax[0, 1].set_xlabel("요금 (파운드)")
ax[0, 1].legend(fontsize=8)
prop_bars(ax[0, 2], "SibSp", "SibSp (형제·배우자 수)")
prop_bars(ax[0, 3], "Parch", "Parch (부모·자녀 수)")
prop_bars(ax[1, 0], "Pclass", "객실 등급")
prop_bars(ax[1, 1], "Sex", "성별")
prop_bars(ax[1, 2], "Embarked", "승선 항구")
sv = train.Survived.value_counts().sort_index()
ax[1, 3].bar(["사망 (0)", "생존 (1)"], sv.values, color=[GREY, TEAL])
for i, v_ in enumerate(sv.values):
    ax[1, 3].text(i, v_ + 8, f"{v_}명 ({v_ / len(train) * 100:.1f}%)", ha="center")
ax[1, 3].set_ylim(0, 640)
ax[1, 3].set_title("정답 분포 (train만)")
save(fig, "s1_basic_stats.png")

# ---------------------------------------------------------------- 2. 결측치 확인
say("\n=== 2. 결측치 확인 ===")
miss = pd.DataFrame({"train": train.isna().sum(), "test": test.isna().sum()})
miss["train_pct"] = miss["train"] / len(train) * 100
miss["test_pct"] = miss["test"] / len(test) * 100
miss = miss[(miss.train > 0) | (miss.test > 0)].sort_values("train_pct", ascending=False)
say(miss.round(2).to_string())
miss.to_csv(TAB / "s2_missing.csv", encoding="utf-8-sig")
by_cls = train.groupby("Pclass").agg(Age=("Age", lambda s: s.isna().mean() * 100),
                                     Cabin=("Cabin", lambda s: s.isna().mean() * 100))
say("등급별 결측률(%)\n" + by_cls.round(1).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ys = np.arange(len(miss))
ax[0].barh(ys + .2, miss.train_pct, .4, color=TEAL, label="train")
ax[0].barh(ys - .2, miss.test_pct, .4, color=ORANGE, label="test")
for y_, (a, b, an, bn) in zip(ys, zip(miss.train_pct, miss.test_pct, miss.train, miss.test)):
    ax[0].text(a + 1, y_ + .2, f"{an}명 ({a:.1f}%)", va="center", fontsize=8)
    ax[0].text(b + 1, y_ - .2, f"{bn}명 ({b:.1f}%)", va="center", fontsize=8)
ax[0].set_yticks(ys)
ax[0].set_yticklabels(miss.index)
ax[0].set_xlim(0, 105)
ax[0].invert_yaxis()
ax[0].legend()
ax[0].set_title("컬럼별 결측률")
ax[0].set_xlabel("결측률 (%)")
xs = np.arange(3)
ax[1].bar(xs - .19, by_cls.Age, .38, color=BLUE, label="Age 결측률")
ax[1].bar(xs + .19, by_cls.Cabin, .38, color=GREY, label="Cabin 결측률")
for x, (a, c) in zip(xs, zip(by_cls.Age, by_cls.Cabin)):
    ax[1].text(x - .19, a + 1.5, f"{a:.0f}%", ha="center", fontsize=8)
    ax[1].text(x + .19, c + 1.5, f"{c:.0f}%", ha="center", fontsize=8)
ax[1].set_xticks(xs)
ax[1].set_xticklabels(["1등급", "2등급", "3등급"])
ax[1].set_ylim(0, 105)
ax[1].legend()
ax[1].set_title("등급별 결측률 — 결측은 무작위가 아니다")
ax[1].set_ylabel("결측률 (%)")
save(fig, "s2_missing.png")

# ---------------------------------------------------------------- 3. 결측치 처리
say("\n=== 3. 결측치 제거·처리 ===")
cc = train.drop(columns="Cabin").dropna()
excluded = train[~train.PassengerId.isin(cc.PassengerId)]
steps = [("원본 전체", train), ("Cabin 제외 후\n행 제거(Age·Embarked)", cc),
         ("제외된 승객", excluded), ("모든 컬럼 dropna", train.dropna())]
st = pd.DataFrame({"집단": [s[0].replace("\n", " ") for s in steps], "인원": [len(s[1]) for s in steps],
                   "생존자": [int(s[1].Survived.sum()) for s in steps],
                   "생존율(%)": [s[1].Survived.mean() * 100 for s in steps]})
st.to_csv(TAB / "s3_drop_effect.csv", index=False, encoding="utf-8-sig")
say(st.round(1).to_string())


def title_of(name):
    t = name.split(",")[1].split(".")[0].strip()
    t = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}.get(t, t)
    return t if t in ("Mr", "Mrs", "Miss", "Master") else "Rare"


both = pd.concat([train, test], ignore_index=True)
df = train.copy()
df["Title"] = df.Name.map(title_of)
df["FamilySize"] = df.SibSp + df.Parch + 1
df["IsAlone"] = (df.FamilySize == 1).astype(int)
df["TicketSize"] = df.Ticket.map(both.Ticket.value_counts())
df["HasCabin"] = df.Cabin.notna().astype(int)
df["Deck"] = df.Cabin.str[0].fillna("U")
df["AgeMissing"] = df.Age.isna().astype(int)
grp_med = df.groupby(["Title", "Pclass"]).Age.median()
df["Age_filled"] = df.apply(lambda r: grp_med[(r.Title, r.Pclass)] if pd.isna(r.Age) else r.Age, axis=1)
df["Embarked_filled"] = df.Embarked.fillna(df.Embarked.mode()[0])
df["AgeBand"] = pd.cut(df.Age, AGE_BINS, right=False, labels=AGE_LABELS)
df.drop(columns=["Cabin"]).to_csv(OUT / "titanic_cleaned.csv", index=False, encoding="utf-8-sig")
say("호칭×등급 Age 중앙값\n" + grp_med.unstack().round(1).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].bar(range(4), st["생존율(%)"], color=[TEAL, BLUE, ORANGE, GREY])
for i, r in st.iterrows():
    ax[0].text(i, r["생존율(%)"] + 1.2, f"{int(r['생존자'])}/{r['인원']}\n{r['생존율(%)']:.1f}%", ha="center", fontsize=9)
ax[0].set_xticks(range(4))
ax[0].set_xticklabels([s[0] for s in steps], fontsize=9)
ax[0].set_ylim(0, 85)
ax[0].set_ylabel("생존율 (%)")
ax[0].set_title("삭제 기준에 따라 달라지는 생존율")
known = df.Age.dropna()
ax[1].hist(known, bins=range(0, 85, 5), color=TEAL, alpha=.85, label=f"관측 나이 ({len(known)}명)")
ax[1].hist(df.Age_filled, bins=range(0, 85, 5), histtype="step", color=RED, lw=2, label=f"호칭×등급 중앙값 대체 후 ({len(df)}명)")
ax[1].set_xlabel("나이 (세)")
ax[1].set_ylabel("인원 (명)")
ax[1].legend()
ax[1].set_title("Age 대체 전·후 분포")
save(fig, "s3_drop_effect.png")

# ---------------------------------------------------------------- 4. 생존 영향 요소
say("\n=== 4. 생존 영향 요소 (Cramér's V) ===")
fb = pd.qcut(df.Fare, 4, labels=["Q1", "Q2", "Q3", "Q4"])
emb = df.dropna(subset=["Embarked"])
agk = df.dropna(subset=["AgeBand"])
v = {"호칭 (Title)": cramers_v(df.Title, df.Survived), "성별 (Sex)": cramers_v(df.Sex, df.Survived),
     "객실 등급 (Pclass)": cramers_v(df.Pclass, df.Survived), "요금 4분위 (Fare)": cramers_v(fb, df.Survived),
     "객실 기록 (HasCabin)": cramers_v(df.HasCabin, df.Survived),
     "가족 규모 값 (FamilySize)": cramers_v(df.FamilySize, df.Survived),
     "티켓 일행 수 값 (TicketSize)": cramers_v(df.TicketSize, df.Survived),
     "승선 항구 (Embarked)": cramers_v(emb.Embarked, emb.Survived),
     "나이 분위 구간 (Age)": cramers_v(agk.AgeBand, agk.Survived)}
vs = pd.Series(v).sort_values(ascending=False)
say(vs.round(3).to_string())
vs.to_csv(TAB / "s4_cramers_v.csv", header=["Cramers_V"], encoding="utf-8-sig")

cand = {}
for (sx, pc), g in df.groupby(["Sex", "Pclass"]):
    cand[f"{'여성' if sx == 'female' else '남성'} {pc}등급"] = g
cand[CHILD] = df[df.Age < AGE_BINS[1]]
cand["혼자 탑승"] = df[df.FamilySize == 1]
cand["가족 2-4명"] = df[df.FamilySize.between(2, 4)]
cand["가족 5명 이상"] = df[df.FamilySize >= 5]
cand["객실 기록 있음"] = df[df.HasCabin == 1]
cand["객실 기록 없음"] = df[df.HasCabin == 0]
gr = pd.DataFrame({k: {"n": len(g), "surv": int(g.Survived.sum()), "rate": g.Survived.mean() * 100} for k, g in cand.items()}).T.sort_values("rate")
gr.to_csv(TAB / "s4_group_rates.csv", encoding="utf-8-sig")
say(gr.round(1).to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
ax[0].barh(vs.index[::-1], vs.values[::-1], color=[TEAL if x >= .25 else GREY for x in vs.values[::-1]])
for i, x in enumerate(vs.values[::-1]):
    ax[0].text(x + .005, i, f"{x:.2f}", va="center", fontsize=9)
ax[0].set_xlim(0, .62)
ax[0].set_xlabel("Cramér's V (0=무관, 1=완전 연관)")
ax[0].set_title("생존과의 단변량 연관 강도 (인과 아님)")
ax[1].barh(gr.index, gr.rate, color=[RED if r < 30 else (TEAL if r > 60 else GREY) for r in gr.rate])
for i, (r, n) in enumerate(zip(gr.rate, gr.n)):
    ax[1].text(r + 1, i, f"{r:.0f}% (n={int(n)})", va="center", fontsize=8)
ax[1].axvline(train.Survived.mean() * 100, color="#333", ls="--", lw=1)
ax[1].set_xlim(0, 118)
ax[1].set_xlabel("생존율 (%) — 점선은 전체 38.4%")
ax[1].set_title("그룹별 생존율 비교")
save(fig, "s4_drivers.png")

# ---------------------------------------------------------------- 5. 팀 질문 10개
say("\n=== 5. 팀 질문 10개 ===")
SX = {"female": "여성", "male": "남성"}
sex_c = {"female": TEAL, "male": ORANGE}

# Q1 성별 x 등급
g1 = rate(df, ["Pclass", "Sex"])
g1.to_csv(TAB / "q1_sex_class.csv", encoding="utf-8-sig")
say("Q1\n" + g1.round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for off, sx in zip([-.2, .2], ["female", "male"]):
    bar_rate(ax[0], g1.xs(sx, level="Sex"), color=sex_c[sx], width=.38, offset=off, label=SX[sx])
ax[0].set_xticks(range(3))
ax[0].set_xticklabels(["1등급", "2등급", "3등급"])
ax[0].legend()
ax[0].set_title("성별 × 등급별 생존율 (95% Wilson 구간)")
comp = g1.copy()
comp["인원 비중"] = comp.n / comp.n.sum() * 100
comp["생존자 비중"] = comp.surv / comp.surv.sum() * 100
yy = np.arange(6)
ax[1].barh(yy + .2, comp["인원 비중"], .4, color=GREY, label="전체 승객 중 비중")
ax[1].barh(yy - .2, comp["생존자 비중"], .4, color=BLUE, label="생존자 중 비중")
for y_, (a, b) in zip(yy, zip(comp["인원 비중"], comp["생존자 비중"])):
    ax[1].text(a + .4, y_ + .2, f"{a:.1f}%", va="center", fontsize=8)
    ax[1].text(b + .4, y_ - .2, f"{b:.1f}%", va="center", fontsize=8)
ax[1].set_yticks(yy)
ax[1].set_yticklabels([f"{SX[s]} {p}등급" for p, s in comp.index])
ax[1].invert_yaxis()
ax[1].set_xlim(0, max(comp["인원 비중"].max(), comp["생존자 비중"].max()) * 1.2)
ax[1].legend(loc="center right")
ax[1].set_xlabel("비중 (%)")
ax[1].set_title("누가 승객이었고 누가 살아남았나")
save(fig, "q1_sex_class.png")

# Q4 가족 규모: 구간으로 묶지 않고 값 그대로
g2 = rate(df, "FamilySize")
sub2 = df[df.FamilySize <= 7]
g2c = rate(sub2, ["FamilySize", "Sex"])
g2.to_csv(TAB / "q4_family_size.csv", encoding="utf-8-sig")
g2c.to_csv(TAB / "q4_family_size_sex.csv", encoding="utf-8-sig")
big_cls = pd.crosstab(df.FamilySize >= 5, df.Pclass)
say("Q4\n" + g2.round(1).to_string() + "\n" + g2c.round(1).to_string() + "\n가족 5명 이상 여부 × 등급\n" + big_cls.to_string())
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
bar_rate(ax[0], g2, labels=[str(i) for i in g2.index], color=TEAL)
ax[0].set_xlabel("가족 규모 (SibSp + Parch + 1, 명)")
ax[0].set_title("가족 규모별 생존율 (값 그대로)")
vals = list(range(1, 8))
for off, sx in zip([-.2, .2], ["female", "male"]):
    gg = g2c.xs(sx, level="Sex")
    bar_rate(ax[1], gg, color=sex_c[sx], width=.38, x=np.array([vals.index(v_) for v_ in gg.index]), offset=off, label=SX[sx])
ax[1].set_xticks(range(7))
ax[1].set_xticklabels(vals)
ax[1].set_xlabel("가족 규모 (명) — 8명·11명은 13명 전원 사망으로 생략")
ax[1].legend()
ax[1].set_title("가족 규모 × 성별 생존율")
save(fig, "q4_family_size.png")

# Q5 가족 구성
df["FamType"] = np.select([(df.SibSp == 0) & (df.Parch == 0), (df.SibSp > 0) & (df.Parch == 0),
                           (df.SibSp == 0) & (df.Parch > 0)], ["혼자", "형제·배우자만", "부모·자녀만"], "둘 다 있음")
order3 = ["혼자", "형제·배우자만", "부모·자녀만", "둘 다 있음"]
g3 = rate(df, "FamType").loc[order3]
g3s = rate(df, ["FamType", "Sex"])
g3.to_csv(TAB / "q5_family_type.csv", encoding="utf-8-sig")
g3s.to_csv(TAB / "q5_family_type_sex.csv", encoding="utf-8-sig")
say("Q5\n" + g3.round(1).to_string() + "\n" + g3s.round(1).to_string())
hm = df.pivot_table(index="SibSp", columns="Parch", values="Survived", aggfunc="mean").reindex(index=range(6), columns=range(6)) * 100
hn = df.pivot_table(index="SibSp", columns="Parch", values="Survived", aggfunc="size").reindex(index=range(6), columns=range(6))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
bar_rate(ax[0], g3, labels=order3, color=BLUE)
ax[0].set_title("가족 구성 유형별 생존율")
im = ax[1].imshow(hm.values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
for i in range(6):
    for j in range(6):
        if not np.isnan(hm.values[i, j]):
            ax[1].text(j, i, f"{hm.values[i, j]:.0f}%\nn={int(hn.values[i, j])}", ha="center", va="center", fontsize=7,
                       color="white" if hm.values[i, j] > 60 else "black")
ax[1].set_xticks(range(6))
ax[1].set_yticks(range(6))
ax[1].set_xlabel("Parch (부모·자녀 수)")
ax[1].set_ylabel("SibSp (형제·배우자 수)")
ax[1].set_title("SibSp × Parch 생존율 (칸 안은 인원)")
fig.colorbar(im, ax=ax[1], label="생존율 (%)")
save(fig, "q5_family_type.png")

# Q3 나이(분위수 구간) x 성별
kn = df.dropna(subset=["Age"])
g4 = rate(kn, ["AgeBand", "Sex"])
g4a = rate(kn, "AgeBand")
g4.to_csv(TAB / "q3_age_sex.csv", encoding="utf-8-sig")
say("Q3\n" + g4a.round(1).to_string() + "\n" + g4.round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
for off, sx in zip([-.2, .2], ["female", "male"]):
    gg = g4.xs(sx, level="Sex")
    bar_rate(ax[0], gg, color=sex_c[sx], width=.38, x=np.array([AGE_LABELS.index(i) for i in gg.index]), offset=off, label=SX[sx])
ax[0].set_xticks(range(len(AGE_LABELS)))
ax[0].set_xticklabels([f"{a}세" for a in AGE_LABELS], fontsize=8)
ax[0].legend(loc="upper right")
ax[0].set_title(f"나이(분위수 구간, 구간별 인원 비슷) × 성별 생존율 (나이 기록 {len(kn)}명)")
ax[1].hist([kn[kn.Survived == 1].Age, kn[kn.Survived == 0].Age], bins=range(0, 85, 5), stacked=True,
           color=[TEAL, GREY], label=["생존", "사망"])
ax[1].set_xlabel("나이 (세)")
ax[1].set_ylabel("인원 (명)")
ax[1].legend()
for e in AGE_BINS[1:-1]:
    ax[1].axvline(e, color="#333", ls=":", lw=1)
ax[1].set_title("나이 분포 — 생존·사망 구성 (5세 간격, 점선 = 구간 경계)")
save(fig, "q3_age_sex.png")

# Q2 호칭
order5 = ["Mr", "Master", "Miss", "Mrs", "Rare"]
g5 = rate(df, "Title").loc[order5]
am = df.groupby("Title").agg(결측률=("AgeMissing", lambda s: s.mean() * 100), 나이중앙값=("Age", "median")).loc[order5]
g5.join(am).to_csv(TAB / "q2_title.csv", encoding="utf-8-sig")
say("Q2\n" + g5.join(am).round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
bar_rate(ax[0], g5, labels=["Mr\n(성인 남성)", "Master\n(남아)", "Miss\n(미혼 여성)", "Mrs\n(기혼 여성)", "Rare\n(기타 직함)"], color=BLUE)
ax[0].set_title("호칭별 생존율 (호칭은 891명 모두 있음)")
x = np.arange(5)
ax[1].bar(x - .2, am.결측률, .4, color=ORANGE, label="Age 결측률 (%)")
ax2 = ax[1].twinx()
ax2.bar(x + .2, am.나이중앙값, .4, color=TEAL, label="관측 나이 중앙값 (세)")
for xi, (a, b) in zip(x, zip(am.결측률, am.나이중앙값)):
    ax[1].text(xi - .2, a + 1, f"{a:.0f}%", ha="center", fontsize=8)
    ax2.text(xi + .2, b + .6, f"{b:.0f}", ha="center", fontsize=8)
ax[1].set_xticks(x)
ax[1].set_xticklabels(order5)
ax[1].set_ylim(0, 40)
ax2.set_ylim(0, 50)
ax[1].set_ylabel("Age 결측률 (%)")
ax2.set_ylabel("나이 중앙값 (세)")
h1, l1 = ax[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
ax[1].set_title("호칭이 나이 결측을 메울 단서가 되는가")
save(fig, "q2_title.png")

# Q10 승선 항구
pk = df.dropna(subset=["Embarked"]).copy()
pk["Port"] = pk.Embarked.map({"C": "C (Cherbourg)", "Q": "Q (Queenstown)", "S": "S (Southampton)"})
mix = pd.crosstab(pk.Port, pk.Pclass, normalize="index") * 100
g6 = rate(pk, "Port")
g6c = rate(pk, ["Pclass", "Port"])
g6.to_csv(TAB / "q10_port.csv", encoding="utf-8-sig")
g6c.to_csv(TAB / "q10_port_class.csv", encoding="utf-8-sig")
say("Q10\n" + g6.round(1).to_string() + "\n" + mix.round(1).to_string() + "\n" + g6c.round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
left = np.zeros(3)
for pc, col in zip([1, 2, 3], [TEAL, BLUE, ORANGE]):
    ax[0].barh(mix.index, mix[pc], left=left, color=col, label=f"{pc}등급")
    for i, (l_, w_) in enumerate(zip(left, mix[pc])):
        ax[0].text(l_ + w_ / 2, i, f"{w_:.0f}%", ha="center", va="center", color="white", fontsize=9)
    left += mix[pc].values
ax[0].invert_yaxis()
ax[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
ax[0].set_xlabel("항구별 승객 중 등급 구성 (%)")
ax[0].set_title("항구마다 승객 등급 구성이 다르다")
pc_col = {"C (Cherbourg)": TEAL, "Q (Queenstown)": ORANGE, "S (Southampton)": BLUE}
xs = np.arange(4)
for off, port in zip([-.27, 0, .27], g6.index):
    sub = g6c.xs(port, level="Port")
    r = [g6.loc[port, "rate"]] + [sub.loc[p, "rate"] if p in sub.index else np.nan for p in [1, 2, 3]]
    ax[1].bar(xs + off, r, .26, color=pc_col[port], label=port)
    for xi, rv in zip(xs + off, r):
        if not np.isnan(rv):
            ax[1].text(xi, rv + 1, f"{rv:.0f}", ha="center", fontsize=7)
ax[1].set_xticks(xs)
ax[1].set_xticklabels(["항구 전체", "1등급", "2등급", "3등급"])
ax[1].set_ylim(0, 100)
ax[1].set_ylabel("생존율 (%)")
ax[1].legend(fontsize=8)
ax[1].set_title("항구 전체 vs 같은 등급끼리 비교")
save(fig, "q10_port.png")

# Q8 요금: 왜도 4.8이라 같은 인원 4분위
df["FareBand"] = pd.qcut(df.Fare, 4, labels=["Q1 (낮음)", "Q2", "Q3", "Q4 (높음)"])
df["FareVsClass"] = np.where(df.Fare >= df.groupby("Pclass").Fare.transform("median"), "등급 중앙값 이상", "등급 중앙값 미만")
g7 = rate(df, "FareBand")
g7c = rate(df, ["Pclass", "FareVsClass"])
g7.to_csv(TAB / "q8_fare_band.csv", encoding="utf-8-sig")
g7c.to_csv(TAB / "q8_fare_in_class.csv", encoding="utf-8-sig")
edges = df.Fare.quantile([0, .25, .5, .75, 1]).round(2).tolist()
say("Q8 요금 분위 경계 " + str(edges) + "\n" + g7.round(1).to_string() + "\n" + g7c.round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
bar_rate(ax[0], g7, labels=g7.index, color=TEAL)
ax[0].set_xlabel(f"요금 같은 인원 4분위 (경계 £{edges[1]}, {edges[2]}, {edges[3]})")
ax[0].set_title("요금 4분위별 생존율")
for off, k, col in zip([-.2, .2], ["등급 중앙값 미만", "등급 중앙값 이상"], [GREY, TEAL]):
    bar_rate(ax[1], g7c.xs(k, level="FareVsClass"), color=col, width=.38, offset=off, label=k)
ax[1].set_xticks(range(3))
ax[1].set_xticklabels(["1등급", "2등급", "3등급"])
ax[1].legend(loc="upper right")
ax[1].set_title("같은 등급 안에서 요금이 높으면 더 살았나")
save(fig, "q8_fare.png")

# Q9 객실 기록·갑판
g8 = rate(df[df.Deck != "T"], "Deck")
g8 = g8.loc[[d for d in list("ABCDEFG") + ["U"] if d in g8.index]]
g8c = rate(df, ["Pclass", "HasCabin"])
g8.to_csv(TAB / "q9_deck.csv", encoding="utf-8-sig")
g8c.to_csv(TAB / "q9_hascabin_class.csv", encoding="utf-8-sig")
say("Q9\n" + g8.round(1).to_string() + "\n" + g8c.round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
bar_rate(ax[0], g8, labels=[("기록 없음" if d == "U" else d) for d in g8.index], color=BLUE)
ax[0].set_xlabel("갑판 (Cabin 첫 글자, T는 1명이라 제외)")
ax[0].set_title("갑판별 생존율")
for off, k, col in zip([-.2, .2], [0, 1], [GREY, TEAL]):
    gg = g8c.xs(k, level="HasCabin")
    bar_rate(ax[1], gg, color=col, width=.38, x=np.array([p - 1 for p in gg.index]), offset=off,
             label="객실 기록 있음" if k else "객실 기록 없음")
ax[1].set_xticks(range(3))
ax[1].set_xticklabels(["1등급", "2등급", "3등급"])
ax[1].legend(loc="upper right")
ax[1].set_title("등급을 같게 두고 본 객실 기록 유무")
save(fig, "q9_cabin.png")

# Q6 티켓 일행
df["Companion"] = np.select(
    [(df.TicketSize == 1) & (df.FamilySize == 1), (df.TicketSize > 1) & (df.FamilySize == 1),
     (df.TicketSize == 1) & (df.FamilySize > 1)], ["혼자", "티켓만 공유\n(가족 기록 없음)", "가족 기록만\n(티켓 따로)"], "가족+티켓 공유")
order9 = ["혼자", "티켓만 공유\n(가족 기록 없음)", "가족 기록만\n(티켓 따로)", "가족+티켓 공유"]
g9 = rate(df, "TicketSize")
g9b = rate(df, "Companion").loc[order9]
g9.to_csv(TAB / "q6_ticket_size.csv", encoding="utf-8-sig")
g9b.to_csv(TAB / "q6_companion.csv", encoding="utf-8-sig")
say("Q6\n" + g9.round(1).to_string() + "\n" + g9b.round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
bar_rate(ax[0], g9, labels=[str(i) for i in g9.index], color=TEAL)
ax[0].set_xlabel("같은 티켓 번호를 가진 승객 수 (train+test 기준, 값 그대로)")
ax[0].set_title("티켓 일행 수별 생존율")
bar_rate(ax[1], g9b, labels=order9, color=ORANGE)
ax[1].tick_params(axis="x", labelsize=8)
ax[1].set_title("동행 유형별 생존율")
save(fig, "q6_ticket.png")


# Q7 여성·어린이 우선: 미성년 기준은 Q3 나이 첫 분위 구간과 같게 맞춤
def who(r):
    if (pd.notna(r.Age) and r.Age < AGE_BINS[1]) or (pd.isna(r.Age) and r.Title == "Master"):
        return CHILD
    return "성인 여성" if r.Sex == "female" else "성인 남성"


df["Who"] = df.apply(who, axis=1)
wh = rate(df, ["Who", "Pclass"])
wh.to_csv(TAB / "q7_who_class.csv", encoding="utf-8-sig")
ch = df[df.Who == CHILD]
chf = rate(ch, "FamilySize")
chf.to_csv(TAB / "q7_child_family.csv", encoding="utf-8-sig")
say("Q7\n" + wh.round(1).to_string() + "\n미성년 가족 규모\n" + chf.round(1).to_string()
    + "\n미성년 가족 5명 이상 × 등급\n" + pd.crosstab(ch.FamilySize >= 5, ch.Pclass).to_string())
who_order = ["성인 여성", CHILD, "성인 남성"]
hm10 = pd.DataFrame({p: [wh.loc[(w_, p), "rate"] if (w_, p) in wh.index else np.nan for w_ in who_order] for p in [1, 2, 3]}, index=who_order)
hn10 = pd.DataFrame({p: [int(wh.loc[(w_, p), "n"]) if (w_, p) in wh.index else 0 for w_ in who_order] for p in [1, 2, 3]}, index=who_order)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
im = ax[0].imshow(hm10.values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
for i in range(3):
    for j in range(3):
        v_ = hm10.values[i, j]
        ax[0].text(j, i, f"{v_:.0f}%\nn={hn10.values[i, j]}", ha="center", va="center", fontsize=10,
                   color="white" if v_ > 60 else "black")
ax[0].set_xticks(range(3))
ax[0].set_xticklabels(["1등급", "2등급", "3등급"])
ax[0].set_yticks(range(3))
ax[0].set_yticklabels(who_order)
ax[0].set_title("승객 유형 × 등급별 생존율")
fig.colorbar(im, ax=ax[0], label="생존율 (%)")
bar_rate(ax[1], chf, labels=[str(i) for i in chf.index], color=TEAL)
ax[1].set_xlabel("미성년의 가족 규모 (명, 값 그대로)")
ax[1].set_title(f"미성년의 가족 규모별 생존율 (미성년 {len(ch)}명, 흐린 막대 n<{MIN_N})")
save(fig, "q7_women_children.png")

# ---------------------------------------------------------------- 6. 질문 가치 검증
say("\n=== 6. 질문 가치 검증: 단변량 χ² + 성별·등급 통제 후 우도비 검정 ===")


def holm(p):
    p = np.asarray(p, dtype=float)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def dummies(d, cols=(), inter=()):
    parts = [pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True) for c in cols]
    for a, b in inter:
        A = pd.get_dummies(d[a].astype(str), prefix=a, drop_first=True)
        B = pd.get_dummies(d[b].astype(str), prefix=b, drop_first=True)
        parts += [(A[ca] * B[cb]).rename(f"{ca}:{cb}") for ca in A for cb in B]
    return pd.concat(parts, axis=1).astype(float)


def loglik(M, yv):
    m = LogisticRegression(C=1e6, max_iter=20000).fit(M, yv)
    p = np.clip(m.predict_proba(M)[:, 1], 1e-12, 1 - 1e-12)
    return np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p))


def rank_of(M):
    return np.linalg.matrix_rank(np.column_stack([np.ones(len(M)), M.values]))


def lr_test(d, extra):
    base = dummies(d, ["Sex", "Pclass"])
    full = pd.concat([base, extra], axis=1)
    full = full.loc[:, (full.std() > 0) & ~full.columns.duplicated()]
    yv = d.Survived.values
    stat = 2 * (loglik(full, yv) - loglik(base, yv))
    dfree = rank_of(full) - rank_of(base)
    return stat, dfree, chi2_dist.sf(stat, dfree)


def uni(var, yv):
    t = pd.crosstab(var, yv)
    return chi2_contingency(t, correction=False)[1], cramers_v(var, yv)


sexcls = df.Sex + "_" + df.Pclass.astype(str)
QV = [("Q1", "성별×등급 6개 그룹", "성별·등급 주효과 위에 성별×등급 교호작용", df, sexcls,
       lambda d: dummies(d, inter=[("Sex", "Pclass")])),
      ("Q4", "가족 규모(값)", "가족 규모(값별 범주)", df, df.FamilySize, lambda d: dummies(d, ["FamilySize"])),
      ("Q5", "가족 구성 유형", "가족 구성 유형 4개", df, df.FamType, lambda d: dummies(d, ["FamType"])),
      ("Q3", "나이 분위 구간", "나이 구간 + 나이 구간×성별", kn, kn.AgeBand,
       lambda d: dummies(d, ["AgeBand"], inter=[("AgeBand", "Sex")])),
      ("Q2", "호칭", "호칭", df, df.Title, lambda d: dummies(d, ["Title"])),
      ("Q10", "승선 항구", "승선 항구", pk, pk.Embarked, lambda d: dummies(d, ["Embarked"])),
      ("Q8", "요금 4분위", "log(요금+1) 연속값", df, df.FareBand, lambda d: np.log1p(d[["Fare"]])),
      ("Q9", "객실 기록 유무", "객실 기록 유무", df, df.HasCabin, lambda d: d[["HasCabin"]].astype(float)),
      ("Q6", "동행 유형", "동행 유형 4개", df, df.Companion, lambda d: dummies(d, ["Companion"])),
      ("Q7", "승객 유형(성인 여성·미성년·성인 남성)", "미성년 여부", df, df.Who,
       lambda d: (d.Who == CHILD).astype(float).to_frame("Child"))]
qv_rows = []
for q, var_name, test_name, d, var, extra in QV:
    p_u, cv = uni(var, d.Survived)
    stat, dfree, p_lr = lr_test(d, extra(d))
    qv_rows.append({"질문": q, "주 변수": var_name, "n": len(d), "Cramér's V": cv, "단변량 p": p_u,
                    "통제 후 추가 검정": test_name, "LR χ²": stat, "자유도": dfree, "통제 후 p": p_lr})
qv = pd.DataFrame(qv_rows)
qv = qv.iloc[qv["질문"].str[1:].astype(int).argsort()].reset_index(drop=True)
qv["단변량 Holm p"] = holm(qv["단변량 p"])
qv["통제 후 Holm p"] = holm(qv["통제 후 p"])
qv["판정"] = np.where(qv["통제 후 Holm p"] < .05, "통제 후에도 설명력 있음 → 심층 분석", "통제하면 근거 부족 → 등급·성별로 설명됨")
qv.to_csv(TAB / "s5_question_value.csv", index=False, encoding="utf-8-sig")
say(qv.to_string())

fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
lab = [f"{r['질문']} {r['주 변수']}" for _, r in qv.iterrows()][::-1]
ax[0].barh(lab, qv["Cramér's V"][::-1], color=BLUE)
for i, (v_, p_) in enumerate(zip(qv["Cramér's V"][::-1], qv["단변량 Holm p"][::-1])):
    ax[0].text(v_ + .005, i, f"{v_:.2f} (Holm p={p_:.1e})", va="center", fontsize=8)
ax[0].set_xlim(0, .8)
ax[0].set_xlabel("Cramér's V")
ax[0].set_title("1단계: 질문 변수와 생존의 단변량 연관")
cols = [TEAL if p_ < .05 else GREY for p_ in qv["통제 후 Holm p"][::-1]]
ax[1].barh(lab, qv["LR χ²"][::-1], color=cols)
for i, (s_, d_, p_) in enumerate(zip(qv["LR χ²"][::-1], qv["자유도"][::-1], qv["통제 후 Holm p"][::-1])):
    ax[1].text(s_ + 1, i, f"χ²={s_:.1f}, df={d_}, Holm p={p_:.1e}", va="center", fontsize=8)
ax[1].set_xlim(0, qv["LR χ²"].max() * 1.6)
ax[1].set_xlabel("우도비 χ² (성별·등급 모델에 추가했을 때 늘어나는 설명력)")
ax[1].set_title("2단계: 성별·등급을 통제해도 남는가 (초록 = Holm p < 0.05)")
save(fig, "s5_question_value.png")

if "--eda-only" in sys.argv:
    sys.exit()

# ---------------------------------------------------------------- 7. 질문 유형별 모델 선택
say("\n=== 7. 모델 ===")
y = df.Survived.values
groups = df.Ticket.str.strip().str.upper().values
X = df.copy()
X["Pclass"] = X.Pclass.astype(str)
X["SexPclass"] = X.Sex + "_" + X.Pclass
X["FamCat"] = X.FamilySize.astype(str)

SET_A = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
SET_B = SET_A + ["Title", "FamilySize", "IsAlone", "TicketSize", "HasCabin"]
SET_LR_ENG = SET_B + ["SexPclass", "FamCat"]
NUM_ALL = {"Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketSize"}
BIN_ALL = {"IsAlone", "HasCabin"}
GRIDS = {"lr": {"m__C": [0.01, 0.03, 0.1, 0.3, 1, 10]},
         "dt": {"m__max_depth": [3, 4, 5], "m__min_samples_leaf": [5, 10]},
         "rf": {"m__max_depth": [5, None], "m__min_samples_leaf": [1, 3, 5]}}


def build(name, feats, **params):
    num = [f for f in feats if f in NUM_ALL]
    binary = [f for f in feats if f in BIN_ALL]
    cat = [f for f in feats if f not in NUM_ALL and f not in BIN_ALL]
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator="Age" in num))]
    if name == "lr":
        num_steps.append(("sc", StandardScaler()))
    parts = [("num", Pipeline(num_steps), num),
             ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                               ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)]
    if binary:
        parts.append(("bin", "passthrough", binary))
    if name == "lr":
        mdl = LogisticRegression(max_iter=5000)
    elif name == "dt":
        mdl = DecisionTreeClassifier(random_state=SEED)
    else:
        mdl = RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=SEED, n_jobs=-1)
    return Pipeline([("pre", ColumnTransformer(parts)), ("m", mdl)]).set_params(**params)


outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))


def nested(name, feats):
    oof = np.zeros(len(y))
    fold_auc, chosen = [], []
    for tr, va in outer:
        inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(X.iloc[tr], y[tr], groups[tr]))
        gs = GridSearchCV(build(name, feats), GRIDS[name], cv=inner, scoring="roc_auc").fit(X.iloc[tr][feats], y[tr])
        oof[va] = gs.predict_proba(X.iloc[va][feats])[:, 1]
        fold_auc.append(roc_auc_score(y[va], oof[va]))
        chosen.append(str(gs.best_params_))
    return oof, fold_auc, chosen


CONFIGS = [("lr", "A", SET_A, "로지스틱 회귀 · A 기본"), ("lr", "B", SET_B, "로지스틱 회귀 · B 파생"),
           ("lr", "ENG", SET_LR_ENG, "로지스틱 회귀 · B + 성별×등급·가족 범주"),
           ("dt", "A", SET_A, "결정트리 · A 기본"), ("dt", "B", SET_B, "결정트리 · B 파생"),
           ("rf", "A", SET_A, "랜덤 포레스트 · A 기본"), ("rf", "B", SET_B, "랜덤 포레스트 · B 파생")]
perf, OOF = [], {}
for name, key, feats, label in CONFIGS:
    oof, fa, ch_ = nested(name, feats)
    OOF[(name, key)] = oof
    perf.append({"설정": label, "AUC(폴드 평균)": np.mean(fa), "AUC 표준편차": np.std(fa),
                 "정확도": accuracy_score(y, oof >= .5), "F1": f1_score(y, oof >= .5),
                 "Brier": brier_score_loss(y, oof), "Log loss": log_loss(y, oof),
                 "선택된 설정(폴드별)": " | ".join(ch_)})
    say(f"{label}: AUC {np.mean(fa):.3f}±{np.std(fa):.3f}, 정확도 {accuracy_score(y, oof >= .5):.3f}, "
        f"Brier {brier_score_loss(y, oof):.4f}, 선택 {ch_}")
pf = pd.DataFrame(perf)
pf.to_csv(TAB / "m1_performance.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
yl = np.arange(len(pf))[::-1]
mc = [BLUE, BLUE, PURPLE, ORANGE, ORANGE, TEAL, TEAL]
ax[0].barh(yl, pf["AUC(폴드 평균)"], xerr=pf["AUC 표준편차"], color=mc, error_kw={"capsize": 3})
for yi, (a, s_) in zip(yl, zip(pf["AUC(폴드 평균)"], pf["AUC 표준편차"])):
    ax[0].text(a + s_ + .004, yi, f"{a:.3f}±{s_:.3f}", va="center", fontsize=8)
ax[0].set_yticks(yl)
ax[0].set_yticklabels(pf["설정"], fontsize=9)
ax[0].set_xlim(.7, .95)
ax[0].set_xlabel("ROC-AUC (Ticket 그룹 5겹 외부 검증, 폴드 평균 ± 표준편차)")
ax[0].set_title("같은 탐색 횟수(6개)로 각각 튜닝한 중첩 교차검증")
ax[1].barh(yl, pf["Brier"], color=mc)
for yi, b in zip(yl, pf["Brier"]):
    ax[1].text(b + .001, yi, f"{b:.4f}", va="center", fontsize=8)
ax[1].set_yticks(yl)
ax[1].set_yticklabels([])
ax[1].set_xlim(.1, .17)
ax[1].set_xlabel("Brier 점수 (확률 오차, 낮을수록 좋음)")
ax[1].set_title("확률의 정확도")
save(fig, "m1_performance.png")

# 7.2 교호작용·비선형: 관측 생존율 vs 외부 폴드 예측 확률
say("\n--- 교호작용(성별×등급)·비선형(가족 규모) 재현 ---")
MODELS4 = [(("lr", "B"), "로지스틱 · 주효과만", BLUE, "o"), (("lr", "ENG"), "로지스틱 · 교호작용·가족 범주 추가", PURPLE, "s"),
           (("dt", "B"), "결정트리", ORANGE, "^"), (("rf", "B"), "랜덤 포레스트", TEAL, "D")]
cell = df.Sex.map(SX) + " " + df.Pclass.astype(str) + "등급"
fam = np.where(df.FamilySize >= 8, "8·11", df.FamilySize.astype(str))
gap_rows = []
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for a, key, order in [(ax[0], cell, [f"{s} {p}등급" for s in ["여성", "남성"] for p in [1, 2, 3]]),
                      (ax[1], pd.Series(fam), [str(i) for i in range(1, 8)] + ["8·11"])]:
    key = pd.Series(np.asarray(key))
    obs = df.Survived.groupby(key.values).mean().reindex(order) * 100
    n_ = key.value_counts().reindex(order)
    a.bar(range(len(order)), obs, color="#dfe3e8", label="실제 생존율")
    for i, (o, n) in enumerate(zip(obs, n_)):
        a.text(i, 3, f"n={n}", ha="center", fontsize=7, color="#555")
    for mk, lbl, col, mark in MODELS4:
        pred = pd.Series(OOF[mk]).groupby(key.values).mean().reindex(order) * 100
        a.plot(range(len(order)), pred, marker=mark, color=col, label=lbl, lw=1.4, ms=6)
        gap_rows.append({"비교": "성별×등급" if a is ax[0] else "가족 규모", "모델": lbl,
                         "가중 평균 절대 오차(%p)": float(np.sum(n_ * (pred - obs).abs()) / n_.sum()),
                         **{f"{o}": round(pv, 1) for o, pv in zip(order, pred)}})
    a.set_xticks(range(len(order)))
    a.set_xticklabels(order, fontsize=8)
    a.set_ylim(0, 105)
    a.set_ylabel("생존율 / 평균 예측 확률 (%)")
ax[0].set_title("성별×등급: 실제(막대) vs 모델 평균 예측(선)")
ax[1].set_title("가족 규모: 실제(막대) vs 모델 평균 예측(선)")
ax[1].set_xlabel("가족 규모 (명)")
ax[0].legend(fontsize=8, loc="upper right")
save(fig, "m2_interaction_nonlinear.png")
gap = pd.DataFrame(gap_rows)
gap.to_csv(TAB / "m2_interaction_gap.csv", index=False, encoding="utf-8-sig")
say(gap.round(1).to_string())

# 7.3 로지스틱 회귀 오즈비 (Ticket 묶음 부트스트랩 95% 구간)
say("\n--- 로지스틱 회귀 오즈비 ---")


def or_design(d):
    return pd.DataFrame({"여성 (기준 남성)": (d.Sex == "female").astype(float),
                         "2등급 (기준 1등급)": (d.Pclass == 2).astype(float),
                         "3등급 (기준 1등급)": (d.Pclass == 3).astype(float),
                         "나이 +10세": d.Age.fillna(df.Age.median()) / 10,
                         "나이 기록 없음": d.Age.isna().astype(float),
                         "형제·배우자 +1명": d.SibSp.astype(float),
                         "부모·자녀 +1명": d.Parch.astype(float),
                         "요금 2배": np.log2(d.Fare + 1),
                         "Q항구 (기준 C)": (d.Embarked == "Q").astype(float),
                         "S항구 (기준 C)": (d.Embarked.fillna("S") == "S").astype(float)})


M = or_design(df)
coef = LogisticRegression(C=1e6, max_iter=20000).fit(M, y).coef_[0]
tix = df.Ticket.values
tix_idx = {t: np.where(tix == t)[0] for t in np.unique(tix)}
utix = np.array(list(tix_idx))
rng = np.random.default_rng(SEED)
boots = []
for _ in range(500):
    idx = np.concatenate([tix_idx[t] for t in rng.choice(utix, len(utix))])
    boots.append(LogisticRegression(C=1e6, max_iter=20000).fit(M.iloc[idx], y[idx]).coef_[0])
boots = np.array(boots)
ors = pd.DataFrame({"오즈비": np.exp(coef), "하한": np.exp(np.percentile(boots, 2.5, axis=0)),
                    "상한": np.exp(np.percentile(boots, 97.5, axis=0))}, index=M.columns)
ors.to_csv(TAB / "m3_odds_ratio.csv", encoding="utf-8-sig")
say(ors.round(3).to_string())
fig, ax = plt.subplots(figsize=(10, 5.2))
yo = np.arange(len(ors))[::-1]
sig = (ors.하한 > 1) | (ors.상한 < 1)
for yi, (o, lo, hi, s_) in zip(yo, zip(ors.오즈비, ors.하한, ors.상한, sig)):
    c = (TEAL if o > 1 else RED) if s_ else GREY
    ax.plot([lo, hi], [yi, yi], color=c, lw=2)
    ax.plot(o, yi, "o", color=c, ms=7)
    ax.text(hi * 1.08, yi, f"{o:.2f} [{lo:.2f}, {hi:.2f}]", va="center", fontsize=8)
ax.axvline(1, color="#333", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlim(.02, 60)
ax.set_yticks(yo)
ax.set_yticklabels(ors.index)
ax.set_xlabel("오즈비 (로그 눈금, 1보다 크면 생존 오즈 증가) · 95% 구간은 Ticket 묶음 부트스트랩 500회")
ax.set_title("로지스틱 회귀: 다른 변수를 고정했을 때 각 변수가 생존 오즈를 몇 배 바꾸나")
save(fig, "m3_odds_ratio.png")

# 7.4 결정트리 규칙 (깊이 3, 잎마다 최소 10명)
say("\n--- 결정트리 규칙 ---")
TF = pd.DataFrame({"여성": (df.Sex == "female").astype(int), "등급": df.Pclass, "나이": df.Age_filled,
                   "가족 규모": df.FamilySize, "요금": df.Fare})
dt3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=SEED).fit(TF, y)
dt3_oof = np.zeros(len(y))
for tr, va in outer:
    dt3_oof[va] = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=SEED).fit(TF.iloc[tr], y[tr]).predict_proba(TF.iloc[va])[:, 1]
say(f"깊이 3 트리 외부 폴드 AUC {roc_auc_score(y, dt3_oof):.3f} (풀링)")


def cond(f, op, t):
    le = op == "<="
    if f == "여성":
        return "남성" if le else "여성"
    if f == "등급":
        return ("1등급" if le else "2·3등급") if t < 2 else ("1·2등급" if le else "3등급")
    if f == "나이":
        return f"나이 {t:.1f}세 {'이하' if le else '초과'}"
    if f == "가족 규모":
        return f"가족 {int(np.floor(t))}명 {'이하' if le else '초과'}"
    return f"요금 £{t:.1f} {'이하' if le else '초과'}"


tr_ = dt3.tree_
leaves = []


def walk(node, path):
    if tr_.children_left[node] == -1:
        leaves.append((node, path))
        return
    f, t = TF.columns[tr_.feature[node]], tr_.threshold[node]
    walk(tr_.children_left[node], path + [cond(f, "<=", t)])
    walk(tr_.children_right[node], path + [cond(f, ">", t)])


walk(0, [])
leaf_of = dt3.apply(TF)
rules = pd.DataFrame([{"규칙": " & ".join(p), "n": int((leaf_of == nd).sum()),
                       "생존자": int(y[leaf_of == nd].sum()), "생존율(%)": y[leaf_of == nd].mean() * 100}
                      for nd, p in leaves]).sort_values("생존율(%)", ascending=False)
rules.to_csv(TAB / "m4_tree_rules.csv", index=False, encoding="utf-8-sig")
say(rules.round(1).to_string())
fig, ax = plt.subplots(figsize=(13, 5))
yr = np.arange(len(rules))[::-1]
ax.barh(yr, rules["생존율(%)"], color=[TEAL if r >= 50 else (RED if r < 20 else GREY) for r in rules["생존율(%)"]])
for yi, (r, k, n) in zip(yr, zip(rules["생존율(%)"], rules["생존자"], rules["n"])):
    ax.text(r + 1, yi, f"{r:.0f}% ({k}/{n})", va="center", fontsize=9)
ax.set_yticks(yr)
ax.set_yticklabels(rules["규칙"], fontsize=9)
ax.set_xlim(0, 115)
ax.set_xlabel("잎(규칙)에 속한 승객의 실제 생존율 (%)")
ax.set_title(f"결정트리(깊이 3): 어떤 조건 조합에서 생존율이 높은가 · 외부 폴드 AUC {roc_auc_score(y, dt3_oof):.3f}")
save(fig, "m4_tree_rules.png")

# 7.5 안정성: 부트스트랩 100회에서 결정트리 vs 랜덤 포레스트
say("\n--- 안정성 (부트스트랩 100회) ---")
pred_dt, pred_rf, roots = [], [], []
for b in range(100):
    idx = rng.integers(0, len(y), len(y))
    pdt = build("dt", SET_B, m__max_depth=4, m__min_samples_leaf=5).fit(X.iloc[idx][SET_B], y[idx])
    prf = build("rf", SET_B, m__min_samples_leaf=3).fit(X.iloc[idx][SET_B], y[idx])
    pred_dt.append(pdt.predict_proba(X[SET_B])[:, 1])
    pred_rf.append(prf.predict_proba(X[SET_B])[:, 1])
    names = pdt.named_steps["pre"].get_feature_names_out()
    t_ = pdt.named_steps["m"].tree_
    roots.append(" / ".join(names[t_.feature[n]].split("__")[1] for n in [0, t_.children_left[0], t_.children_right[0]]))
sd_dt, sd_rf = np.std(pred_dt, axis=0), np.std(pred_rf, axis=0)
rc = pd.Series(roots).value_counts()
stab = pd.DataFrame({"모델": ["결정트리", "랜덤 포레스트"], "승객별 예측 표준편차 평균": [sd_dt.mean(), sd_rf.mean()],
                     "표준편차 0.15 이상 승객 비율(%)": [(sd_dt >= .15).mean() * 100, (sd_rf >= .15).mean() * 100]})
stab.to_csv(TAB / "m5_stability.csv", index=False, encoding="utf-8-sig")
rc.to_csv(TAB / "m5_tree_top_splits.csv", header=["횟수"], encoding="utf-8-sig")
say(stab.round(3).to_string() + "\n트리 상위 3개 분기(뿌리 / 왼쪽 / 오른쪽) 빈도\n" + rc.head(8).to_string())
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
bins = np.linspace(0, max(sd_dt.max(), sd_rf.max()), 30)
ax[0].hist(sd_dt, bins=bins, alpha=.6, color=ORANGE, label=f"결정트리 (평균 {sd_dt.mean():.3f})")
ax[0].hist(sd_rf, bins=bins, alpha=.6, color=TEAL, label=f"랜덤 포레스트 (평균 {sd_rf.mean():.3f})")
ax[0].set_xlabel("같은 승객의 예측 확률이 부트스트랩 100회에서 흔들린 정도 (표준편차)")
ax[0].set_ylabel("승객 수")
ax[0].legend()
ax[0].set_title("학습 데이터가 조금 바뀌면 예측이 얼마나 흔들리나")
top = rc.head(6)[::-1]
ax[1].barh(range(len(top)), top.values, color=ORANGE)
for i, v_ in enumerate(top.values):
    ax[1].text(v_ + .5, i, f"{v_}회", va="center", fontsize=9)
ax[1].set_yticks(range(len(top)))
ax[1].set_yticklabels(top.index, fontsize=8)
ax[1].set_xlabel("100회 중 등장 횟수")
ax[1].set_title(f"결정트리 상위 3개 분기 조합 ({len(rc)}가지가 나옴)")
save(fig, "m5_stability.png")

(OUT / "_report_titanic.txt").write_text("\n".join(log), encoding="utf-8")
