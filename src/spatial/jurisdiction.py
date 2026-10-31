"""119안전센터 관할구역 텍스트를 센터별 동 목록으로 정리한다 (시스템 표시용, 학습 피처 아님)."""

import re

# '구로2동 중 387번지', '서초1동 중 다음번지 제외'처럼 동 일부만 맡는 표기. '중계동', '송중동'의 '중'은 제외
_PART_OF = re.compile(r"중(?=[\d산(다:]|$)")
_PARTIAL_WORDS = ("일부", "제외", "번지", "일대", "단지", "호)", "만해당")
_NAME = re.compile(
    r"^([가-힣]+?(?:본|\d+)(?:동|가)?(?:[ㆍ∼](?:본|\d+)(?:동|가)?)*(?<=[동가])(?=$|[^\dㆍ∼가동])"
    r"|[가-힣]+?(?:본동|동)(?=$|[^가-힣]))"
)


def _is_partial(text: str) -> bool:
    return any(w in text for w in _PARTIAL_WORDS) or bool(_PART_OF.search(text))


def _split_top(text: str) -> list[str]:
    """괄호 밖 쉼표로 나눈다."""
    out, buf, depth = [], "", 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    out.append(buf)
    return [t for t in out if t]


def _valid(dong: str) -> bool:
    return (
        bool(re.match(r"^[가-힣]", dong))
        and dong.endswith(("동", "가"))
        and not re.match(r"^산\d", dong)
    )


def expand_dong(name: str, prefix: str = "") -> list[str]:
    """'가양2ㆍ3동', '을지로3∼7가', '반포본ㆍ2동' 같은 묶음 표기를 동 이름 목록으로 펼친다."""
    name = name.replace("가동", "가")
    m = re.match(r"^([가-힣]*?)((?:본|\d+)(?:동|가)?(?:[ㆍ∼](?:본|\d+)(?:동|가)?)*)$", name)
    if not m:
        return [name]
    base = m.group(1) or prefix
    nums = m.group(2)
    suffix = "가" if nums.endswith("가") else "동"
    out = []
    for part in nums.split("ㆍ"):
        if "∼" in part:
            a, b = (x.rstrip("동가") for x in part.split("∼"))
            out += [f"{base}{i}{suffix}" for i in range(int(a), int(b) + 1)]
        else:
            p = part.rstrip("동가")
            out.append(f"{base}본동" if p == "본" else f"{base}{p}{suffix}")
    return out


def _parse(body: str, prefix: str = "") -> list[dict]:
    while body.startswith("(") and body.count("(") > body.count(")"):
        body = body[1:]
    rows = []
    for tok in _split_top(body):
        cut = min([i for i in (tok.find("("), tok.find("[")) if i >= 0], default=len(tok))
        head, inner = tok[:cut], tok[cut + 1 :]
        if inner.endswith((")", "]")):
            inner = inner[:-1]
        head_partial = _is_partial(head) or "관할구역제외" in inner
        clean = _PART_OF.split(head, maxsplit=1)[0].replace("일부", "").split(":")[0]
        m = _NAME.match(clean)
        if not m and not re.fullmatch(
            r"(?:본|\d+)(?:동|가)?(?:[ㆍ∼](?:본|\d+)(?:동|가)?)*(?:동|가)", clean
        ):
            continue
        name = m.group(1) if m else clean

        own_prefix = re.sub(r"(본|\d.*)?동$", "", name) if m else prefix
        subs = _parse(inner, own_prefix) if inner else []
        if subs:
            rows += [{**r, "parent": name, "partial": r["partial"] or head_partial} for r in subs]
            continue
        partial = head_partial or (bool(inner) and _is_partial(inner))
        rows += [
            {"dong": d, "parent": name, "partial": partial}
            for d in expand_dong(name, prefix)
            if _valid(d)
        ]
    return rows


def parse_jurisdiction(text: str) -> list[dict]:
    """관할구역 문장을 [{dong, parent, partial}] 목록으로 바꾼다. partial=True는 번지·'일부' 단위로 나눠 맡는 동."""
    body = text.split(":", 1)[-1]
    body = re.sub(r"\s+", "", body).replace("～", "∼").replace("~", "∼")
    body = re.sub(r"(\d(?:,\d)+)동", lambda m: m.group(1).replace(",", "ㆍ") + "동", body)
    body = re.sub(r"([)\]])(?=[가-힣])", r"\1,", body)
    return _parse(body)


def bdong_of(hdong: str) -> str:
    """행정동 이름에서 법정동 이름을 추정한다 (예: '도곡1동' → '도곡동')."""
    return re.sub(r"(본|\d+)동$", "동", hdong)


def find_centers(sgg: str, dongs: list[str], table: list[dict]) -> list[dict]:
    """신고 지점의 구·동 이름들로 관할 센터 후보를 찾는다. table 행: {center_id, sgg, dong, parent, partial}."""
    names = set(dongs) | {bdong_of(d) for d in dongs}
    hits = [r for r in table if r["sgg"] == sgg and r["dong"] in names]
    exact = [r for r in hits if r["dong"] in dongs]
    return exact or hits
