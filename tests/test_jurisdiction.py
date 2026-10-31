from src.spatial.jurisdiction import bdong_of, expand_dong, find_centers, parse_jurisdiction


def dongs(text):
    return [r["dong"] for r in parse_jurisdiction(text)]


def test_expand_dong():
    assert expand_dong("가양2ㆍ3동") == ["가양2동", "가양3동"]
    assert expand_dong("을지로3∼5가") == ["을지로3가", "을지로4가", "을지로5가"]
    assert expand_dong("반포본ㆍ2동") == ["반포본동", "반포2동"]
    assert expand_dong("4ㆍ5동", "면목") == ["면목4동", "면목5동"]
    assert expand_dong("청담동") == ["청담동"]


def test_parse_plain_and_hdong():
    assert dongs("강남구중 : 삼성동(삼성 1ㆍ2동), 청담동") == ["삼성1동", "삼성2동", "청담동"]
    assert dongs("관악구중 : 신림동(서원동, 신원동)") == ["서원동", "신원동"]
    assert dongs("강북구중 : 번1,2,3동, 수유3동") == ["번1동", "번2동", "번3동", "수유3동"]


def test_parse_partial():
    rows = parse_jurisdiction("강동구중 : 강일동, 상일동 일부 (1～84, 121～122)")
    assert rows[0] == {"dong": "강일동", "parent": "강일동", "partial": False}
    assert rows[1]["dong"] == "상일동" and rows[1]["partial"]


def test_find_centers():
    table = [
        {
            "center_id": "A",
            "sgg": "강남구",
            "dong": "도곡1동",
            "parent": "도곡동",
            "partial": False,
        },
        {"center_id": "B", "sgg": "강남구", "dong": "청담동", "parent": "청담동", "partial": False},
        {"center_id": "C", "sgg": "관악구", "dong": "청담동", "parent": "청담동", "partial": False},
    ]
    assert [r["center_id"] for r in find_centers("강남구", ["도곡1동"], table)] == ["A"]
    assert [r["center_id"] for r in find_centers("강남구", ["청담동"], table)] == ["B"]
    assert bdong_of("일원본동") == "일원동"


def test_nested_and_false_partial():
    rows = parse_jurisdiction("송파구중 : 잠실동[잠실3동(잠실주공5단지아파트 제외), 잠실4ㆍ6동]")
    assert [(r["dong"], r["partial"]) for r in rows] == [
        ("잠실3동", True),
        ("잠실4동", False),
        ("잠실6동", False),
    ]
    assert [r["partial"] for r in parse_jurisdiction("노원구중 : 중계1ㆍ4동, 송중동")] == [
        False,
        False,
        False,
    ]
    assert dongs("강서구중 : 가양1동 일부(524∼526) 공항동 일부 (4∼7)") == ["가양1동", "공항동"]
