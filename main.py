import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = (
    "https://raw.githubusercontent.com/igaeulbyeol835-commits/data/"
    "refs/heads/main/stroke.csv"
)
RECORD_FILE = Path(__file__).parent / "내_기록" / "기록.json"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ---------- 저장 / 불러오기 ----------
def load_record():
    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("meanings", {}), data.get("source", "")
        except (json.JSONDecodeError, OSError):
            pass
    return {}, ""


def save_record(meanings, source):
    RECORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump({"meanings": meanings, "source": source}, f,
                  ensure_ascii=False, indent=2)


# ---------- 데이터 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series):
    """열의 값 종류를 짧은 글로 설명한다."""
    values = series.dropna()
    n_unique = values.nunique()
    if pd.api.types.is_numeric_dtype(values) and n_unique > 10:
        return f"숫자 {values.min():g} ~ {values.max():g} ({n_unique:,}가지)"
    kinds = sorted(values.unique().tolist(), key=str)
    return ", ".join(str(k) for k in kinds)


try:
    df = load_data()
except Exception as e:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

# ---------- 저장된 기록을 세션에 한 번만 올리기 ----------
if "loaded" not in st.session_state:
    saved_meanings, saved_source = load_record()
    st.session_state["saved_meanings"] = saved_meanings
    st.session_state["source_text"] = saved_source
    st.session_state["base_table"] = pd.DataFrame({
        "열 이름": df.columns,
        "우리말 뜻": [saved_meanings.get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    })
    st.session_state["last_saved"] = (saved_meanings, saved_source)
    st.session_state["loaded"] = True

# ---------- 제목 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("먼저 우리가 다룰 데이터가 어떤 데이터인지 살펴봐요.")

# ---------- 큰 숫자 카드 ----------
st.markdown(
    """
    <style>
    .big-card {
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 16px;
        padding: 1.2rem 0.5rem;
        text-align: center;
        background: rgba(128,128,128,0.08);
    }
    .big-card .label { font-size: 1rem; opacity: 0.75; }
    .big-card .value { font-size: 2.8rem; font-weight: 800; line-height: 1.3; }
    </style>
    """,
    unsafe_allow_html=True,
)

total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

cards = [
    ("전체 사람 수", f"{total:,}명"),
    ("열 개수", f"{n_cols:,}개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}명"),
    ("stroke가 1인 비율", f"{ratio:.2f}%"),
]
for col, (label, value) in zip(st.columns(4), cards):
    col.markdown(
        f'<div class="big-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ---------- 열 설명 표 ----------
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장돼요.")

edited = st.data_editor(
    st.session_state["base_table"],
    key="meaning_editor",
    hide_index=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (직접 입력)"),
    },
)

# ---------- 처음 다섯 줄 ----------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5))

# ---------- 데이터 출처 ----------
st.subheader("데이터 출처")
st.text_area(
    "교재에 있는 출처 글을 적어 주세요. (입력 칸 밖을 누르면 저장돼요)",
    key="source_text",
    height=120,
)

# ---------- 변경되면 저장 ----------
current_meanings = {
    row["열 이름"]: (row["우리말 뜻"] or "").strip()
    for _, row in edited.iterrows()
    if (row["우리말 뜻"] or "").strip()
}
current_source = st.session_state["source_text"]

if (current_meanings, current_source) != st.session_state["last_saved"]:
    try:
        save_record(current_meanings, current_source)
        st.session_state["last_saved"] = (current_meanings, current_source)
        st.toast("내_기록 폴더에 저장했어요 ✅")
    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")
