import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "탐색"
PAGE_ICON = "🔍"
DATA_URL = (
    "https://raw.githubusercontent.com/igaeulbyeol835-commits/data/"
    "refs/heads/main/stroke.csv"
)

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon=PAGE_ICON, layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def show_chart(fig):
    """streamlit 버전에 상관없이 그림을 화면 너비에 맞춰 보여 준다."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


try:
    df = load_data()
except Exception as e:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption("데이터를 여러 방향에서 살펴봐요.")

STROKE_LABEL = {0: "겪지 않음", 1: "겪음"}
YES_NO_LABEL = {0: "없음", 1: "있음"}
df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map(STROKE_LABEL)
group_order = ["겪지 않음", "겪음"]

# ---------- 1. 분포 (히스토그램) ----------
st.subheader("1. 나이와 평균 혈당의 분포")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포",
                       labels={"age": "나이", "count": "사람 수"})
    fig.update_layout(yaxis_title="사람 수")
    show_chart(fig)
with col2:
    fig = px.histogram(df, x="avg_glucose_level", nbins=40, title="평균 혈당 분포",
                       labels={"avg_glucose_level": "평균 혈당"})
    fig.update_layout(yaxis_title="사람 수")
    show_chart(fig)

# ---------- 2. 뇌졸중 여부별 비교 (상자그림) ----------
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
col1, col2 = st.columns(2)
with col1:
    fig = px.box(df, x="뇌졸중 여부", y="age", color="뇌졸중 여부",
                 category_orders={"뇌졸중 여부": group_order},
                 title="나이", labels={"age": "나이"})
    fig.update_layout(showlegend=False)
    show_chart(fig)
with col2:
    fig = px.box(df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
                 category_orders={"뇌졸중 여부": group_order},
                 title="평균 혈당", labels={"avg_glucose_level": "평균 혈당"})
    fig.update_layout(showlegend=False)
    show_chart(fig)

mean_table = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(group_order)
    .round(2)
    .reset_index()
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당 평균"})
)
st.markdown("**두 그룹의 평균값**")
st.dataframe(mean_table, hide_index=True)

# ---------- 3. 고혈압·심장병별 뇌졸중 비율 (막대그래프) ----------
st.subheader("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")


def stroke_rate_by(column):
    rate = df.groupby(column)["stroke"].mean().mul(100).round(2).reset_index()
    rate.columns = [column, "뇌졸중 비율(%)"]
    rate[column] = rate[column].map(YES_NO_LABEL)
    return rate


col1, col2 = st.columns(2)
for col, column, name in [(col1, "hypertension", "고혈압"),
                          (col2, "heart_disease", "심장병")]:
    rate = stroke_rate_by(column)
    fig = px.bar(rate, x=column, y="뇌졸중 비율(%)", text="뇌졸중 비율(%)",
                 title=f"{name} 여부별 뇌졸중 비율",
                 labels={column: f"{name} 여부"},
                 category_orders={column: ["없음", "있음"]})
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_yaxes(range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.25, 1)])
    with col:
        show_chart(fig)

# ---------- 4. bmi가 빈 사람들 ----------
st.subheader("4. bmi가 비어 있는 사람들의 뇌졸중 비율")
missing = df[df["bmi"].isna()]
rows = [
    ("bmi가 빈 사람", len(missing), int((missing["stroke"] == 1).sum())),
    ("전체", len(df), int((df["stroke"] == 1).sum())),
]
bmi_table = pd.DataFrame(rows, columns=["구분", "사람 수", "뇌졸중 1인 사람 수"])
bmi_table["뇌졸중 비율(%)"] = (
    bmi_table["뇌졸중 1인 사람 수"] / bmi_table["사람 수"] * 100
).round(2)
st.dataframe(bmi_table, hide_index=True)

# ---------- 5. 흡연 상태별 사람 수 ----------
st.subheader("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"].value_counts(dropna=False)
    .rename_axis("흡연 상태(smoking_status)")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True)
