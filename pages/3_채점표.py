import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="채점표", layout="wide")

# ──────────────────────────────────────────────────────────────
# 설정: '분류 모델' 페이지와 반드시 같은 값으로 맞춰야 하는 부분
# ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "stroke"        # 1 = 뇌졸중, 0 = 아님
TEST_SIZE = 0.2
RANDOM_STATE = 42
TREE_MAX_DEPTH = 5

NAME_LR = "로지스틱 회귀(확률로 답하는 모델)"
NAME_DT = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "다수 클래스 분류기(훈련용에서 많은 쪽으로만 답하는 모델)"
NA = "계산할 수 없음"
MAX_ROWS = 20


# ──────────────────────────────────────────────────────────────
# 데이터 읽기 · 나누기 · 학습
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for folder in [ROOT, os.path.join(ROOT, "data")]:
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".csv"):
                df = pd.read_csv(os.path.join(folder, name))
                if TARGET in df.columns:
                    return df
    return None


def encode(train_raw, test_raw):
    """훈련용 기준으로 결측을 채우고 문자 열을 숫자 열로 바꾼다."""
    train_raw = train_raw.copy()
    test_raw = test_raw.copy()
    if "bmi" in train_raw.columns:
        median = train_raw["bmi"].median()
        train_raw["bmi"] = train_raw["bmi"].fillna(median)
        test_raw["bmi"] = test_raw["bmi"].fillna(median)
    train_enc = pd.get_dummies(train_raw).astype(float)
    test_enc = (
        pd.get_dummies(test_raw)
        .astype(float)
        .reindex(columns=train_enc.columns, fill_value=0.0)
    )
    return train_enc, test_enc


@st.cache_resource
def run_all():
    df = load_data()
    if df is None:
        return None

    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns="id")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train_enc, X_test_enc = encode(X_train, X_test)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(X_train_enc, y_train)

    tree = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE)
    tree.fit(X_train_enc, y_train)

    majority = int(y_train.value_counts().idxmax())

    y_true = y_test.to_numpy()
    preds = {
        NAME_LR: lr.predict(X_test_enc).astype(int),
        NAME_DT: tree.predict(X_test_enc).astype(int),
        NAME_BASE: np.full(len(y_test), majority, dtype=int),
    }
    return {
        "X_test_raw": X_test.reset_index(drop=True),
        "y_true": y_true,
        "preds": preds,
    }


# ──────────────────────────────────────────────────────────────
# 계산 도우미
# ──────────────────────────────────────────────────────────────
def count_cells(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def calc_metrics(tp, fp, fn, tn):
    """계산할 수 없는 값은 None으로 돌려준다."""
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    actual_pos = tp + fn      # 재현율의 분모: 실제 뇌졸중인 사람 수
    predicted_pos = tp + fp   # 정밀도의 분모: 뇌졸중이라 예측한 사람 수
    recall = tp / actual_pos if actual_pos else None
    precision = tp / predicted_pos if predicted_pos else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def fmt(value):
    return NA if value is None else f"{value:.3f}"


def matrix_figure(tp, fp, fn, tn):
    """행은 실제, 열은 예측. 네 칸에 약자와 사람 수, 바깥에 합계."""
    x_labels = ["예측: 뇌졸중", "예측: 아님", "합계"]
    y_labels = ["실제: 뇌졸중", "실제: 아님", "합계"]
    z = [
        [tp, fn, np.nan],
        [fp, tn, np.nan],
        [np.nan, np.nan, np.nan],
    ]
    texts = [
        [f"<b>TP</b><br>{tp}명", f"<b>FN</b><br>{fn}명", f"{tp + fn}명"],
        [f"<b>FP</b><br>{fp}명", f"<b>TN</b><br>{tn}명", f"{fp + tn}명"],
        [f"{tp + fp}명", f"{fn + tn}명", f"{tp + fp + fn + tn}명"],
    ]
    z_max = max(tp, fp, fn, tn, 1)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            zmin=0,
            zmax=z_max,
            colorscale=[[0, "#e8f0fe"], [1, "#1a56b0"]],
            showscale=False,
            xgap=4,
            ygap=4,
            hoverinfo="skip",
        )
    )
    for i, y_label in enumerate(y_labels):
        for j, x_label in enumerate(x_labels):
            value = z[i][j]
            font = {"size": 16}
            if not np.isnan(value):
                font["color"] = "#ffffff" if value / z_max > 0.55 else "#111111"
            fig.add_annotation(
                x=x_label,
                y=y_label,
                text=texts[i][j],
                showarrow=False,
                font=font,
            )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(side="top", fixedrange=True),
        yaxis=dict(autorange="reversed", fixedrange=True),
    )
    return fig


def make_list(sub_df):
    """나이 많은 순으로 정렬하고 앞 MAX_ROWS줄만 표로 만든다."""
    sub_df = sub_df.sort_values("age", ascending=False, kind="stable")
    shown = sub_df.head(MAX_ROWS)
    return pd.DataFrame(
        {
            "번호": range(1, len(shown) + 1),
            "나이": [f"{a:g}" for a in shown["age"]],
            "평균 혈당": [f"{g:.1f}" for g in shown["avg_glucose_level"]],
            "고혈압 여부": ["예" if h == 1 else "아니오" for h in shown["hypertension"]],
            "심장병 여부": ["예" if h == 1 else "아니오" for h in shown["heart_disease"]],
        }
    )


def list_title(label, n):
    title = f"#### {label} (전체 {n}명)"
    if n > MAX_ROWS:
        title += f" · 앞 {MAX_ROWS}명만 표시"
    return title


# ──────────────────────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────────────────────
st.title("채점표")

result = run_all()
if result is None:
    st.error(
        f"'{TARGET}' 열이 들어 있는 CSV 파일을 찾지 못했습니다. "
        "저장소 맨 위 폴더나 data 폴더에 데이터 파일이 있는지 확인해 주세요."
    )
    st.stop()

X_test_raw = result["X_test_raw"]
y_true = result["y_true"]
preds = result["preds"]

n_test = len(y_true)
n_test_pos = int((y_true == 1).sum())
st.markdown(f"테스트용 **{n_test}명** 중 뇌졸중인 사람은 **{n_test_pos}명**입니다.")

cells = {name: count_cells(y_true, pred) for name, pred in preds.items()}

# 1. 혼동행렬
st.subheader("혼동행렬 (행: 실제, 열: 예측)")
col_left, col_right = st.columns(2)
for col, name in zip([col_left, col_right], [NAME_LR, NAME_DT]):
    tp, fp, fn, tn = cells[name]
    with col:
        st.markdown(f"**{name}**")
        st.plotly_chart(matrix_figure(tp, fp, fn, tn))

# 2. 지표 비교표
st.subheader("네 가지 지표 비교")
rows = []
for name in [NAME_LR, NAME_DT, NAME_BASE]:
    accuracy, recall, precision, f1 = calc_metrics(*cells[name])
    rows.append(
        {
            "모델": name,
            "정확도": fmt(accuracy),
            "재현율": fmt(recall),
            "정밀도": fmt(precision),
            "F1": fmt(f1),
        }
    )
st.table(pd.DataFrame(rows).set_index("모델"))
st.caption(
    "정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ (TP+FN, 실제 뇌졸중인 사람 수) · "
    "정밀도 = TP ÷ (TP+FP, 뇌졸중이라 예측한 사람 수) · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)

# 3. 틀린 사람 목록
st.subheader("모델이 틀린 사람들")
choice = st.radio("어느 모델의 목록을 볼까요?", [NAME_LR, NAME_DT], horizontal=True)

pred = preds[choice]
missed = X_test_raw[(y_true == 1) & (pred == 0)]
wrong = X_test_raw[(y_true == 0) & (pred == 1)]
_, fp_cell, fn_cell, _ = cells[choice]

list_left, list_right = st.columns(2)
with list_left:
    st.markdown(list_title("놓친 환자 — 실제 뇌졸중인데 아니라고 예측한 사람", len(missed)))
    if len(missed):
        st.dataframe(make_list(missed), hide_index=True)
    else:
        st.info("해당하는 사람이 없습니다.")
with list_right:
    st.markdown(list_title("헛짚은 사람 — 아닌데 뇌졸중이라고 예측한 사람", len(wrong)))
    if len(wrong):
        st.dataframe(make_list(wrong), hide_index=True)
    else:
        st.info("해당하는 사람이 없습니다.")

same = len(missed) == fn_cell and len(wrong) == fp_cell
st.markdown(
    f"놓친 환자 {len(missed)}명은 혼동행렬 FN 칸({fn_cell}명), "
    f"헛짚은 사람 {len(wrong)}명은 FP 칸({fp_cell}명)과 "
    f"{'같습니다.' if same else '다릅니다.'}"
)
