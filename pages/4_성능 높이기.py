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

st.set_page_config(page_title="성능 높이기", layout="wide")

# ──────────────────────────────────────────────────────────────
# 설정: '채점표' 페이지와 같은 값
# ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "stroke"        # 1 = 뇌졸중, 0 = 아님
TEST_SIZE = 0.2
RANDOM_STATE = 42
TREE_MAX_DEPTH = 5

NAME_LR = "로지스틱 회귀(확률로 답하는 모델)"
NAME_DT = "의사결정트리(질문으로 답하는 모델)"
KIND = {NAME_LR: "lr", NAME_DT: "dt"}
BEFORE = "가중치를 맞추기 전"
AFTER = "가중치를 맞춘 뒤"
NA = "계산할 수 없음"

COLOR_BEFORE = "#1f77b4"   # 파랑
COLOR_AFTER = "#2ca02c"    # 초록
COLOR_NEG = "#9aa5b1"      # 회색
COLOR_POS = "#d62728"      # 빨강

DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LABELS = {
    "gender": "성별",
    "age": "나이",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
    "ever_married": "결혼 여부",
    "work_type": "직업 유형",
    "Residence_type": "거주 형태",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수(bmi)",
    "smoking_status": "흡연 상태",
}


def label(col):
    return LABELS.get(col, col)


# ──────────────────────────────────────────────────────────────
# 데이터 읽기 · 나누기 · 학습 ('채점표'와 같은 방법)
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
    """bmi가 비어 있으면 훈련용 중앙값으로 채우고, 문자 열은 숫자 열로 바꾼다."""
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
    return train_raw, test_raw, train_enc, test_enc


def make_model(kind, weighted):
    class_weight = "balanced" if weighted else None
    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight=class_weight),
        )
    return DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )


@st.cache_resource
def train_all(features):
    df = load_data()
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns="id")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # 속성을 고르기 전의 전체 표로 나누므로, 어떤 속성을 골라도 같은 사람들이 테스트용이 된다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    cols = list(features)
    X_train = X_train[cols]
    X_test = X_test[cols]

    train_raw, test_raw, train_enc, test_enc = encode(X_train, X_test)

    models = {}
    for name, kind in KIND.items():
        for weighted in (False, True):
            model = make_model(kind, weighted)
            model.fit(train_enc, y_train)
            models[(name, weighted)] = model

    return {
        "models": models,
        "test_raw": test_raw.reset_index(drop=True),
        "test_enc": test_enc.reset_index(drop=True),
        "y_true": y_test.to_numpy(),
        "medians": train_enc.median(),
        "columns": list(train_enc.columns),
        "n_train_neg": int((y_train == 0).sum()),
        "n_train_pos": int((y_train == 1).sum()),
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


def table_column(cell_tuple):
    tp, fp, fn, tn = cell_tuple
    accuracy, recall, precision, f1 = calc_metrics(tp, fp, fn, tn)
    return [
        f"{tp}명", f"{fp}명", f"{fn}명", f"{tn}명",
        fmt(accuracy), fmt(recall), fmt(precision), fmt(f1),
    ]


ROW_LABELS = [
    "찾아낸 환자 (TP)",
    "헛짚은 사람 (FP)",
    "놓친 환자 (FN)",
    "제대로 아니라고 한 사람 (TN)",
    "정확도",
    "재현율",
    "정밀도",
    "F1",
]


def axis_range(values):
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo if hi > lo else 1.0
    pad = span * 0.08
    return lo - pad, hi + pad


def jitter(values, rng):
    """값이 두세 가지뿐인 축은 점이 겹치지 않게 살짝 흩뜨려 그린다(그림에서만)."""
    values = np.asarray(values, dtype=float)
    if len(np.unique(values)) <= 2:
        return values + rng.uniform(-0.05, 0.05, len(values))
    return values


def grid_proba(model, columns, medians, coords):
    """고른 축 외의 속성은 훈련용 중앙값으로 고정하고, 뇌졸중일 확률을 계산한다."""
    n = len(next(iter(coords.values())))
    frame = pd.DataFrame({c: np.full(n, medians[c]) for c in columns})
    for col, values in coords.items():
        frame[col] = values
    return model.predict_proba(frame)[:, 1]


def outside_note(prob, name, place):
    """0.5 경계선이 그림 밖에 있으면 그 사실을 글로 돌려준다."""
    if (prob >= 0.5).any() and (prob < 0.5).any():
        return None
    if (prob >= 0.5).all():
        side = "0.5 이상입니다."
    else:
        side = "0.5보다 낮습니다."
    return (
        f"{name}의 0.5 경계선은 {place} 밖에 있습니다. "
        f"{place} 안 모든 곳에서 뇌졸중일 확률이 {side}"
    )


# ──────────────────────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────────────────────
st.title("성능 높이기")

data = load_data()
if data is None:
    st.error(
        f"'{TARGET}' 열이 들어 있는 CSV 파일을 찾지 못했습니다. "
        "저장소 맨 위 폴더나 data 폴더에 데이터 파일이 있는지 확인해 주세요."
    )
    st.stop()

options = [c for c in data.columns if c not in ("id", TARGET)]
defaults = [c for c in DEFAULT_FEATURES if c in options]
selected = st.multiselect(
    "입력으로 사용할 속성을 고르세요",
    options=options,
    default=defaults,
    format_func=label,
)
if not selected:
    st.warning("속성을 하나 이상 골라 주세요.")
    st.stop()
features = tuple(c for c in options if c in selected)

state = train_all(features)
models = state["models"]
test_raw = state["test_raw"]
test_enc = state["test_enc"]
y_true = state["y_true"]
medians = state["medians"]
columns = state["columns"]

n_test = len(y_true)
n_test_pos = int((y_true == 1).sum())
n0, n1 = state["n_train_neg"], state["n_train_pos"]
n_all = n0 + n1
st.markdown(f"테스트용 **{n_test}명** 중 뇌졸중인 사람은 **{n_test_pos}명**입니다.")
st.caption(
    f"훈련용에는 뇌졸중이 아닌 사람 {n0}명, 뇌졸중인 사람 {n1}명이 있어 뇌졸중이 아주 드뭅니다. "
    f"'가중치를 맞춘 뒤'는 두 쪽의 영향이 같아지도록 뇌졸중이 아닌 사람에게 "
    f"{n_all / (2 * n0):.2f}, 뇌졸중인 사람에게 {n_all / (2 * n1):.2f}의 가중치를 주고 다시 학습한 결과입니다."
)

# 1. 전·후 비교표
st.subheader("가중치를 맞추기 전과 뒤")
cells = {}
for key, model in models.items():
    cells[key] = count_cells(y_true, model.predict(test_enc).astype(int))

col_left, col_right = st.columns(2)
for col, name in zip([col_left, col_right], [NAME_LR, NAME_DT]):
    before = cells[(name, False)]
    after = cells[(name, True)]
    with col:
        st.markdown(f"**{name}**")
        st.table(
            pd.DataFrame(
                {BEFORE: table_column(before), AFTER: table_column(after)},
                index=ROW_LABELS,
            )
        )
        st.markdown(
            f"찾아낸 환자 {before[0]}명 → {after[0]}명, "
            f"놓친 환자 {before[2]}명 → {after[2]}명, "
            f"헛짚은 사람 {before[1]}명 → {after[1]}명"
        )

# 2. 경계선 그림
st.subheader("0.5 경계선 비교")
plot_model = st.radio("어느 모델의 경계선을 볼까요?", [NAME_LR, NAME_DT], horizontal=True)
st.caption(
    "경계선은 뇌졸중일 확률이 0.5인 곳입니다. 그림에 고르지 않은 나머지 속성은 "
    "훈련용의 가운데 값(중앙값)으로 고정하고 그렸습니다."
)

numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(test_raw[c])]
model_before = models[(plot_model, False)]
model_after = models[(plot_model, True)]
rng = np.random.default_rng(0)
neg_mask = y_true == 0
pos_mask = y_true == 1

# 2-1. 2차원
st.markdown("#### 2차원 산점도")
if len(numeric_cols) < 2:
    st.info("숫자로 된 속성을 둘 이상 골라야 2차원 그림을 그릴 수 있습니다.")
else:
    ax_left, ax_right = st.columns(2)
    x_col = ax_left.selectbox("가로축", numeric_cols, index=0, format_func=label, key="x2")
    y_col = ax_right.selectbox("세로축", numeric_cols, index=1, format_func=label, key="y2")

    if x_col == y_col:
        st.warning("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
    else:
        x_vals = test_raw[x_col].to_numpy(dtype=float)
        y_vals = test_raw[y_col].to_numpy(dtype=float)
        x_lo, x_hi = axis_range(x_vals)
        y_lo, y_hi = axis_range(y_vals)
        xs = np.linspace(x_lo, x_hi, 150)
        ys = np.linspace(y_lo, y_hi, 150)
        XX, YY = np.meshgrid(xs, ys)

        probs = {}
        for tag, model in [(BEFORE, model_before), (AFTER, model_after)]:
            p = grid_proba(model, columns, medians, {x_col: XX.ravel(), y_col: YY.ravel()})
            probs[tag] = p.reshape(XX.shape)

        fig = go.Figure()
        for mask, name, color, size, opacity in [
            (neg_mask, f"실제 뇌졸중 아님 ({int(neg_mask.sum())}명)", COLOR_NEG, 6, 0.45),
            (pos_mask, f"실제 뇌졸중 ({int(pos_mask.sum())}명)", COLOR_POS, 9, 0.9),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=jitter(x_vals[mask], rng),
                    y=jitter(y_vals[mask], rng),
                    mode="markers",
                    name=name,
                    marker=dict(size=size, color=color, opacity=opacity,
                                line=dict(width=0.5, color="white")),
                    customdata=np.column_stack([x_vals[mask], y_vals[mask]]),
                    hovertemplate=(
                        f"{label(x_col)}: %{{customdata[0]:g}}<br>"
                        f"{label(y_col)}: %{{customdata[1]:g}}<extra></extra>"
                    ),
                )
            )
        for tag, color in [(BEFORE, COLOR_BEFORE), (AFTER, COLOR_AFTER)]:
            fig.add_trace(
                go.Contour(
                    x=xs, y=ys, z=probs[tag],
                    contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                    colorscale=[[0, color], [1, color]],
                    line=dict(width=3, color=color),
                    showscale=False, showlegend=False, hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(color=color, width=3),
                    name=f"0.5 경계선 ({tag})",
                )
            )
        fig.update_layout(
            height=560,
            xaxis=dict(title=label(x_col), range=[x_lo, x_hi]),
            yaxis=dict(title=label(y_col), range=[y_lo, y_hi]),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig)

        notes = [
            outside_note(probs[tag], f"{tag}", "그림")
            for tag in (BEFORE, AFTER)
        ]
        for note in notes:
            if note:
                st.markdown(note)
        if any(len(np.unique(v)) <= 2 for v in (x_vals, y_vals)):
            st.caption("값이 두세 가지뿐인 축은 점이 겹치지 않도록 점만 살짝 흩뜨려 그렸습니다. 경계선은 그대로입니다.")

# 2-2. 3차원
if len(selected) >= 3:
    st.markdown("#### 3차원 산점도 (마우스로 돌려 볼 수 있습니다)")
    if len(numeric_cols) < 3:
        st.info("숫자로 된 속성을 셋 이상 골라야 3차원 그림을 그릴 수 있습니다.")
    else:
        a_col, b_col, c_col = st.columns(3)
        x3 = a_col.selectbox("가로축(x)", numeric_cols, index=0, format_func=label, key="x3")
        y3 = b_col.selectbox("세로축(y)", numeric_cols, index=1, format_func=label, key="y3")
        z3 = c_col.selectbox("깊이축(z)", numeric_cols, index=2, format_func=label, key="z3")

        if len({x3, y3, z3}) < 3:
            st.warning("세 축은 서로 다른 속성으로 골라 주세요.")
        else:
            vx = test_raw[x3].to_numpy(dtype=float)
            vy = test_raw[y3].to_numpy(dtype=float)
            vz = test_raw[z3].to_numpy(dtype=float)
            rx = axis_range(vx)
            ry = axis_range(vy)
            rz = axis_range(vz)
            n_grid = 22
            gx, gy, gz = np.meshgrid(
                np.linspace(*rx, n_grid),
                np.linspace(*ry, n_grid),
                np.linspace(*rz, n_grid),
                indexing="ij",
            )
            coords = {x3: gx.ravel(), y3: gy.ravel(), z3: gz.ravel()}
            probs3 = {
                BEFORE: grid_proba(model_before, columns, medians, coords),
                AFTER: grid_proba(model_after, columns, medians, coords),
            }

            fig3 = go.Figure()
            for mask, name, color, size, opacity in [
                (neg_mask, f"실제 뇌졸중 아님 ({int(neg_mask.sum())}명)", COLOR_NEG, 3, 0.35),
                (pos_mask, f"실제 뇌졸중 ({int(pos_mask.sum())}명)", COLOR_POS, 5, 0.95),
            ]:
                fig3.add_trace(
                    go.Scatter3d(
                        x=jitter(vx[mask], rng),
                        y=jitter(vy[mask], rng),
                        z=jitter(vz[mask], rng),
                        mode="markers",
                        name=name,
                        marker=dict(size=size, color=color, opacity=opacity),
                        customdata=np.column_stack([vx[mask], vy[mask], vz[mask]]),
                        hovertemplate=(
                            f"{label(x3)}: %{{customdata[0]:g}}<br>"
                            f"{label(y3)}: %{{customdata[1]:g}}<br>"
                            f"{label(z3)}: %{{customdata[2]:g}}<extra></extra>"
                        ),
                    )
                )
            for tag, color in [(BEFORE, COLOR_BEFORE), (AFTER, COLOR_AFTER)]:
                fig3.add_trace(
                    go.Isosurface(
                        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(),
                        value=probs3[tag],
                        isomin=0.5, isomax=0.5, surface_count=1,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        colorscale=[[0, color], [1, color]],
                        showscale=False, showlegend=False,
                        opacity=0.45, hoverinfo="skip",
                    )
                )
                fig3.add_trace(
                    go.Scatter3d(
                        x=[None], y=[None], z=[None], mode="markers",
                        marker=dict(size=7, color=color, symbol="square"),
                        name=f"0.5 경계 평면 ({tag})",
                    )
                )
            fig3.update_layout(
                height=680,
                scene=dict(
                    xaxis=dict(title=label(x3), range=list(rx)),
                    yaxis=dict(title=label(y3), range=list(ry)),
                    zaxis=dict(title=label(z3), range=list(rz)),
                    aspectmode="cube",
                ),
                legend=dict(orientation="h", y=1.05),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig3)

            for tag in (BEFORE, AFTER):
                note = outside_note(probs3[tag], tag, "그림 상자")
                if note:
                    st.markdown(note)
