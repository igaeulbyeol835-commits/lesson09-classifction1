import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "분류 모델"
PAGE_ICON = "🤖"
DATA_URL = (
    "https://raw.githubusercontent.com/igaeulbyeol835-commits/data/"
    "refs/heads/main/stroke.csv"
)

# 열 이름 → 우리말 이름
FEATURES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
DT_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델"

NEG_COLOR = "#1f77b4"   # 아님
POS_COLOR = "#d62728"   # 뇌졸중(양성)
NEG_LIGHT = "rgba(31,119,180,0.18)"
POS_LIGHT = "rgba(214,39,40,0.18)"

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


def fmt_value(feature, value):
    text = f"{value:g}"
    if feature in BINARY and value in (0, 1):
        text += "(있음)" if value == 1 else "(없음)"
    return text


# ---------- 제목 ----------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption("stroke가 1이면 뇌졸중, 0이면 아님이에요. 뇌졸중을 양성으로 둬요.")

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

# ---------- 1. 입력 속성 고르기 ----------
st.subheader("1. 입력으로 사용할 속성 고르기")
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=list(FEATURES.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda c: FEATURES[c],
)
selected = [c for c in FEATURES if c in chosen]  # 항상 같은 순서로
if len(selected) < 2:
    st.info("속성을 두 개 이상 골라 주세요. 그림을 그리려면 가로축과 세로축이 필요해요.")
    st.stop()
names = [FEATURES[c] for c in selected]

# ---------- 2. 데이터 나누기 ----------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3  # 열 명씩 묶어 앞 세 명은 테스트용
train_df, test_df = df[~is_test], df[is_test]

X_train = train_df[selected].copy()
X_test = test_df[selected].copy()
y_train = train_df["stroke"].to_numpy()
y_test = test_df["stroke"].to_numpy()

st.subheader("2. 데이터 나누기")
st.write(
    f"번호(id) 순으로 정렬한 뒤 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"테스트용은 **{len(test_df):,}명**, 나머지 일곱 명으로 이루어진 훈련용은 **{len(train_df):,}명**이에요."
)
if "bmi" in selected:
    bmi_fill = float(X_train["bmi"].median())
    n_tr = int(X_train["bmi"].isna().sum())
    n_te = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_fill)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_fill)
    st.write(
        f"체질량지수는 비어 있는 사람이 있어서 훈련용의 중앙값 **{bmi_fill:.2f}**로 채웠어요. "
        f"(훈련용 {n_tr:,}명, 테스트용 {n_te:,}명)"
    )
st.caption(
    "크기 맞추기(표준화)는 훈련용 사람들로만 계산해서 로지스틱 회귀에 써요. "
    "의사결정트리는 원래 값을 그대로 써서 질문을 읽기 쉽게 했어요."
)

# ---------- 모델 만들기 ----------
Xtr = X_train.to_numpy(dtype=float)
Xte = X_test.to_numpy(dtype=float)

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

lr = LogisticRegression(max_iter=1000).fit(Xtr_s, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(Xtr, y_train)
base = DummyClassifier(strategy="most_frequent").fit(Xtr, y_train)

acc = {
    LR_NAME: (accuracy_score(y_train, lr.predict(Xtr_s)), accuracy_score(y_test, lr.predict(Xte_s))),
    DT_NAME: (accuracy_score(y_train, tree.predict(Xtr)), accuracy_score(y_test, tree.predict(Xte))),
    BASE_NAME: (accuracy_score(y_train, base.predict(Xtr)), accuracy_score(y_test, base.predict(Xte))),
}

# ---------- 3. 정확도 카드 ----------
st.subheader("3. 모델의 정확도")
st.markdown(
    """
    <style>
    .acc-card {
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 16px;
        padding: 1rem 0.8rem;
        text-align: center;
        background: rgba(128,128,128,0.08);
        height: 100%;
    }
    .acc-card .name { font-size: 0.95rem; font-weight: 700; min-height: 3.2rem; }
    .acc-card .label { font-size: 0.85rem; opacity: 0.7; }
    .acc-card .value { font-size: 2.8rem; font-weight: 800; line-height: 1.3; }
    .acc-card .sub { display: flex; justify-content: space-around; font-size: 0.8rem;
                     opacity: 0.8; margin-top: 0.4rem; gap: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
for col, (name, (tr_acc, te_acc)) in zip(st.columns(3), acc.items()):
    col.markdown(
        f"""
        <div class="acc-card">
          <div class="name">{name}</div>
          <div class="label">테스트 데이터 정확도</div>
          <div class="value">{te_acc * 100:.2f}%</div>
          <div class="sub">
            <span>훈련 데이터 {tr_acc * 100:.2f}%</span>
            <span>테스트 데이터 {te_acc * 100:.2f}%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.caption("입력을 보지 않는 모델보다 정확도가 높아야 입력 속성이 도움이 된다고 볼 수 있어요.")

# ---------- 4. 그림으로 보기 ----------
st.subheader("4. 그림으로 보기")
c1, c2 = st.columns(2)
with c1:
    x_feat = st.selectbox("가로축", options=selected, index=0, format_func=lambda c: FEATURES[c])
with c2:
    y_options = [c for c in selected if c != x_feat]
    y_feat = st.selectbox("세로축", options=y_options, index=0, format_func=lambda c: FEATURES[c])

ia, ib = selected.index(x_feat), selected.index(y_feat)
x_name, y_name = FEATURES[x_feat], FEATURES[y_feat]
test_median = np.median(Xte, axis=0)  # 테스트 데이터의 중앙값(체질량지수는 채운 뒤의 값)


def axis_range(feature, values):
    if feature in BINARY:
        return -0.3, 1.3
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


a_lo, a_hi = axis_range(x_feat, Xte[:, ia])
b_lo, b_hi = axis_range(y_feat, Xte[:, ib])

fig = go.Figure()

# (1) 의사결정트리가 나눈 칸 (옅은 색)
xs = np.linspace(a_lo, a_hi, 200)
ys = np.linspace(b_lo, b_hi, 200)
gx, gy = np.meshgrid(xs, ys)
grid = np.tile(test_median, (gx.size, 1))
grid[:, ia], grid[:, ib] = gx.ravel(), gy.ravel()
region = tree.predict(grid).reshape(gx.shape)
fig.add_trace(go.Heatmap(
    x=xs, y=ys, z=region, zmin=0, zmax=1, showscale=False, hoverinfo="skip",
    colorscale=[[0, NEG_LIGHT], [1, POS_LIGHT]],
))
present = set(np.unique(region).tolist())
for cls, color, label in [(0, NEG_LIGHT, "아님"), (1, POS_LIGHT, "뇌졸중")]:
    if cls in present:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14, color=color),
            name=f"의사결정트리가 '{label}'이라 답하는 칸",
        ))

# (2) 테스트 데이터 점 (이진 속성은 겹치지 않게 살짝 흩뿌려 그림)
rng = np.random.default_rng(0)
jx = rng.uniform(-0.08, 0.08, len(Xte)) if x_feat in BINARY else np.zeros(len(Xte))
jy = rng.uniform(-0.08, 0.08, len(Xte)) if y_feat in BINARY else np.zeros(len(Xte))
for cls, color, label in [(0, NEG_COLOR, "아님"), (1, POS_COLOR, "뇌졸중")]:
    m = y_test == cls
    fig.add_trace(go.Scatter(
        x=Xte[m, ia] + jx[m], y=Xte[m, ib] + jy[m], mode="markers",
        marker=dict(size=7 if cls == 0 else 9, color=color,
                    opacity=0.55 if cls == 0 else 0.95,
                    line=dict(width=0.5, color="white")),
        name=f"실제: {label}",
        customdata=np.column_stack([Xte[m, ia], Xte[m, ib]]),
        hovertemplate=f"{x_name}: %{{customdata[0]:g}}<br>{y_name}: %{{customdata[1]:g}}"
                      "<extra>%{fullData.name}</extra>",
    ))

# (3) 로지스틱 회귀가 확률 0.5로 가르는 선
coef, intercept = lr.coef_[0], float(lr.intercept_[0])
mean, scale = scaler.mean_, scaler.scale_
u = coef[ia] / scale[ia]
v = coef[ib] / scale[ib]
k = intercept - u * mean[ia] - v * mean[ib]
for j in range(len(selected)):
    if j not in (ia, ib):
        k += coef[j] * (test_median[j] - mean[j]) / scale[j]

corner_z = [u * a + v * b + k for a in (a_lo, a_hi) for b in (b_lo, b_hi)]
line_inside = min(corner_z) < 0 < max(corner_z)
if line_inside:
    if abs(v) * (b_hi - b_lo) >= abs(u) * (a_hi - a_lo):
        la = np.linspace(a_lo, a_hi, 400)
        lb = -(u * la + k) / v
    else:
        lb = np.linspace(b_lo, b_hi, 400)
        la = -(v * lb + k) / u
    ok = (la >= a_lo) & (la <= a_hi) & (lb >= b_lo) & (lb <= b_hi)
    fig.add_trace(go.Scatter(
        x=np.where(ok, la, np.nan), y=np.where(ok, lb, np.nan), mode="lines",
        line=dict(color="black", width=3, dash="dash"),
        name="로지스틱 회귀(확률로 답하는 모델)의 0.5 선", hoverinfo="skip",
    ))

fig.update_layout(
    height=600,
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
)
x_axis = dict(title=x_name, range=[a_lo, a_hi])
y_axis = dict(title=y_name, range=[b_lo, b_hi])
if x_feat in BINARY:
    x_axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
if y_feat in BINARY:
    y_axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_xaxes(**x_axis)
fig.update_yaxes(**y_axis)
show_chart(fig)

others = [j for j in range(len(selected)) if j not in (ia, ib)]
if others:
    fixed = ", ".join(f"{names[j]} = {fmt_value(selected[j], float(test_median[j]))}" for j in others)
    st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed}**")
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둘 속성은 없어요.")

if line_inside:
    st.write(f"{LR_NAME}의 0.5 선(검은 점선)이 그림 안에 있어요.")
else:
    side = "0.5보다 작아서 모두 '아님'" if max(corner_z) < 0 else "0.5보다 커서 모두 '뇌졸중'"
    st.write(
        f"{LR_NAME}의 0.5 선은 이 그림 밖에 있어요. "
        f"그림 안에서는 어디서나 뇌졸중일 확률이 {side}이라고 답해요."
    )
st.caption("고혈압·심장병 축은 점이 겹치지 않도록 살짝 흩어 찍었어요. 칸의 색과 선은 실제 값으로 계산했어요.")

# ---------- 5. 의사결정트리가 던진 질문 ----------
st.subheader("5. 의사결정트리가 던진 질문")
st.caption("질문은 세 번까지만 던지고, 마지막 마디에 사람이 다섯 명 미만이 되게 나누지는 않아요. 난수는 42로 고정했어요.")

t = tree.tree_
path = tree.decision_path(Xtr)
n_people = np.asarray(path.sum(axis=0)).ravel()
n_stroke = np.asarray(path.T.dot(y_train)).ravel()


def question_text(feature, threshold):
    if feature in BINARY:
        return f"{FEATURES[feature]}이 없나요?"
    return f"{FEATURES[feature]} ≤ {round(float(threshold), 2):g} ?"


lines = [
    "digraph Tree {",
    '  graph [rankdir=TB, bgcolor="transparent"];',
    '  node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=12, '
    'color="#888888", fontcolor="#222222"];',
    '  edge [fontname="sans-serif", fontsize=12, color="#888888", fontcolor="#888888"];',
]
leaf_answers = []
for i in range(t.node_count):
    people, stroke = int(n_people[i]), int(n_stroke[i])
    stat = f"훈련용 {people:,}명 중 뇌졸중 {stroke:,}명 ({stroke / people * 100:.1f}%)"
    if t.children_left[i] == -1:
        answer = int(tree.classes_[np.argmax(t.value[i][0])])
        leaf_answers.append(answer)
        fill = "#cfe3f4" if answer == 0 else "#f6cfcf"
        word = "아님" if answer == 0 else "뇌졸중"
        lines.append(f'  {i} [label="답: {word}\\n{stat}", fillcolor="{fill}"];')
    else:
        q = question_text(selected[t.feature[i]], t.threshold[i])
        lines.append(f'  {i} [label="{q}\\n{stat}", fillcolor="#f2f2f2"];')
        lines.append(f'  {i} -> {t.children_left[i]} [label="예"];')
        lines.append(f'  {i} -> {t.children_right[i]} [label="아니요"];')
lines.append("}")
st.graphviz_chart("\n".join(lines))

n_leaf = len(leaf_answers)
n_no = leaf_answers.count(0)
n_yes = n_leaf - n_no
st.write(
    f"답을 내는 마디는 모두 **{n_leaf}칸**이고, 그중 **{n_no}칸**이 '아님'이라고 답해요. "
    f"(뇌졸중이라고 답하는 칸: {n_yes}칸)"
)
used = sorted({int(f) for f in t.feature if f >= 0})
if used:
    asked = ", ".join(names[j] for j in used)
    not_asked = [names[j] for j in range(len(selected)) if j not in used]
    extra = f" (묻지 않은 속성: {', '.join(not_asked)})" if not_asked else ""
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것: **{asked}**{extra}")
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것: **없음** (질문 없이 바로 답을 내요)")
