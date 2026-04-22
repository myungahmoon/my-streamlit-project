import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# ---------------------------
# 연구 배경
# ---------------------------
st.header("연구의 동기 및 배경")

st.write("""
최근 청소년의 행복도는 다양한 요인의 영향을 받는다.
특히 인간관계와 교육 환경은 중요한 영향을 미치는 요소로 알려져 있다.

본 연구에서는 관계 영역과 교육 영역이 행복도에 미치는 영향을 분석하고자 한다.
""")

# ---------------------------
# 연구 방법
# ---------------------------
st.header("연구 방법")

st.write("""
설문 데이터를 기반으로 관계 영역과 교육 영역 변수를 생성하였다.
이후 상관관계 분석과 회귀분석을 통해 각 변수의 영향을 분석하였다.
""")
# ⭐ 추가 (한글 깨짐 해결)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# 제목
# ---------------------------
st.title("청소년 행복도 분석 앱")

# ---------------------------
# 데이터 불러오기
# ---------------------------
df = pd.read_csv("data.csv")

st.subheader("데이터 미리보기")
st.write(df.head())

# 컬럼 확인 (필요 없으면 나중에 삭제 가능)
st.write("컬럼 목록:", df.columns)

# ---------------------------
# 변수 생성
# ---------------------------
st.subheader("변수 생성")

df['관계영역'] = df[['Q6A2', 'Q6A3']].mean(axis=1)
df['교육영역'] = df['Q18A2']
df['행복도'] = df['Q2A3']

st.write(df[['관계영역', '교육영역', '행복도']].head())

# ---------------------------
# 결측값 제거
# ---------------------------
df_model = df[['관계영역', '교육영역', '행복도']].dropna()

st.write("결측값 제거 후 데이터 수:", len(df_model))

# ---------------------------
# 상관관계 분석
# ---------------------------
st.subheader("상관관계 분석")

corr = df_model.corr()

fig1, ax1 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
ax1.set_title("변수 간 상관관계")
st.pyplot(fig1)

# ---------------------------
# 회귀분석
# ---------------------------
st.subheader("회귀분석 결과")

X = df_model[['관계영역', '교육영역']]
y = df_model['행복도']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

st.text(model.summary())

# ---------------------------
# 시각화 (그래프 선택)
# ---------------------------
st.subheader("그래프 시각화")

option = st.selectbox(
    "그래프 선택",
    ["관계영역 vs 행복도", "교육영역 vs 행복도"]
)

fig2, ax2 = plt.subplots()

if option == "관계영역 vs 행복도":
    sns.regplot(
        x='관계영역',
        y='행복도',
        data=df_model,
        ax=ax2,
        scatter_kws={'alpha': 0.3}
    )
    ax2.set_title("관계영역이 행복도에 미치는 영향")

else:
    sns.regplot(
        x='교육영역',
        y='행복도',
        data=df_model,
        ax=ax2,
        scatter_kws={'alpha': 0.3}
    )
    ax2.set_title("교육영역이 행복도에 미치는 영향")

st.pyplot(fig2)