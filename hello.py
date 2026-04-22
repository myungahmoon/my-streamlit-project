import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# -----------------------------
# 한글 깨짐 해결
# -----------------------------
import matplotlib.font_manager as fm
import urllib.request
import os

font_path = "NanumGothic.ttf"

if not os.path.exists(font_path):
    url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    urllib.request.urlretrieve(url, font_path)

fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 제목
# -----------------------------
st.title("청소년 행복도 영향 요인 분석")

# -----------------------------
# 연구 개요
# -----------------------------
st.header("연구 개요")

st.markdown("""
▪ **활용한 데이터**  
- 2019년 청소년이 행복한 지역사회 지표조사 및 조성사업 연구Ⅶ  
(한국 아동·청소년·청년 데이터 아카이브)

▪ **연구주제**  
- 관계영역(부모님, 친구들)과 교육과정(교육과정 만족도)이  
청소년(중·고등학생)의 행복도에 미치는 영향 분석  

▪ **분석 방법 및 과정**  
1. 설문조사 중 해당 문항 추출  
2. 다중선형 회귀분석을 통해  
   관계 영역과 교육 영역이 행복도에 미치는 상대적 영향력 비교 분석
""")

# -----------------------------
# 데이터 불러오기
# -----------------------------
df = pd.read_csv("data.csv")

# -----------------------------
# 변수 생성
# -----------------------------
st.header("변수 구성")

df["관계영역"] = df[["Q6A2", "Q6A3"]].mean(axis=1)
df["교육영역"] = df["Q18A2"]
df["행복도"] = df["Q2A3"]

df_model = df[["관계영역", "교육영역", "행복도"]].dropna()

st.markdown("""
▪ 독립변수 1 [관계 영역]: 부모님 관계(Q6A2) + 친구 관계(Q6A3) 평균  
▪ 독립변수 2 [교육 영역]: 교과과정 만족도(Q18A2)  
▪ 종속변수 [행복도]: 행복한 감정(Q2A3)  

▪ 분석 대상: 결측치 제거 후 약 5,000~6,000명
""")

st.write("데이터 수:", len(df_model))

# -----------------------------
# 기술통계
# -----------------------------
st.header("기술통계 및 상관관계")

desc = df_model.describe().T[['mean', 'std']]
st.write(desc)

corr = df_model.corr()

fig1, ax1 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
ax1.set_xticklabels(['관계영역', '교육영역', '행복도'])
ax1.set_yticklabels(['관계영역', '교육영역', '행복도'])

st.pyplot(fig1)

st.markdown("""
모든 독립변수는 행복도와 정(+)의 상관관계를 보였다.  
특히 관계 영역이 교육 영역보다 더 높은 상관관계를 나타냈다.
""")

# -----------------------------
# 회귀분석
# -----------------------------
st.header("다중회귀분석 결과")

X = df_model[["관계영역", "교육영역"]]
y = df_model["행복도"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

st.text(model.summary())

st.markdown("""
모형 적합도: F ≈ 1245, p < .001  
설명력(R²) ≈ 0.32
""")

# -----------------------------
# 시각화
# -----------------------------
st.header("시각화")

option = st.selectbox(
    "그래프 선택",
    ["관계영역 vs 행복도", "교육영역 vs 행복도"]
)

fig2, ax2 = plt.subplots()

if option == "관계영역 vs 행복도":
    sns.regplot(x=df_model["관계영역"], y=df_model["행복도"], ax=ax2)
    ax2.set_xlabel("관계영역")
    ax2.set_ylabel("행복도")
    ax2.set_title("관계영역과 행복도의 관계")

else:
    sns.regplot(x=df_model["교육영역"], y=df_model["행복도"], ax=ax2)
    ax2.set_xlabel("교육영역")
    ax2.set_ylabel("행복도")
    ax2.set_title("교육영역과 행복도의 관계")

st.pyplot(fig2)

# -----------------------------
# 결과 해석
# -----------------------------
st.header("결과 해석 및 논의")

st.markdown("""
① **통계적 유의성 확인**  
관계 영역과 교육 영역 모두 행복도에 유의미한 정(+)의 영향을 미친다.  

② **상대적 영향력 비교**  
관계 영역의 영향력이 교육 영역보다 약 2배 정도 더 크다.  

③ **결론**  
청소년의 행복을 높이기 위해서는 교육 환경뿐 아니라  
부모 및 친구와의 긍정적인 관계 형성이 핵심 요인이다.
""")