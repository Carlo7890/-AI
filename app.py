import streamlit as st
import pandas as pd
import re

# CSV 불러오기
@st.cache_data
def load_data():
    words_df = pd.read_csv("사고도구어_단어등급매핑.csv", encoding='euc-kr')
    score_df = pd.read_csv("온독지수범위.csv", encoding='euc-kr')

    # 등급별 점수 및 STTR 보정치 적용
    base_scores = {1: 4, 2: 3, 3: 2, 4: 1}
    sttr_weights = {1: 0.73, 2: 0.68, 3: 0.61, 4: 0.55}
    words_df['점수'] = words_df['등급'].apply(lambda g: base_scores[g] * sttr_weights[g])

    # 온독지수 범위 파싱
    score_df[['min', 'max']] = score_df['온독지수 범위'].str.split('~', expand=True).astype(int)

    return words_df, score_df

# 텍스트에서 사고도구어 추출
def extract_words(text, word_list):
    return [word for word in word_list if word in text]

# 온독지수 계산
def calculate_ondok_score(matched_df):
    if matched_df.empty:
        return 0
    max_score = len(matched_df) * max(matched_df['점수'])
    return (matched_df['점수'].sum() / max_score) * 280 if max_score > 0 else 0

# 학년 변환
def estimate_grade(score, score_df):
    for _, row in score_df.iterrows():
        if row['min'] <= score <= row['max']:
            return row['대상 학년']
    return "범위 외"

# Streamlit 앱 시작
st.title("📚 온독AI: 사고도구어 기반 독서지수 분석")

user_input = st.text_area("✍️ 분석할 문장을 입력하세요:")

if user_input:
    words_df, score_df = load_data()
    word_list = words_df['단어'].tolist()
    used_words = extract_words(user_input, word_list)
    matched_df = words_df[words_df['단어'].isin(used_words)]

    if matched_df.empty:
        st.warning("사고도구어가 발견되지 않았어요.")
    else:
        score = calculate_ondok_score(matched_df)
        grade = estimate_grade(score, score_df)

        st.success(f"🧠 온독지수: {score:.1f}점")
        st.info(f"🎓 추정 학년 수준: {grade}")
        st.dataframe(matched_df[['단어', '등급', '점수']])
