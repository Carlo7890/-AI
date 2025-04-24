import streamlit as st
import pandas as pd
import re
import requests

# CSV 불러오기
@st.cache_data
def load_data():
    words_df = pd.read_csv("사고도구어_단어등급매핑.csv", encoding='euc-kr')
    score_df = pd.read_csv("온독지수범위.csv", encoding='utf-8')

    # 온독지수 범위 파싱
    score_df[['min', 'max']] = score_df['온독지수 범위'].str.split('~', expand=True).astype(int)

    return words_df, score_df

# 텍스트에서 사고도구어 추출
def extract_words(text, word_list):
    return [word for word in word_list if word in text]

# 온독지수 계산 (논문 기반 알고리즘 적용)
def calculate_ondok_score_v2(matched_df, total_tokens):
    if matched_df.empty or total_tokens == 0:
        return 0

    # 등급별 STTR 보정치 (논문 참고 기준값 예시)
    sttr_weights = {1: 0.73, 2: 0.68, 3: 0.61, 4: 0.55}

    # 조정 출현 비율 = 출현 수 / 전체 토큰 수 * STTR
    matched_df['보정비율'] = matched_df['등급'].apply(lambda g: sttr_weights[g])
    adjusted_ratio_sum = len(matched_df) / total_tokens * matched_df['보정비율'].mean()

    # 온독지수 스케일 조정 (100~280 사이로 정규화)
    score = adjusted_ratio_sum * 500  # 보정값
    score = max(0, min(score, 280))
    return score

# 학년 변환
def estimate_grade(score, score_df):
    for _, row in score_df.iterrows():
        if row['min'] <= score <= row['max']:
            return row['대상 학년']
    return "범위 외"

# Groq 기반 LLaMA3 요약 기능

def llama3_summary(text):
    headers = {
        "Authorization": f"Bearer YOUR_GROQ_API_KEY",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "당신은 초등학생을 위한 독서지수 분석 전문가입니다."},
            {"role": "user", "content": f"다음 문장에서 사용된 사고도구어와 의미를 간단히 분석해줘:\n{text}"}
        ]
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return res.json()['choices'][0]['message']['content']
    except:
        return "LLM 요약에 실패했습니다."

# Streamlit 앱 시작
st.title("📚 온독AI: 사고도구어 기반 독서지수 분석")

text_input = st.text_area("✍️ 분석할 문장을 입력하세요:")
run_button = st.button("🔍 분석하기")

if run_button and text_input:
    words_df, score_df = load_data()
    word_list = words_df['단어'].tolist()
    used_words = extract_words(text_input, word_list)
    matched_df = words_df[words_df['단어'].isin(used_words)].copy()

    if matched_df.empty:
        st.warning("사고도구어가 발견되지 않았어요.")
    else:
        total_tokens = len(re.findall(r'\b\w+\b', text_input))
        score = calculate_ondok_score_v2(matched_df, total_tokens)
        grade = estimate_grade(score, score_df)

        st.success(f"🧠 온독지수: {score:.1f}점")
        st.info(f"🎓 추정 학년 수준: {grade}")
        st.dataframe(matched_df[['단어', '등급']].reset_index(drop=True))

        st.markdown("---")
        st.subheader("🧠 LLaMA3 요약 분석 결과")
        st.write(llama3_summary(text_input))


