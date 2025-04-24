import streamlit as st
import pandas as pd
import re
import requests
import base64
import json

# CSV 불러오기
@st.cache_data
def load_data():
    words_df = pd.read_csv("사고도구어_단어등급매핑.csv", encoding='euc-kr')
    score_df = pd.read_csv("온독지수범위.csv", encoding='utf-8')
    score_df[['min', 'max']] = score_df['온독지수 범위'].str.split('~', expand=True).astype(int)
    return words_df, score_df

# 온독지수 계산 (LLaMA3 추출 기반)
def calculate_ondok_score_from_words(matched_df, score_df):
    total = len(matched_df)
    if total == 0:
        return 0, "사고도구어가 감지되지 않았습니다."

    weighted = sum({1: 4, 2: 3, 3: 2, 4: 1}.get(row["등급"], 1) for _, row in matched_df.iterrows())
    score = min(280, (weighted / (4 * total)) * 280)

    for _, row in score_df.iterrows():
        if row['min'] <= score <= row['max']:
            return round(score), row['대상 학년']
    return round(score), "해석 불가"

# LLaMA3 사고도구어 추출 (CSV 기반 필터)
def llama3_extract_csv_concepts(text, word_list):
    headers = {
        "Authorization": f"Bearer {st.secrets['groq_api_key']}",
        "Content-Type": "application/json"
    }
    prompt = f"""
    다음 문장에서 사고도구어를 추출하되, 반드시 다음 목록에 포함된 단어만 사용하세요:
    {', '.join(word_list)}

    출력은 표 형식으로, 다음 항목을 포함하세요:
    번호 / 단어 / 등급 (예상) / 사전적 의미 / 비슷한 말 / 반대말

    영어는 절대 사용하지 말고, 반드시 한국어로 작성하세요.
    문장:
    {text}
    """
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "당신은 사고도구어 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']
    except:
        return "LLaMA3 API 호출 오류"

# Google Vision OCR

def image_to_text_google_vision(image_file):
    api_key = st.secrets["google_api_key"]
    content = base64.b64encode(image_file.read()).decode("utf-8")
    body = json.dumps({
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    })
    response = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
        headers={"Content-Type": "application/json"},
        data=body
    )
    try:
        return response.json()['responses'][0]['fullTextAnnotation']['text']
    except:
        return ""

# Streamlit 앱 시작
st.title("📚 온독AI: LLaMA3 기반 사고도구어 분석 및 독서지수")

image_file = st.file_uploader("📷 또는 이미지에서 텍스트 추출 (OCR)", type=['png', 'jpg', 'jpeg', 'heic'])
if image_file:
    extracted_text = image_to_text_google_vision(image_file)
    st.text_area("📝 OCR 추출 결과:", value=extracted_text, height=150, key="ocr_output")
    text_input = extracted_text
else:
    text_input = st.text_area("✍️ 분석할 문장을 입력하세요:")

run_button = st.button("🔍 분석하기")

if run_button and text_input:
    words_df, score_df = load_data()
    word_list = words_df['단어'].tolist()

    st.markdown("---")
    st.subheader("🧠 LLaMA3 사고도구어 분석 결과")
    llama_output = llama3_extract_csv_concepts(text_input, word_list)
    st.write(llama_output)

    # 간단한 사고도구어 추출 (CSV 기반 단어만 필터링)
    found_words = [word for word in word_list if word in text_input]
    matched_df = words_df[words_df['단어'].isin(found_words)].copy()
    matched_df.insert(0, '번호', range(1, len(matched_df) + 1))

    score, level = calculate_ondok_score_from_words(matched_df, score_df)
    st.success(f"🧠 온독지수: {score}점")
    st.info(f"🎓 추정 학년 수준: {level}")
    if not matched_df.empty:
        st.dataframe(matched_df.set_index('번호')[['단어', '등급']])

