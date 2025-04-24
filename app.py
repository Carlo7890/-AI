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

# 정확한 일치 단어만 추출
def extract_exact_words(text, word_list):
    tokens = re.findall(r"[\w가-힣]+", text)
    return [word for word in tokens if word in word_list]

# 온독지수 계산 개선 버전 (CTTR, 밀도 반영)
def calculate_ondok_score_advanced(text, matched_df, grade_ranges):
    seen = set(matched_df['단어'].tolist())
    total = len(matched_df)
    if total == 0:
        return 0, "사고도구어가 감지되지 않았습니다.", [], 0, 0

    word_tokens = re.findall(r"[\w가-힣]+", text)
    cttr = min(len(seen) / (2 * total) ** 0.5, 1.0)
    weighted = sum({1: 4, 2: 3, 3: 2, 4: 1}[g] for g in matched_df['등급'])
    norm_weight = weighted / (4 * total)
    density = total / len(word_tokens)
    index = ((0.7 * cttr + 0.3 * norm_weight) * 500 + 100) * (0.5 + 0.5 * density)
    if len(word_tokens) < 5:
        index *= 0.6

    matched = [g for s, e, g in grade_ranges if s <= index < e]
    level = "~".join(matched) if matched else "해석 불가"
    return round(index), level, seen, total, len(word_tokens)

# Groq 기반 LLaMA3 사고도구어 분석 기능

def llama3_extract_concepts(text):
    headers = {
        "Authorization": f"Bearer {st.secrets['groq_api_key']}",
        "Content-Type": "application/json"
    }
    prompt = f"""
    다음 글에서 사고도구어를 추출해줘. 각 단어마다 다음 정보를 표 형식으로 작성해줘:
    번호 / 단어 / 등급(1~4 중 예상) / 사전적 의미 / 비슷한 말 / 반대말

    문장:
    {text}

    결과는 반드시 한글로 출력해줘.
    """
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "당신은 초등학생을 위한 사고도구어 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return res.json()['choices'][0]['message']['content']
    except:
        return "LLaMA3 요약 실패 또는 키 오류"

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
st.title("📚 온독AI: 사고도구어 기반 독서지수 분석")

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
    used_words = extract_exact_words(text_input, word_list)
    matched_df = words_df[words_df['단어'].isin(used_words)].copy()

    if matched_df.empty:
        st.warning("사고도구어가 발견되지 않았어요.")
    else:
        grade_ranges = [(row['min'], row['max'], row['대상 학년']) for _, row in score_df.iterrows()]
        score, level, seen_words, total_used, total_words = calculate_ondok_score_advanced(text_input, matched_df, grade_ranges)

        st.success(f"🧠 온독지수: {score:.1f}점")
        st.info(f"🎓 추정 학년 수준: {level}")

        # 1부터 시작하는 번호 열 추가
        display_df = matched_df[['단어', '등급']].copy()
        display_df.insert(0, '번호', range(1, len(display_df) + 1))
        st.dataframe(display_df.set_index('번호'))

        st.markdown("---")
        st.subheader("🧠 LLaMA3 사고도구어 분석 결과")
        st.write(llama3_extract_concepts(text_input))
