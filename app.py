import streamlit as st
import pandas as pd
import re
import requests
import base64
import json

@st.cache_data
def load_data():
    words_df = pd.read_csv("사고도구어_단어등급매핑.csv", encoding='euc-kr')
    score_df = pd.read_csv("온독지수범위.csv", encoding='utf-8')
    score_df[['min', 'max']] = score_df['온독지수 범위'].str.split('~', expand=True).astype(int)
    return words_df, score_df

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

def llama3_extract_csv_concepts(text, word_list):
    headers = {
        "Authorization": f"Bearer {st.secrets['groq_api_key']}",
        "Content-Type": "application/json"
    }
    ambiguous_guide = """
    다음 단어들은 문맥에 따라 등급이 다르므로, 문장에서 어떤 의미로 쓰였는지를 판단하여 해당 의미와 등급을 선택하세요:
    - 기술: 2등급 기술1: 사물을 잘 다룰 수 있는 방법이나 능력(기능/방법).
            3등급 기술3: 열거하거나 기록하여 서술함(기록/서술).
    - 유형: 2등급 유형7: 성질이나 특징 따위가 공통적인 것끼리 묶은 하나의 틀(갈래).
            3등급 유형2: 형상이나 형체가 있음.
    - 의지: 2등급 의지6: 어떠한 일을 이루고자 하는 마음(결심).
            3등급 의지4: 다른 것에 몸을 기대거나 마음을 기대어 도움을 받음.
    - 지적: 2등급 지적5: 꼭 집어서 가리킴(지목/지시).
            3등급 지적1: 지식이나 지성에 관한 것(지성적).
    """
    prompt = f"""
    다음 문장에서 사고도구어를 추출하세요. 반드시 아래 목록에 포함된 단어만 사용할 수 있습니다:
    {', '.join(word_list)}

    {ambiguous_guide}

    출력은 표가 아닌 순수 텍스트로 작성하세요. 예: '기술/2', '유형/3'
    출력은 반드시 한국어로만 작성하십시오. 영어가 섞여 있으면 오답 처리됩니다.

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

    llama_output = llama3_extract_csv_concepts(text_input, word_list)

    matched_words = []
    for line in re.split(r'[,
]', llama_output):
        parts = line.strip().split('/')
        if len(parts) == 2:
            word = parts[0].strip()
            try:
                grade = int(parts[1])
                match = words_df[(words_df['단어'] == word) & (words_df['등급'] == grade)]
                matched_words.append(match)
            except:
                continue

    if any(not df.empty for df in matched_words):
        matched_df = pd.concat([df for df in matched_words if not df.empty]).drop_duplicates().reset_index(drop=True)
        score, level = calculate_ondok_score_from_words(matched_df, score_df)
        st.markdown("---")
        st.success(f"🧠 온독지수: {score}점")
        st.info(f"🎓 추정 학년 수준: {level}")
        st.markdown("---")
        st.subheader("🔍 사용된 사고도구어")
        used_list = [f"{row['단어']}/{row['등급']}등급" for _, row in matched_df.iterrows()]
        st.write(', '.join(used_list))
    else:
        st.warning("사고도구어가 발견되지 않았어요.")
