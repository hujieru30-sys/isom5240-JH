import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 页面配置
st.set_page_config(page_title="MediTriage AI", page_icon="🏥", layout="wide")

# 侧边栏
with st.sidebar:
    st.markdown("## ⚠️ 免责声明")
    st.warning("本系统仅供辅助分诊参考，不能替代专业医生诊断。如遇紧急情况请立即就医。")
    st.markdown("### 关于 MediTriage AI")
    st.info("基于微调 RoBERTa 模型，将症状分类至 5 个科室，并评估紧急程度。")

# 加载模型（缓存）
@st.cache_resource
def load_models():
    # Pipeline 1: 微调的分诊模型
    tokenizer = AutoTokenizer.from_pretrained("roberta_finetuned")
    model = AutoModelForSequenceClassification.from_pretrained("roberta_finetuned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Pipeline 2: 情感分析模型（用于紧急程度）
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0 if torch.cuda.is_available() else -1
    )
    return model, tokenizer, sentiment_pipeline

def predict_department(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(-1).item()
    # 标签映射（与训练时一致）
    id2label = {0: "内科", 1: "外科", 2: "儿科", 3: "急诊", 4: "其他"}
    return id2label[pred]

def map_urgency(score):
    # 情感分析返回星级（1-5星），映射为紧急程度
    if score <= 2:
        return "低 (建议门诊)"
    elif score == 3:
        return "中 (建议尽快就诊)"
    else:
        return "高 (建议急诊)"

# 加载模型
try:
    model, tokenizer, sentiment_pipeline = load_models()
    st.sidebar.success("模型加载成功")
except Exception as e:
    st.sidebar.error(f"模型加载失败: {e}")
    st.stop()

# 主界面
st.title("🏥 MediTriage AI - 智能医疗分诊助手")
st.markdown("输入患者症状描述，系统将推荐就诊科室并评估紧急程度。")

symptom = st.text_area("症状描述", height=150, placeholder="例如：患者发热38.5度，咳嗽三天，伴有胸闷。")

if st.button("开始分诊", type="primary"):
    if not symptom.strip():
        st.warning("请输入症状描述")
    else:
        with st.spinner("分析中..."):
            # Pipeline 1: 科室分类
            department = predict_department(model, tokenizer, symptom)
            # Pipeline 2: 紧急程度
            sentiment_result = sentiment_pipeline(symptom)[0]
            score = int(sentiment_result['label'].split()[0])  # 提取星级
            urgency = map_urgency(score)
        
        # 显示结果
        st.success("分诊完成")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("推荐科室", department)
        with col2:
            st.metric("紧急程度", urgency)
        
        st.info("提示：此结果仅为 AI 辅助建议，请结合临床判断。")

# 示例
with st.expander("查看示例症状"):
    st.code("患者头痛剧烈，伴有呕吐，视力模糊，持续2小时。")
    if st.button("使用此示例"):
        st.session_state.symptom = "患者头痛剧烈，伴有呕吐，视力模糊，持续2小时。"
        st.experimental_rerun()
