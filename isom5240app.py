import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ==================== 配置 ====================
# Pipeline 1: 科室分类（你的微调模型）
MODEL_DEPT = "JR-2026/CustomModel_medical"
# Pipeline 2: 紧急程度判断（使用预训练情感分析模型）
MODEL_URGENCY = "nlptown/bert-base-multilingual-uncased-sentiment"  # 输出 1-5 星

# 科室标签映射（如果模型直接输出 label 字符串，则不需要）
# 但为了保险，从模型配置中读取 id2label
@st.cache_resource
def load_department_pipeline():
    # 使用 pipeline 自动加载模型和 tokenizer
    return pipeline("text-classification", model=MODEL_DEPT)

@st.cache_resource
def load_urgency_pipeline():
    # 加载情感分析 pipeline
    return pipeline("sentiment-analysis", model=MODEL_URGENCY)

def map_urgency(score):
    """将情感分数（1-5星）映射为紧急程度"""
    rating = int(score.split()[0])  # 输出格式 "1 star" -> 1
    if rating <= 0.4:
        return "低（建议普通门诊）"
    elif rating <= 0.7:
        return "中（建议尽快就诊）"
    else:
        return "高（建议立即就医）"

# ==================== UI ====================
st.set_page_config(page_title="MediTriage AI", page_icon="🏥")
st.title("🏥 MediTriage AI - 智能医疗分诊助手")
st.markdown("请输入您的症状描述，系统将为您推荐就诊科室并评估紧急程度。")

# 用户输入
user_input = st.text_area("症状描述", height=150, placeholder="例如：我头痛、发烧已经两天了...")

if st.button("开始分诊", type="primary"):
    if not user_input.strip():
        st.warning("请输入症状描述。")
    else:
        with st.spinner("正在分析..."):
            # Pipeline 1: 科室分类
            dept_result = load_department_pipeline()(user_input)
            dept_label = dept_result[0]['label']
            dept_score = dept_result[0]['score']

            # Pipeline 2: 紧急程度判断
            urgency_result = load_urgency_pipeline()(user_input)
            urgency_raw = urgency_result[0]['label']  # 例如 "5 stars"
            urgency_level = map_urgency(urgency_raw)
            urgency_confidence = urgency_result[0]['score']

        # 显示结果
        st.success("分析完成")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 建议科室")
            st.metric(label=dept_label, value=f"{dept_score:.2%}")

        with col2:
            st.subheader("⚠️ 紧急程度")
            st.metric(label=urgency_level, value=f"{urgency_confidence:.2%}")

        # 附加说明
        st.info("注意：本系统仅作为辅助参考，最终诊断请咨询专业医生。")
