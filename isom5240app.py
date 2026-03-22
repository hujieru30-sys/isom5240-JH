import streamlit as st
from transformers import pipeline
from huggingface_hub import InferenceClient
from PIL import Image
import os

# ==================== Configuration ====================
# Pipeline 1: Department classification (fine-tuned model)
MODEL_DEPT = "JR-2026/CustomModel_medical"

# Image generation model (Hugging Face Inference API)
MODEL_IMAGE = "zai-org/GLM-Image"  # 适合生成包含文字信息的科室图片

# 科室图片提示词映射（可根据实际模型输出调整）
PROMPT_TEMPLATE = "A professional, realistic image of the {department} department in a modern hospital, clean design, soft lighting, 4k quality"

# ==================== Load models ====================
@st.cache_resource
def load_department_pipeline():
    """加载部门分类模型"""
    return pipeline("text-classification", model=MODEL_DEPT)

@st.cache_resource
def get_inference_client():
    """获取 Hugging Face Inference API 客户端"""
    # 从环境变量或 secrets 读取 token
    token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
    if not token:
        st.warning("未设置 HF_TOKEN，图像生成功能将不可用。请在 .streamlit/secrets.toml 中配置。")
    return InferenceClient(token=token) if token else InferenceClient()

# ==================== Image generation ====================
def generate_department_image(department, client):
    """根据科室名称生成图片"""
    prompt = PROMPT_TEMPLATE.format(department=department)
    try:
        image = client.text_to_image(prompt, model=MODEL_IMAGE)
        return image
    except Exception as e:
        st.error(f"图片生成失败: {e}")
        return None

# ==================== UI ====================
st.set_page_config(page_title="MediTriage AI", page_icon="🏥")
st.title("🏥 MediTriage AI - Smart Medical Triage Assistant")
st.markdown("请描述您的症状，系统将推荐科室并生成对应科室图片。")

user_input = st.text_area("症状描述", height=150, placeholder="例如：头痛、发烧两天，伴有恶心...")

if st.button("开始分诊", type="primary"):
    if not user_input.strip():
        st.warning("请输入症状描述。")
    else:
        with st.spinner("正在分析..."):
            # Pipeline 1: 科室分类
            dept_result = load_department_pipeline()(user_input)
            dept_label = dept_result[0]['label']
            dept_score = dept_result[0]['score']

        # 显示科室推荐结果
        st.success("分析完成")
        col1, col2 = st.columns(2)

        with col1:
            st.write("📋 推荐科室")
            st.metric(value=f"{dept_score:.2%}", label=dept_label)

        # 图像生成部分（原 Pipeline 2 的替代）
        with col2:
            st.write("🖼️ 科室图片（AI 生成）")
            client = get_inference_client()
            # 检查是否有 token
            if client.token:
                with st.spinner("正在生成科室图片..."):
                    image = generate_department_image(dept_label, client)
                if image:
                    st.image(image, caption=f"{dept_label} 科室", use_container_width=True)
                else:
                    st.info("图片生成失败，请检查网络或稍后重试。")
            else:
                st.info("未配置 Hugging Face Token，无法生成图片。请在 .streamlit/secrets.toml 中添加 HF_TOKEN。")

        # 说明信息
        st.info("注：本系统仅供参考，最终诊断请咨询专业医生。")
