import streamlit as st
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

# ==================== 配置 ====================
# Pipeline 1: 科室分类（微调模型）
MODEL_DEPT = "JR-2026/CustomModel_medical"   # 请替换为你的实际模型

# 图像生成模型（使用本地 diffusers）
MODEL_IMAGE = "stabilityai/stable-diffusion-2-1"  # 可选：runwayml/stable-diffusion-v1-5

# 提示词模板
PROMPT_TEMPLATE = "A realistic photo of the {department} department in a modern hospital, clean, professional, bright lighting"

# ==================== 加载模型 ====================
@st.cache_resource
def load_department_pipeline():
    """加载科室分类模型"""
    return pipeline("text-classification", model=MODEL_DEPT)

@st.cache_resource
def load_image_pipeline():
    """加载本地 Stable Diffusion 模型（仅当 GPU 可用时）"""
    if not torch.cuda.is_available():
        st.warning("❌ GPU 不可用，无法加载图像生成模型。图片生成功能将禁用。")
        return None
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_IMAGE,
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        # 启用内存优化（可选）
        pipe.enable_attention_slicing()
        return pipe
    except Exception as e:
        st.error(f"图像模型加载失败: {e}")
        return None

# ==================== 图像生成 ====================
def generate_department_image(department, pipe):
    """使用本地 pipeline 生成图片"""
    prompt = PROMPT_TEMPLATE.format(department=department)
    try:
        with st.spinner(f"正在为 {department} 科室生成图片..."):
            image = pipe(prompt, num_inference_steps=25).images[0]
        return image
    except Exception as e:
        st.error(f"生成失败: {e}")
        return None

# ==================== UI ====================
st.set_page_config(page_title="MediTriage AI", page_icon="🏥")
st.title("🏥 MediTriage AI - 智能分诊助手")
st.markdown("请描述症状，系统将推荐科室并自动生成科室图片。")

user_input = st.text_area("症状描述", height=150, placeholder="例如：头痛、发烧两天，伴有恶心...")

if st.button("开始分诊", type="primary"):
    if not user_input.strip():
        st.warning("请输入症状描述。")
    else:
        with st.spinner("正在分析症状..."):
            # 科室分类
            classifier = load_department_pipeline()
            dept_result = classifier(user_input)
            dept_label = dept_result[0]['label']
            dept_score = dept_result[0]['score']

        st.success("分析完成")
        col1, col2 = st.columns(2)

        with col1:
            st.write("📋 推荐科室")
            st.metric(value=f"{dept_score:.2%}", label=dept_label)

        with col2:
            st.write("🖼️ AI 生成科室图片")
            # 加载图像生成模型
            img_pipe = load_image_pipeline()
            if img_pipe is None:
                st.info("⚠️ 未检测到 GPU，无法生成图片。如需启用，请使用支持 GPU 的环境（如 Colab 或本地 GPU）。")
            else:
                image = generate_department_image(dept_label, img_pipe)
                if image:
                    st.image(image, caption=f"{dept_label} 科室", use_container_width=True)
                else:
                    st.info("图片生成失败，请稍后重试。")

        st.info("注：本系统仅供参考，最终诊断请咨询专业医生。")
