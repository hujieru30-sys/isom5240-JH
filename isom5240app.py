import streamlit as st
from transformers import pipeline
import torch

# 设置页面
st.set_page_config(page_title="智能医疗咨询助手", page_icon="🏥")
st.title("🏥 智能医疗咨询辅助系统")
st.markdown("请输入您的健康问题，系统将分析意图、提取关键信息，并生成专业建议。")

# 加载 Pipeline（使用缓存）
@st.cache_resource
def load_intent_pipeline():
    # 使用微调后的意图分类模型
    return pipeline(
        "text-classification",
        model="your-username/medical-intent-classifier",
        tokenizer="your-username/medical-intent-classifier",
        device=-1  # CPU
    )

@st.cache_resource
def load_ner_pipeline():
    # 使用微调后的实体识别模型
    return pipeline(
        "token-classification",
        model="your-username/medical-ner-model",
        tokenizer="your-username/medical-ner-model",
        device=-1,
        aggregation_strategy="simple"  # 合并连续实体
    )

@st.cache_resource
def load_generation_pipeline():
    # 使用预训练的 FLAN-T5 生成建议（不微调）
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1
    )

# 加载模型
intent_pipeline = load_intent_pipeline()
ner_pipeline = load_ner_pipeline()
generator = load_generation_pipeline()

# 用户输入
user_input = st.text_area("描述您的症状或问题：", height=150, placeholder="例如：我头痛两天了，吃了阿司匹林也没好转，怎么办？")

if st.button("分析并生成建议", type="primary"):
    if not user_input.strip():
        st.warning("请输入内容")
    else:
        with st.spinner("分析中..."):
            # 1. 意图分类
            intent_result = intent_pipeline(user_input)[0]
            intent_label = intent_result['label']
            intent_score = intent_result['score']

            # 2. 实体识别
            entities = ner_pipeline(user_input)
            # 整理实体
            entity_dict = {}
            for ent in entities:
                entity_type = ent['entity_group']
                entity_value = ent['word']
                if entity_type not in entity_dict:
                    entity_dict[entity_type] = []
                entity_dict[entity_type].append(entity_value)

            # 3. 构建提示并生成建议
            # 意图映射到中文
            intent_map = {
                "LABEL_0": "症状描述",
                "LABEL_1": "用药咨询",
                "LABEL_2": "检查结果解读",
                "LABEL_3": "治疗建议",
                "LABEL_4": "其他"
            }
            intent_cn = intent_map.get(intent_label, "其他")

            # 构建实体描述
            entity_text = ""
            for etype, vals in entity_dict.items():
                entity_text += f"{etype}: {', '.join(vals)}; "

            # 提示模板
            prompt = f"""
患者咨询意图：{intent_cn}
提取的关键信息：{entity_text}
原问题：{user_input}
请提供医学建议，包括：1）可能的原因，2）建议的下一步行动，3）注意事项。
            """
            # 生成建议
            generated = generator(prompt, max_length=300, do_sample=False)[0]['generated_text']

        # 显示结果
        st.subheader("📊 分析结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("咨询意图", intent_cn)
        with col2:
            st.metric("置信度", f"{intent_score:.2%}")

        if entity_dict:
            st.subheader("🔍 提取的关键信息")
            for etype, vals in entity_dict.items():
                st.write(f"- **{etype}**: {', '.join(vals)}")

        st.subheader("💡 专业建议")
        st.markdown(generated)

        # 免责声明
        st.caption("⚠️ 本系统生成的建议仅供参考，不能替代专业医疗诊断。如有严重不适，请及时就医。")
