import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt


st.header("NeuroNote")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is ALL You Need ,BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Code-Oriented", "Mathematical"]
)

#template
template = load_prompt('template1.json')
template.invoke(
    {'paper_input': paper_input, 'style_input': style_input}
)

if st.button("Summarize"):
    rendered_prompt = template.format(
        paper_input=paper_input,
        style_input=style_input
    )
    st.write(rendered_prompt)