from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template="""

Task:
1. Read the text below.
2. Summarize it according to the chosen style.

Styles:
- Short: 1–2 sentences
- Medium: 3–4 sentences
- Detailed: Covers all major points

Style chosen: "{style_input}"

Document:
"{paper_input}"

""",
    input_variables=['paper_input','style_input'] ,
    validate_template=True
)

template.save('template1.json')