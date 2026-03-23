from __future__ import annotations
from typing import Annotated,List
import operator
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END 
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from pydantic import BaseModel,Field
from langgraph.types import Send
from typing import TypedDict,Literal,Optional





load_dotenv()


model = ChatOllama(model="llama3.2:1b")
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (120–450).",
    )
    
    tags: List[str] = Field(default_factory=list) 
    requires_research: bool = False 
    requires_citations : bool = False 
    requires_code  : bool  = False 
class Plan(BaseModel):

    title_blog : str
    viewers : str = Field(...,description="who this blog is for.")
    tone : str = Field(...,description="Writing tone (eg.theoretical,practicle,crisp)")
    blog_kind : Literal["explainer","news_roundup","comparision","system_design","tutorial"] = "explainer"

    constraints : List[str] = Field(default_factory=list) 
    tasks : List[Task]
class EvidenceItem(BaseModel):
    # make sure evidenceitem return from tavli used here  
    title :  str 
    url  : str 
    published_at : Optional[str] = None
    code_snippet: Optional[str] = None 
    source : Optional[str] = None
class RouterDecide(BaseModel):

    needs_research : bool 
    mode : Literal["closed_book","open_book","hybrid"]  # hybrid and open_book need research 
    queries: List[str] # query runs if mode is open or hybrid
class EvidencePack(BaseModel):
    evidence : List[EvidenceItem] = Field(default_factory=list)
class State(TypedDict):

    topic : str
    plan : Optional[Plan] 
    needs_research : bool 
    mode: str
    queries : List[str]
    evidence : List[EvidenceItem]

    division : Annotated[List[tuple[int,str]],operator.add]
    final : str

    merged_md : str 
    md_with_placeholders : str 
    image_specs : List[dict]           
    
    
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""
def router_node(state: State) -> dict:
    topic = state["topic"]
    decider = model.with_structured_output(RouterDecide)
    decision = decider.invoke([SystemMessage(content=ROUTER_SYSTEM),
                                HumanMessage(content=f"Topic: {topic}")])
    if not decision:
        return {
        "needs_research": False,
        "mode": "closed_book",
        "queries": []
        }   
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "generator"  
RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""
def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized

#node
def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])
    max_results = 5

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = model.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}

def generator_node(state:State):

    evidence = state.get("evidence",[])
    mode = state.get("mode","closed_book")

    generate_plan = model.with_structured_output(Plan).invoke(

                    [ SystemMessage(
                    content = f"""You are a senior technical writer and developer advocate.
                    Your job is to produce a highly actionable outline for a technical blog post.

                    Hard requirements:
                    - Create 5–9 sections (tasks) suitable for the topic and audience.
                    - Each task must include:
                    1) goal (1 sentence)
                    2) 3–6 bullets that are concrete, specific, and non-overlapping
                    3) target word count (120–550)

                    Quality bar:
                    - Assume the reader is a developer; use correct terminology.
                    - Bullets must be actionable: build/compare/measure/verify/debug.
                    - Ensure the overall plan includes at least 2 of these somewhere:
                    * minimal code sketch / MWE (set requires_code=True for that section)
                    * edge cases / failure modes
                    * performance/cost considerations
                    * security/privacy considerations (if relevant)
                    * debugging/observability tips

                    Grounding rules:
                    - Mode closed_book: keep it evergreen; do not depend on evidence.
                    - Mode hybrid:
                    - Use evidence for up-to-date examples (models/tools/releases) in bullets.
                    - Mark sections using fresh info as requires_research=True and requires_citations=True.
                    - Mode open_book:
                    - Set blog_kind = "news_roundup".
                    - Every section is about summarizing events + implications.
                    - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
                    - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
                        and includes only what can be supported.

                    Output must strictly match the Plan schema."""),
                    HumanMessage(
                        content= f"topic : {state['topic']}"
                                 f"Mode: {mode} \n"
                                 f"Evidence (Only use fresh claims; may be empty) ; \n"
                                 f"{[e.model_dump() for e in evidence][:16]}"
                    )]
    )


    return {"plan":generate_plan}

# def find_no_of_sections(state:State):
#     return [Send("assistant",{"task":task.model_dump(), 
#                               "topic": state["topic"], 
#                               "mode":state["mode"],
#                               "evidence": [e.model_dump() for e in state.get("evidence",[])],
#                               "plan":state['plan'].model_dump()
#                               }
#                 ) 
#             for task in state["plan"].tasks
#             ] 

def fanout_node(state: State):
    return [
        Send(
            "assistant",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
                "plan": state["plan"].model_dump()
            }
        )
        for task in state["plan"].tasks
    ]

def assistant(load:dict)->dict:
    task = Task(**load["task"])
    topic = load["topic"]
    plan  = Plan(**load["plan"])
    title = plan.title_blog

    evidence  = [EvidenceItem(**e) for e in load.get("evidence",[])]
    mode = load["mode"]

    text_evidence = ""
    if evidence:
        text_evidence = "\n".join(
                        f"-{e.title}| {e.url} | {e.published_at or 'data:unknown'}"
                        for e in evidence[:20] )


        
    total_text_bullet = "\n- " + "\n- ".join(task.bullets)
    division1 = model.invoke(
        [
        SystemMessage(content="""You are a senior technical writer and developer advocate.
                    Write ONE section of a technical blog post in Markdown.

                    Hard constraints:
                    - Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
                    - Stay close to Target words (±15%).
                    - Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
                    - Start with a '## <Section Title>' heading.

                    Scope guard:
                    - If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
                    Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
                    Focus on summarizing events and implications.

                    Grounding policy:
                    - If mode == open_book:
                    - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
                    - For each event claim, attach a source as a Markdown link: ([Source](URL)).
                    - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
                    - If requires_citations == true:
                    - For outside-world claims, cite Evidence URLs the same way.
                    - Evergreen reasoning is OK without citations unless requires_citations is true.

                    Code:
                    - If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

                    Style:
                    - Short paragraphs, bullets where helpful, code fences for code.
                    - Avoid fluff/marketing. Be precise and implementation-oriented.
                    """),  
            HumanMessage(
                content = (
                    f"Blog: {title}\n"
                    f"Audience: {plan.viewers}\n"
                    f"Tone: {plan.tone}\n"
                    f"blog_kind :{plan.blog_kind}\n "
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Target words: {task.target_words}\n"
                    f"Bullets:{total_text_bullet}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{text_evidence}\n"
                )
            )
        ]
    ).content.strip()

    return {"division": [(task.id,division1)]}

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str  #md- markdown  
    images: List[ImageSpec] = Field(default_factory=list)
from pathlib import Path

def merge_content(state: State) -> dict:

    plan = state["plan"]
    if state["plan"] is None:
        raise ValueError("Plan is missing")

    ordered_sections = [md for _, md in sorted(state["division"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.title_blog}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 2 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: ex.[[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    
    planner = model.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def generate_and_place_images(state: State) -> dict:

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{plan.title_blog}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{plan.title_blog}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}
reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content",merge_content)
reducer_graph.add_node("decide_images",decide_images)
reducer_graph.add_node("generate_and_place_images",generate_and_place_images)


reducer_graph.add_edge(START,"merge_content")
reducer_graph.add_edge("merge_content","decide_images")

reducer_graph.add_edge("decide_images","generate_and_place_images")

reducer_graph.add_edge("generate_and_place_images",END)

subgraph = reducer_graph.compile()

graph = StateGraph(State)

graph.add_node("router",router_node)
graph.add_node("fanout_node", fanout_node)
graph.add_node("research",research_node)
graph.add_node("generator",generator_node)
graph.add_node("assistant", assistant)
graph.add_node("reducer",subgraph)


graph.add_edge(START,"router")
graph.add_conditional_edges("router",route_next, {"research":"research","generator":"generator"})
graph.add_edge("research","generator")

graph.add_edge("generator", "fanout_node")
graph.add_edge("fanout_node", "assistant")
graph.add_edge("assistant","reducer",)
graph.add_edge("reducer",END)

workflow = graph.compile()

def run_workflow(topic:str):

    res = workflow.invoke(
        {
            "topic": topic , 
            "mode": "" , 
            "needs_research": False ,
            "plan": None ,
            "division": [] ,
            "final": "",
            "queries":[],
            "evidence" : []
        }
    )

    return res 

