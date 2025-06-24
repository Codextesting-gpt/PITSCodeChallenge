from typing import Dict, List, Any
from pydantic import BaseModel, create_model, Field
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from multi_tool_ai import MultitoolAI  # Import your protected class

load_dotenv()

# ────────────────────────────────────────────────────────────
# 2️⃣  Convert your FAISS matches into a *tiny* prompt
# ────────────────────────────────────────────────────────────
def build_prompt(matches: List[Dict[str, Any]], time) -> str:
    """Public prompt building function - visible to users"""
    header = f"""
You are an expert categorizer. Extract the minimal value for each variable from its context.

Guidelines
• Use the match query plus the previous and next segments to find the value.
• If the text doesnt make sense you must transform it to make sense, it utilizes OCR so clean the text if it is not comprehensible
• Return only the value—no labels, units, or extra words. (If the words dont make sense make it make sense, additionally make sure the text has correct grammar)
• Text cannot be only in caps should be correctly placed and should have correct english syntax transform WeIRDTEXT into Weird Text
Additionally make sure you summarize into its minimal format for instance if it says The values given are cats and some dogs you should answer it as --> Cats and dogs.
• Leave any missing value blank.
• After all variables, also return:
    • time        – a concise date/time of how long did the system balanced the document. The total time the system balanced the document is {time} seconds
    • confidence  – a float 0-1 (two decimals) expressing overall certainty and sense the variable makes in comparison with the segment
""".strip()

    blocks = []
    for m in matches:
        blocks.append(f"""
### Variable: {m['entity']}
page: {m['page']}  •  category/file: {m['category']}

previous segment:    {m['prev_segment'] or '<none>'}
matched segment:   {m['matched_text']}
next segment:    {m['next_segment'] or '<none>'}
""".strip())

    return header + "\n\n" + "\n\n".join(blocks)


# ────────────────────────────────────────────────────────────
# 3️⃣  Main extraction function (calls protected methods)
# ────────────────────────────────────────────────────────────
def extract_values(
    matches: List[Dict[str, Any]],
    entities: List[str],
    *,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.0, 
    time: int = 30
) -> Dict[str, Any]:
    ai = MultitoolAI()
    OutputModel = ai._build_extraction_model(entities)
    
    # Standard LLM setup (visible)
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    structured_llm = llm.with_structured_output(OutputModel)
    
    # Build prompt using public function
    prompt = build_prompt(matches, time)
    
    # Get raw result
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    
    # Use protected method to process the result
    final_result = ai._process_extraction_result(result, entities)
    
    return final_result
