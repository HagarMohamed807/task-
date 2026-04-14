"""
Academic Advisor Agent — FastAPI Service
Recommends one of 12 programming tracks through a 10-question adaptive MCQ interview.
"""

import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from config import GEMINI_API_KEY, GEMINI_BASE_URL, MODEL_NAME

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Academic Advisor Agent",
    description="Recommends the ideal programming track through a 10-question MCQ interview.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

# ── In-Memory Session Store ────────────────────────────────────────────────
# sessions_db[session_id] = list of OpenAI message dicts (chat history)
sessions_db: dict[str, list[dict]] = {}

# ── Available Tracks ──────────────────────────────────────────────────────
TRACKS = [
    "Frontend Development",
    "Backend Development",
    "Mobile Development",
    "Data Science & Analysis",
    "Data Engineering (ETL, Pipelines, Big Data)",
    "Computer Vision (Image Processing, Deep Learning)",
    "Internet of Things (IoT & Hardware)",
    "Cloud Engineering & DevOps",
    "Cybersecurity",
    "Game Development",
    "Artificial Intelligence (General ML/DL)",
    "Embedded Systems",
]

# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a Senior Technical Architect and Career Counselor.
Your task is to interview a student to discover their ideal programming track.

Available tracks:
{chr(10).join(f"- {t}" for t in TRACKS)}

RULES — follow these exactly:
1. Ask exactly 10 questions total, one at a time, in MCQ format.
2. Each question must have exactly 4 options labeled A, B, C, and D.
3. Questions must be adaptive: use the student's previous answers to dig deeper.
4. Ask deep, scenario-based questions about logic preferences, system design,
   hardware vs software, data intuition, creativity, performance, and more.
5. Vary the starting angle and phrasing each session so re-taking feels fresh.
6. After the student answers the 10th question — and ONLY then — call the
   `recommend_track` tool. Do NOT recommend earlier.
7. Never reveal the track list to the student. Never explain your reasoning
   in plain text — only through the tool call.

OUTPUT FORMAT (questions 1-10):
Return a JSON object like this — nothing else:
{{
  "question_number": <1-10>,
  "question": "<scenario-based question text>",
  "options": {{
    "A": "<option A>",
    "B": "<option B>",
    "C": "<option C>",
    "D": "<option D>"
  }}
}}

After the 10th answer, call recommend_track instead of returning JSON.
"""

# ── Tool Definition ────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_track",
            "description": (
                "Call this ONLY after the student has answered all 10 questions. "
                "Provide the recommended track and a deep analysis of their choices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_name": {
                        "type": "string",
                        "description": "The recommended programming track.",
                        "enum": TRACKS,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "A deep, personalised analysis (3-5 sentences) of why "
                            "this track suits the student based on all 10 answers."
                        ),
                    },
                },
                "required": ["track_name", "reasoning"],
            },
        },
    }
]

# ── Pydantic Schemas ───────────────────────────────────────────────────────
class StartResponse(BaseModel):
    session_id: str
    question: str
    options: dict[str, str]
    question_number: int
    is_finished: bool
    recommendation: dict | None


class AnswerRequest(BaseModel):
    session_id: str
    user_answer: str   # e.g. "A" or the full answer text


class AnswerResponse(BaseModel):
    session_id: str
    question: str | None
    options: dict[str, str] | None
    question_number: int | None
    is_finished: bool
    recommendation: dict | None


# ── Helper: Call the LLM ───────────────────────────────────────────────────
def call_llm(messages: list[dict]) -> dict:
    """
    Send the full conversation to OpenAI.
    Returns either:
      {"type": "question", "data": {question_number, question, options}}
      {"type": "recommendation", "data": {track_name, reasoning}}
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.85,
    )

    choice = response.choices[0].message

    # ── The LLM called recommend_track ──
    if choice.tool_calls:
        tool_call = choice.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        return {
            "type": "recommendation",
            "data": {
                "track_name": args["track_name"],
                "reasoning": args["reasoning"],
            },
        }

    # ── The LLM returned a question ──
    content = json.loads(choice.content)
    return {
        "type": "question",
        "data": {
            "question_number": content["question_number"],
            "question": content["question"],
            "options": content["options"],
        },
    }


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/start", response_model=StartResponse, summary="Start a new interview session")
def start_session():
    """
    Creates a new session and returns the first MCQ question.
    Call this once per student at the beginning of the interview.
    """
    session_id = str(uuid.uuid4())

    # Initial message history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Start the interview. Ask me the first question."},
    ]

    result = call_llm(messages)

    if result["type"] != "question":
        raise HTTPException(status_code=500, detail="Unexpected LLM response on session start.")

    # Add the assistant's reply to history before saving
    messages.append({
        "role": "assistant",
        "content": json.dumps(result["data"]),
    })

    sessions_db[session_id] = messages

    return StartResponse(
        session_id=session_id,
        question=result["data"]["question"],
        options=result["data"]["options"],
        question_number=result["data"]["question_number"],
        is_finished=False,
        recommendation=None,
    )


@app.post("/answer", response_model=AnswerResponse, summary="Submit an answer and get the next question")
def submit_answer(body: AnswerRequest):
    """
    Receives the student's answer and returns either:
    - The next MCQ question (questions 1-9)
    - The final recommendation + reasoning (after question 10)
    """
    # ── Session validation ──
    if body.session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found. Please call /start first.")

    messages = sessions_db[body.session_id]

    # ── Append the student's answer ──
    messages.append({
        "role": "user",
        "content": f"My answer is: {body.user_answer}",
    })

    # ── Call LLM ──
    result = call_llm(messages)

    # ── Recommendation ──
    if result["type"] == "recommendation":
        # Clean up session (interview is done)
        del sessions_db[body.session_id]
        return AnswerResponse(
            session_id=body.session_id,
            question=None,
            options=None,
            question_number=None,
            is_finished=True,
            recommendation=result["data"],
        )

    # ── Next question ──
    messages.append({
        "role": "assistant",
        "content": json.dumps(result["data"]),
    })
    sessions_db[body.session_id] = messages   # Update session

    return AnswerResponse(
        session_id=body.session_id,
        question=result["data"]["question"],
        options=result["data"]["options"],
        question_number=result["data"]["question_number"],
        is_finished=False,
        recommendation=None,
    )


# ── Health Check ───────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "active_sessions": len(sessions_db)}
