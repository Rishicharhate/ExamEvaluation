"""
╔══════════════════════════════════════════════════════════════════╗
║         AI EXAM EVALUATION SYSTEM  —  COMMERCIAL GRADE          ║
║                                                                  ║
║  OCR    : Groq VLM  (LLaMA 4 Scout)  — your existing script     ║
║  LLM    : Qwen2.5 7B via Ollama      — local, private           ║
║                                                                  ║
║  Handles:                                                        ║
║    ✓ Paragraph / long answers                                    ║
║    ✓ Diagram-based questions (LLM describes what it sees)        ║
║    ✓ Hindi + English mixed                                       ║
║    ✓ Any subject, any exam — fully dynamic                       ║
║    ✓ Dual-reference semantic evaluation (your idea)             ║
║    ✓ Examiner + LLM answer both used for fairness               ║
║    ✓ Keyword fuzzy matching                                      ║
║    ✓ Cheating detection                                          ║
║    ✓ Full audit log                                              ║
║                                                                  ║
║  Setup:                                                          ║
║    pip install groq sentence-transformers keybert scikit-learn   ║
║    pip install requests Pillow numpy                             ║
║    ollama serve  (in separate terminal)                          ║
║    ollama pull qwen2.5:7b                                        ║
║    export GROQ_API_KEY="your_key_here"                           ║
║                                                                  ║
║  Usage:                                                          ║
║    python exam_evaluator_v2.py                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import gc
import re
import json
import time
import base64
import requests
import warnings
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── optional: load .env if present ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit before running
# ══════════════════════════════════════════════════════════════════

CONFIG = {
    # Groq VLM (for OCR)
    "groq_model"           : "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq_api_key"         : os.environ.get("GROQ_API_KEY", ""),

    # Ollama LLM (for evaluation)
    "ollama_url"           : "http://localhost:11434/api/generate",
    "ollama_model"         : "gemma3:1b",        # change to qwen2.5:3b for 4GB GPU

    # Embedding model (for keyword fuzzy match + cheating detection)
    "embedding_model"      : "paraphrase-multilingual-MiniLM-L12-v2",

    # Score fusion weights (must sum to 1.0)
    # semantic = LLM understanding, keyword = term presence
    "alpha"                : 0.6,
    "beta"                 : 0.4,

    # Dual reference weights (your idea — examiner vs LLM answer)
    # how much weight to give each reference when combining
    "examiner_ref_weight"  : 0.6,
    "llm_ref_weight"       : 0.4,

    # Cheating detection
    "similarity_threshold" : 0.85,

    # Output
    "output_dir"           : "results",
    "log_file"             : "audit_log.jsonl",
}


# ══════════════════════════════════════════════════════════════════
#  LOGGER  — every action recorded to audit log
# ══════════════════════════════════════════════════════════════════

class AuditLogger:
    def __init__(self):
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        self.log_path = os.path.join(CONFIG["output_dir"], CONFIG["log_file"])

    def log(self, event: str, data: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event"    : event,
            **data,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def info(self, msg: str):
        print(f"  [INFO]  {msg}")
        self.log("info", {"message": msg})

    def warn(self, msg: str):
        print(f"  [WARN]  {msg}")
        self.log("warning", {"message": msg})

    def error(self, msg: str, exc: Exception = None):
        print(f"  [ERROR] {msg}")
        self.log("error", {
            "message"  : msg,
            "exception": traceback.format_exc() if exc else "",
        })


logger = AuditLogger()


# ══════════════════════════════════════════════════════════════════
#  MODULE 1 — GROQ VLM OCR
#  Your existing script — integrated cleanly
#  Handles: text answers, diagram answers, Hindi, English, mixed
# ══════════════════════════════════════════════════════════════════

class GroqVLM:
    """
    Uses Groq's LLaMA 4 Scout to extract text from handwritten images.
    Works for:
      - Normal handwritten text (Hindi + English)
      - Diagrams with labels
      - Mixed content pages
      - Multiple questions on one page
    """

    # ── OCR prompt — handles all answer types ─────────────────────
    OCR_PROMPT_ANSWER_KEY = """You are an expert OCR system for government exam answer keys.
Carefully analyze this handwritten or printed image.

Extract ALL content visible:

**ENGLISH TEXT:**
[All English text exactly as written, preserve line breaks]

**HINDI TEXT (हिंदी पाठ):**
[All Hindi/Devanagari text exactly as written, preserve line breaks]

**DIAGRAM DESCRIPTIONS:**
[For any diagram, flowchart, graph or drawing found:
 - Describe what the diagram shows
 - List all labels, arrows, boxes visible in it
 - Note what concept it illustrates]

**COMBINED TEXT (Original Layout):**
[Everything together in top-to-bottom order as written on paper]

**QUESTION BOUNDARIES:**
[Identify where each question starts and ends. Format:
 Q1: lines 1-5
 Q2: lines 6-12
 etc.]

Rules:
- Preserve original spelling including mistakes
- Do NOT correct grammar
- Do NOT translate
- Unclear words: write best guess with (?)
- For diagrams: describe thoroughly so an LLM can evaluate without seeing the image
"""

    OCR_PROMPT_STUDENT = """You are an expert OCR system for university engineering exam answer sheets.
Carefully analyze this handwritten image of a student's answer sheet.

IMPORTANT: Indian exam answer sheets have a "Question No." or "Q.No." column on the LEFT MARGIN of each page.
The student WRITES which question they are answering in that left column (e.g. "Q.9", "9a", "9(A)", "5b", "Q7" etc.).
These question-number labels are YOUR PRIMARY ANCHORS for segmenting the answers.

TWO-STEP EXTRACTION PROCESS:
  STEP 1 — SCAN THE LEFT COLUMN:
    Look along the LEFT MARGIN only. Find every question number the student wrote there.
    Write them all down first before looking at any answer text.
    Example finds: "9a", "6b", "2b", "7b", "12a"

  STEP 2 — FOR EACH QUESTION NUMBER FOUND, EXTRACT ITS FULL ANSWER:
    Go back to the page body. Starting from where the student wrote "9a" in the left column,
    extract EVERYTHING the student wrote as the answer — all lines, all paragraphs, all bullet points —
    until the NEXT question number appears in the left column.
    Do NOT stop early. Capture the COMPLETE answer.

**QUESTION-ANCHORED ANSWERS:**
[For each question number the student wrote, produce one block:

Q_NUM: <exactly what the student wrote, like "9a", "9A", "5b", "Q.9" — normalize to digits+letter, e.g. "9a">
ANSWER_TEXT: <COMPLETE answer the student wrote for this question — ALL lines, paragraphs, bullet points, numbered points, everything — word for word including mistakes. Do NOT truncate.>
DIAGRAM: <if any diagram drawn for this question: describe labels, arrows, boxes, components; else write "None">
---
]

**COMBINED TEXT (Original Layout):**
[Everything together in top-to-bottom order as written on paper]

**ENGLISH TEXT:**
[All English text exactly as written, preserve mistakes]

**HINDI TEXT (हिंदी पाठ):**
[All Hindi text exactly as written, preserve mistakes]

**DIAGRAM DESCRIPTIONS:**
[For any diagram: describe what was drawn with all visible labels, arrows, boxes]

**ANSWER BOUNDARIES:**
[Format:
 Q9a: lines 1-15
 Q9b: lines 16-28
 etc.]

Rules:
- Preserve student's EXACT wording — do NOT improve or correct
- Preserve all spelling mistakes
- Do NOT translate Hindi to English
- If a question number column value is unclear, write best guess with (?)
- NEVER invent or add text the student did not write
- CRITICAL: Extract the COMPLETE answer for each question — do not stop after a few lines
- A single answer may span multiple paragraphs and multiple bullet/numbered points — capture ALL of it
"""

    def __init__(self):
        self._client = None
        self._check_groq()

    def _check_groq(self):
        if not CONFIG["groq_api_key"]:
            logger.warn("GROQ_API_KEY not set. OCR will use fallback (manual input).")
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=CONFIG["groq_api_key"])
            logger.info(f"Groq VLM ready. Model: {CONFIG['groq_model']}")
        except ImportError:
            logger.warn("groq package not installed. Run: pip install groq")

    @staticmethod
    def _encode_image(image_path: str) -> tuple[str, str]:
        """Returns (base64_data, media_type)"""
        ext = Path(image_path).suffix.lower().lstrip(".")
        media_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "gif": "image/gif",
            "webp": "image/webp",
        }
        media_type = media_map.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return data, media_type

    def _parse_response(self, text: str) -> dict:
        """Parse VLM response into structured sections."""
        result = {
            "english_text"        : "",
            "hindi_text"          : "",
            "diagram_descriptions": "",
            "combined_text"       : "",
            "boundaries"          : "",
            "full_response"       : text,
            "anchored_answers"    : {},   # {q_num_str: {text_answer, diagram_description}}
        }

        sections = {
            "english_text"        : ("**ENGLISH TEXT:**",                     ["**HINDI TEXT", "**DIAGRAM"]),
            "hindi_text"          : ("**HINDI TEXT",                          ["**DIAGRAM", "**COMBINED"]),
            "diagram_descriptions": ("**DIAGRAM DESCRIPTIONS:**",             ["**COMBINED", "**ANSWER"]),
            "combined_text"       : ("**COMBINED TEXT",                       ["**QUESTION BOUNDARIES", "**ANSWER BOUNDARIES", "**ENGLISH"]),
            "boundaries"          : (["**QUESTION BOUNDARIES:", "**ANSWER BOUNDARIES:"], []),
        }

        for key, (start_marker, end_markers) in sections.items():
            markers = [start_marker] if isinstance(start_marker, str) else start_marker
            for marker in markers:
                if marker in text:
                    s = text.find(marker)
                    s = text.find("\n", s) + 1
                    e = len(text)
                    for em in end_markers:
                        pos = text.find(em, s)
                        if pos != -1:
                            e = min(e, pos)
                    result[key] = text[s:e].strip()
                    break

        # ── Parse QUESTION-ANCHORED ANSWERS blocks ───────────────────────
        anchor_marker = "**QUESTION-ANCHORED ANSWERS:**"
        if anchor_marker in text:
            anchor_start = text.find(anchor_marker) + len(anchor_marker)
            # Find end (next ** section)
            anchor_end = len(text)
            for next_sec in ["**COMBINED TEXT", "**ENGLISH TEXT", "**HINDI TEXT", "**DIAGRAM"]:
                pos = text.find(next_sec, anchor_start)
                if pos != -1:
                    anchor_end = min(anchor_end, pos)
            anchor_block = text[anchor_start:anchor_end].strip()

            # Split by --- separator that marks each Q block
            raw_blocks = anchor_block.split("---")
            for block in raw_blocks:
                block = block.strip()
                if not block:
                    continue
                q_num = ""
                answer_text = ""
                diagram_text = ""
                for line in block.splitlines():
                    line = line.strip()
                    if line.upper().startswith("Q_NUM:"):
                        raw_q = line[len("Q_NUM:"):].strip()
                        # Normalise: strip Q., spaces → e.g. '9a', '5b', '7'
                        import re
                        q_num = re.sub(r'[Qq][.\s]*', '', raw_q).strip().lower()
                    elif line.upper().startswith("ANSWER_TEXT:"):
                        answer_text = line[len("ANSWER_TEXT:"):].strip()
                    elif line.upper().startswith("DIAGRAM:"):
                        diagram_text = line[len("DIAGRAM:"):].strip()
                    elif answer_text and not line.upper().startswith("Q_NUM:"):
                        answer_text += " " + line  # continuation lines
                if q_num:
                    result["anchored_answers"][q_num] = {
                        "text_answer"        : answer_text,
                        "diagram_description": diagram_text if diagram_text.lower() != "none" else "",
                        "is_blank"           : not bool(answer_text.strip()),
                    }

        return result

    def extract(self, image_path: str, mode: str = "student") -> dict:
        """
        mode: "answer_key" or "student"
        Returns structured dict with extracted text sections.
        Falls back to manual input if Groq unavailable.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"OCR extracting: {image_path} (mode={mode})")

        # ── Fallback if no Groq ────────────────────────────────────
        if not self._client:
            logger.warn("No Groq client — using manual text input.")
            text = input(f"\n  Paste the text content of '{image_path}':\n  > ").strip()
            return {
                "english_text"        : text,
                "hindi_text"          : "",
                "diagram_descriptions": "",
                "combined_text"       : text,
                "boundaries"          : "",
                "full_response"       : text,
            }

        prompt = (
            self.OCR_PROMPT_ANSWER_KEY
            if mode == "answer_key"
            else self.OCR_PROMPT_STUDENT
        )

        try:
            img_data, media_type = self._encode_image(image_path)
            data_url = f"data:{media_type};base64,{img_data}"

            response = self._client.chat.completions.create(
                model=CONFIG["groq_model"],
                messages=[{
                    "role"   : "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text",      "text"     : prompt},
                    ],
                }],
                max_tokens  = 2048,
                temperature = 0.1,
            )

            raw    = response.choices[0].message.content
            result = self._parse_response(raw)
            result["tokens_used"] = response.usage.total_tokens
            result["model_used"]  = response.model

            logger.info(f"OCR complete. Tokens: {result.get('tokens_used', '?')}")
            logger.log("ocr_result", {
                "image"    : image_path,
                "mode"     : mode,
                "preview"  : result["combined_text"][:200],
            })

            return result

        except Exception as e:
            logger.error(f"Groq OCR failed for {image_path}", e)
            # Graceful fallback
            return {
                "english_text"        : "",
                "hindi_text"          : "",
                "diagram_descriptions": "",
                "combined_text"       : "",
                "boundaries"          : "",
                "full_response"       : f"OCR_ERROR: {str(e)}",
            }


# ══════════════════════════════════════════════════════════════════
#  MODULE 2 — OLLAMA LLM CORE
# ══════════════════════════════════════════════════════════════════

class OllamaLLM:
    """Ollama wrapper — all prompts return JSON."""

    def __init__(self):
        self._check()

    def _check(self):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                model_base = CONFIG["ollama_model"].split(":")[0]
                found = any(model_base in m for m in models)
                if found:
                    logger.info(f"Ollama ready. Model: {CONFIG['ollama_model']}")
                else:
                    logger.warn(
                        f"{CONFIG['ollama_model']} not found. "
                        f"Run: ollama pull {CONFIG['ollama_model']}"
                    )
        except Exception:
            logger.warn("Ollama not running. Start with: ollama serve")

    def query(self, prompt: str, max_tokens: int = 1024) -> str:
        # Hard cap — allow enough tokens for multiple questions and points
        max_tokens = min(max_tokens, 32000)

        payload = {
            "model"  : CONFIG["ollama_model"],
            "prompt" : prompt,
            "stream" : False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
                "top_p"      : 0.9,
                "num_ctx"    : 32768,   # context window — keeps memory usage stable
            },
        }
        try:
            r = requests.post(
                CONFIG["ollama_url"],
                json    = payload,
                timeout = 600,         # 10 min — enough for any prompt on 8GB GPU
            )
            r.raise_for_status()
            response_text = r.json().get("response", "").strip()
            return response_text.replace('\x00', '')
        except KeyboardInterrupt:
            logger.warn("Query interrupted by user.")
            return ""
        except requests.exceptions.Timeout:
            logger.error("Ollama timeout after 10min. GPU may be overloaded.")
            return ""
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Run: ollama serve")
            return ""
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""

    @staticmethod
    def parse_json(raw: str, expect_dict: bool = False):
        """Safely extract and parse JSON from LLM response, stripping null bytes."""
        if not raw:
            return None

        def _clean_null_bytes(obj):
            if isinstance(obj, str):
                return obj.replace('\x00', '')
            elif isinstance(obj, dict):
                return {k: _clean_null_bytes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_null_bytes(item) for item in obj]
            return obj

        # clean = raw.replace("```json", "").replace("```", "").strip()
        clean = raw.replace('\x00', '').replace("```json", "").replace("```", "").strip()

        # Find outermost { } or [ ]
        for open_c, close_c in [("{", "}"), ("[", "]")]:
            start = clean.find(open_c)
            end   = clean.rfind(close_c)
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(clean[start: end + 1])
                    result = _clean_null_bytes(result)
                    if expect_dict and not isinstance(result, dict):
                        return None
                    return result
                except json.JSONDecodeError:
                    continue
        return None


# ══════════════════════════════════════════════════════════════════
#  MODULE 3 — ANSWER KEY PARSER
#  Question paper → full rubric for ALL questions
#  Answer key   → model answers + keywords for each question
# ══════════════════════════════════════════════════════════════════

class AnswerKeyParser:
    """
    Builds the complete rubric in two phases:

    Phase 1 — Question paper drives the structure:
        Extract every question with its number, marks, and type.
        This gives us the complete list of questions the student could answer.

    Phase 2 — Answer key fills in model answers + rubric points:
        For each question found, locate its model answer / marking scheme
        from the answer key OCR text.

    If no question paper is provided, falls back to extracting everything
    from the answer key text alone (old behaviour).
    """

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    # ── Phase 1: extract all questions from question paper ─────────
    def _parse_question_paper(self, question_paper_text: str) -> list:
        """Returns list of {q_num, question_text, marks, answer_type}"""
        if not question_paper_text or len(question_paper_text.strip()) < 20:
            return []

        text = question_paper_text[:15000]
        prompt = f"""You are an expert at parsing Indian university engineering exam question papers.

Question paper text:
\"\"\" 
{text}
\"\"\"

Extract EVERY SINGLE question from this exam paper.
Identify the question number, the exact question text, the marks allocated, and the type of answer required.

Respond ONLY with valid JSON — no markdown, no extra text:
{{
  "total_marks": 80,
  "questions": [
    {{"q_num": "1a", "question_text": "Explain the generalized concept of electric drive & state function of each block in the diagram?", "marks": 7, "answer_type": "paragraph"}},
    {{"q_num": "1b", "question_text": "Explain the power transistor with its characteristics?", "marks": 6, "answer_type": "paragraph"}},
    {{"q_num": "2a", "question_text": "Draw V-I characteristics of SCR and explain it?", "marks": 6, "answer_type": "diagram"}},
    {{"q_num": "2b", "question_text": "Explain the multi motor Drives & individuals drive used in industry?", "marks": 7, "answer_type": "paragraph"}}
  ]
}}

STRICT Rules:
- q_num format: ONLY digits + letter, lowercase, no spaces, no dots, no Q prefix
  CORRECT:  "1a"  "2b"  "9a"  "12b"
  WRONG:    "Q1a"  "1A"  "q.1a"  "1 a"
- Extract ALL questions — do not stop early
- marks: EXACT integer from the brackets e.g. (7) → 7, (6) → 6, (8) → 8
- total_marks is always 80 for this paper
- answer_type: "diagram" if question says "draw", "neat diagram", "sketch"; else "paragraph"; "numerical" for calculations; "mixed" for explain+draw
- question_text: exact wording from the paper
"""
        raw    = self.llm.query(prompt, max_tokens=5000)
        logger.info(f"Raw question paper LLM response: {raw[:1000]}") # Debug
        
        parsed = self.llm.parse_json(raw)
        result = []
        
        if parsed and "questions" in parsed and isinstance(parsed["questions"], list):
            # Perfect parse
            q_list = parsed["questions"]
        else:
            # Fallback: The model might have been truncated mid-JSON. Let's try to extract 
            # objects using regex since it's a very simple schema.
            logger.warn(f"Question paper JSON parse failed or truncated. Attempting regex salvage on raw: {raw[:500]}...")
            q_list = []
            
            # Look for blocks that look like {"q_num": "...", "question_text": "...", "marks": ..., "answer_type": "..."}
            # Find all potential question dictionaries in the string
            # We look for something that starts with { and has q_num inside it, up to the closing }
            blocks = re.findall(r'\{[^{}]*"q_num"[^{}]*\}', raw, re.IGNORECASE | re.DOTALL)
            
            for block in blocks:
                try:
                    # Clean up the block to make it valid JSON if needed
                    clean_block = block.strip()
                    # Some models put trailing commas
                    clean_block = re.sub(r',\s*\}', '}', clean_block)
                    q_obj = json.loads(clean_block)
                    if "q_num" in q_obj:
                        q_list.append(q_obj)
                except Exception as e:
                    logger.warn(f"Regex block parse failed: {e}. Block: {block[:50]}")

        if not q_list:
            logger.error("Could not extract any questions even with regex fallback.")
            return []
            
        seen   = set()
        for q in q_list:
            raw_num = str(q.get("q_num", "")).strip()
            # Aggressive normalisation: strip Q/q/./spaces, lowercase
            num = re.sub(r'[Qq][.\s]*', '', raw_num)
            num = re.sub(r'[\s.()\[\]]', '', num).lower()
            if not num or num in seen:
                continue
            seen.add(num)
            try:
                marks = int(float(str(q.get("marks", 7))))
            except Exception:
                marks = 7
            if marks <= 0:
                marks = 7
            result.append({
                "q_num"        : num,
                "question_text": q.get("question_text", "").strip(),
                "marks"        : marks,
                "answer_type"  : q.get("answer_type", "paragraph"),
            })
        logger.info(f"Question paper: {len(result)} questions → {[q['q_num'] for q in result]}")
        return result

    # ── Phase 2: extract model answers from answer key text ────────
    def _parse_model_answers(self, answer_key_text: str, questions: list) -> dict:
        """
        Returns {q_num: {model_answer, keywords, rubric_points, marking_scheme, diagram_description}}
        """
        if not answer_key_text or not questions:
            return {}

        q_list = "\n".join(
            f'  Q{q["q_num"]}: {q["question_text"][:80]} ({q["marks"]} marks)'
            for q in questions
        )
        text = answer_key_text[:18000]

        prompt = f"""You are an expert exam answer key analyser for Indian university engineering exams.

The following questions are in this exam:
{q_list}

Answer key / model answer text:
\"\"\"
{text}
\"\"\"

For EACH question listed above, extract the model answer from the text.
If a question's answer is not in the text, generate a brief model answer based on your knowledge.

Respond ONLY with valid JSON:
{{
  "answers": [
    {{
      "q_num": "9a",
      "model_answer": "The Hall effect is the phenomenon in which a voltage (Hall voltage) is generated across a current-carrying conductor or semiconductor when placed in a perpendicular magnetic field. Working: When a magnetic field is applied perpendicular to the current flow, a magnetic force acts on charge carriers pushing them to one side, creating a potential difference (Hall voltage). Applications: proximity sensor, speed sensor, position sensing, current sensing.",
      "keywords": ["Hall effect", "Hall voltage", "magnetic field", "charge carriers", "perpendicular", "transducer"],
      "rubric_points": ["Define Hall effect (1m)", "Explain working principle (3m)", "Hall voltage generation (1m)", "Applications (2m)"],
      "marking_scheme": "1 mark definition + 3 marks working + 2 marks applications",
      "diagram_expected": false,
      "diagram_description": ""
    }}
  ]
}}

Rules:
- q_num must match EXACTLY the q_num values listed above
- model_answer: comprehensive answer a top student should write
- keywords: 5-10 key technical terms the student MUST mention
- rubric_points: specific measurable points with marks breakdown
- If diagram expected (diagram/mixed type), set diagram_expected true and describe the diagram
"""
        raw    = self.llm.query(prompt, max_tokens=8000)
        parsed = self.llm.parse_json(raw)
        
        a_list = []
        if parsed and "answers" in parsed and isinstance(parsed["answers"], list):
            a_list = parsed["answers"]
        else:
            logger.warn("Model answer JSON parse failed or truncated. Attempting regex salvage...")
            # Find all potential answer dictionaries
            blocks = re.findall(r'\{[^{}]*"q_num"[^{}]*"model_answer"[^{}]*\}', raw, re.IGNORECASE | re.DOTALL)
            for block in blocks:
                try:
                    clean_block = block.strip()
                    clean_block = re.sub(r',\s*\}', '}', clean_block)
                    a_obj = json.loads(clean_block)
                    if "q_num" in a_obj:
                        a_list.append(a_obj)
                except Exception as e:
                    logger.warn(f"Regex block parse failed in model answers: {e}")

        if not a_list:
            logger.warn("Model answer extraction completely failed.")
            return {}

        result = {}
        for a in a_list:
            raw_num = str(a.get("q_num", "")).strip()
            num = re.sub(r'[Qq][.\s]*', '', raw_num)
            num = re.sub(r'[\s.]', '', num).lower()
            if num:
                result[num] = {
                    "model_answer"       : a.get("model_answer", ""),
                    "keywords"           : a.get("keywords", []),
                    "rubric_points"      : a.get("rubric_points", []),
                    "marking_scheme"     : a.get("marking_scheme", ""),
                    "diagram_expected"   : bool(a.get("diagram_expected", False)),
                    "diagram_description": a.get("diagram_description", ""),
                }
        logger.info(f"Model answers extracted for: {sorted(result.keys())}")
        return result

    # ── Main parse entry point ─────────────────────────────────────
    def parse(self, ocr_result: dict, question_paper_text: str = "") -> dict:
        """
        ocr_result: OCR output of the answer key image/PDF
        question_paper_text: raw text of the question paper (strongly recommended)

        Returns: {"questions": {q_num: {...}}, "total_exam_marks": int}
        """
        logger.info("Building rubric...")

        answer_key_text = ocr_result.get("combined_text", "")

        # ── Phase 1: get all questions from question paper ──────────
        qp_questions = []
        total_marks  = 80
        if question_paper_text and question_paper_text.strip():
            qp_questions = self._parse_question_paper(question_paper_text)
            if qp_questions:
                total_marks = sum(q["marks"] for q in qp_questions)

        # ── Fallback: extract questions from answer key text itself ─
        if not qp_questions:
            logger.warn("No question paper — extracting questions from answer key text only.")
            qp_questions = self._parse_question_paper(answer_key_text)

        if not qp_questions:
            logger.error("Could not extract any questions. Rubric will be empty.")
            return {"questions": {}, "total_exam_marks": 80}

        # ── Phase 2: extract model answers ─────────────────────────
        # Combine question paper + answer key for best coverage
        combined_for_answers = ""
        if question_paper_text:
            combined_for_answers += "QUESTIONS:\n" + question_paper_text + "\n\n"
        combined_for_answers += "MODEL ANSWERS:\n" + answer_key_text
        model_answers = self._parse_model_answers(combined_for_answers, qp_questions)

        # ── Build final rubric ──────────────────────────────────────
        rubric = {}
        for q in qp_questions:
            num = q["q_num"]
            ma  = model_answers.get(num, {})
            rubric[num] = {
                "question"           : q["question_text"],
                "answer_type"        : q["answer_type"],
                "model_answer"       : ma.get("model_answer", ""),
                "diagram_expected"   : ma.get("diagram_expected", False),
                "diagram_description": ma.get("diagram_description", ""),
                "max_marks"          : q["marks"],
                "keywords"           : ma.get("keywords", []),
                "rubric_points"      : ma.get("rubric_points", []),
                "marking_scheme"     : ma.get("marking_scheme", ""),
            }

        logger.info(f"Rubric complete: {len(rubric)} questions, {total_marks} total marks")
        for num, r in rubric.items():
            logger.info(f"  Q{num} [{r['answer_type']}] [{r['max_marks']}m]: {r['question'][:55]}...")

        logger.log("rubric_built", {"num_questions": len(rubric), "total_marks": total_marks})
        return {"questions": rubric, "total_exam_marks": total_marks}


# ══════════════════════════════════════════════════════════════════
#  MODULE 3b — QUESTION PAPER PARSER
#  Converts the question-paper text into a lookup dict so the
#  segmenter can attach the real question text to each answer.
# ══════════════════════════════════════════════════════════════════

class QuestionPaperParser:
    """
    Parses the question paper (OCR text or raw text) into:
        { "9a": "Explain the working of Hall effect sensor.",
          "9b": "Explain the working principle of AC Tachogenerator.",
          "6b": "Explain principle and working of Universal Motor?",
          ... }

    This lookup is used by AnswerSegmenter to:
      1. Confirm which question the student answered
      2. Provide the real question text to the evaluator
    """

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def parse(self, question_paper_text: str) -> dict:
        """
        Returns: { q_num_str: question_text_str, ... }
        e.g.     { "9a": "Explain Hall effect sensor.", "9b": "...", ... }
        """
        if not question_paper_text or not question_paper_text.strip():
            return {}

        logger.info("Parsing question paper into question lookup...")

        # Feed at most 12000 chars to the LLM to stay inside context window
        text = question_paper_text[:12000]

        prompt = f"""You are an expert at parsing Indian university engineering exam question papers.
Below is the OCR-extracted text of an exam question paper.

Extract EVERY question with its question number.
The question numbers are like: Q.1 a), Q.1 b), Q.2 a), ... or written as 9a, 9b, 10a etc.
Each main question (e.g. Q.9) usually has two sub-parts: a) and b).

Question paper text:
\"\"\"
{text}
\"\"\"

Respond ONLY with valid JSON — no extra text, no markdown fences.

{{
  "questions": [
    {{
      "q_num": "9a",
      "question_text": "Explain the working of Hall effect Sensor?",
      "marks": 6
    }},
    {{
      "q_num": "9b",
      "question_text": "Explain the working principle of AC Tachogenerator?",
      "marks": 7
    }}
  ]
}}

Rules:
- q_num must be lowercase letters + digits only, e.g. "9a", "9b", "6b", "12a"
- Include ALL questions found, even if they appear after an OR
- question_text must be the EXACT question wording as written in the paper
- marks is the integer mark value shown next to the question (e.g. (6) means 6)
- If marks not visible, use 7 as default
"""

        raw    = self.llm.query(prompt, max_tokens=3000)
        parsed = self.llm.parse_json(raw)

        lookup = {}
        if parsed and "questions" in parsed:
            for item in parsed["questions"]:
                q_num = str(item.get("q_num", "")).strip().lower()
                q_text = item.get("question_text", "").strip()
                marks  = item.get("marks", 7)
                if q_num and q_text:
                    lookup[q_num] = {
                        "question_text": q_text,
                        "marks": marks,
                    }
            logger.info(f"Question paper parsed: {len(lookup)} questions found → {sorted(lookup.keys())}")
        else:
            logger.warn("Question paper parse failed — question lookup will be empty.")

        return lookup


# ══════════════════════════════════════════════════════════════════
#  MODULE 4 — STUDENT ANSWER SEGMENTER
# ══════════════════════════════════════════════════════════════════

class AnswerSegmenter:
    """
    Splits student OCR output into per-question answers.
    Handles diagrams, partial answers, skipped questions.
    """

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def segment(self, ocr_result: dict, rubric: dict,
                question_paper_lookup: dict = None) -> dict:
        """
        Returns: {q_num: {text_answer, diagram_description, is_blank,
                           resolved_question_text}}

        Strategy:
          1. Use anchored_answers from OCR (student wrote Q.9a in the column)
          2. Cross-reference each detected q_num with question_paper_lookup
             so the evaluator gets the real question text, not a guessed one.
          3. Fallback: ask Ollama LLM to segment from raw combined text.
        """
        question_paper_lookup = question_paper_lookup or {}
        logger.info("Segmenting student answers...")

        combined   = ocr_result.get("combined_text", "")
        # Protect LLM fallback segmentation from excessively large text (e.g. 17+ pages)
        if len(combined) > 8000:
            logger.warn(f"Truncating student sheet text from {len(combined)} to 8000 chars for segmentation.")
            combined = combined[:8000]

        diagrams   = ocr_result.get("diagram_descriptions", "")
        boundaries = ocr_result.get("boundaries", "")

        # ── PRIMARY: use explicit question-number anchors detected by OCR ───
        anchored = ocr_result.get("anchored_answers", {})
        if anchored:
            answers = {}

            def _norm(k):
                """Normalise a question key to digits+letter, lowercase.
                   'Q.9(A)' 'Q9a' '9A' '9 a' 'q.9' → '9a'
                   'Q9'     '9'               → '9'
                """
                k = str(k).strip()
                k = re.sub(r'[Qq][.\s]*', '', k)        # strip leading Q/q
                k = re.sub(r'[\s.()\[\]]', '', k)        # strip spaces, dots, parens
                return k.lower()

            # Build a normalised lookup of rubric keys so matching is fast
            rubric_norm = {_norm(rk): rk for rk in rubric}

            for raw_q, data in anchored.items():
                norm_raw = _norm(raw_q)
                # 1. exact normalised match
                matched_key = rubric_norm.get(norm_raw)
                if matched_key is None:
                    # 2. substring match: rubric key contained in anchor or vice versa
                    for nk, rk in rubric_norm.items():
                        if norm_raw in nk or nk in norm_raw:
                            matched_key = rk
                            break
                if matched_key is None:
                    # 3. keep unmatched — never throw away student work
                    matched_key = raw_q
                    logger.warn(f"  Anchor '{raw_q}' (norm='{norm_raw}') not in rubric. "
                                f"Kept as-is. Rubric keys: {sorted(rubric_norm.keys())}")
                else:
                    logger.info(f"  Anchor matched: '{raw_q}' → Q{matched_key}")

                # Merge: if same key already seen (multi-page), append text
                if matched_key in answers:
                    existing = answers[matched_key]
                    add_text = data.get("text_answer", "").strip()
                    if add_text:
                        existing["text_answer"] = (
                            existing.get("text_answer", "") + "\n" + add_text
                        ).strip()
                    existing["is_blank"] = not bool(existing["text_answer"])
                else:
                    answers[matched_key] = data
                    # Recompute is_blank from actual text — never trust OCR's is_blank alone
                    answers[matched_key]["is_blank"] = not bool(
                        data.get("text_answer", "").strip()
                    )

            if answers:
                # Validate that at least one extracted anchor actually exists in the rubric.
                # If all anchors are bogus (e.g. "12" from a cover page), we must fallback.
                valid_anchors = [k for k in answers if k in rubric]
                
                if valid_anchors:
                    logger.info(f"Anchored segmentation: found Q{sorted(answers.keys())} (valid: {valid_anchors})")
                    # ── Cross-reference with question paper ─────────────────────
                    # For each detected question number, look up the real question
                    # text from the question paper and attach it to the answer block.
                    if question_paper_lookup:
                        for q_key, ans_data in answers.items():
                            norm_key = re.sub(r'[\s.]', '', str(q_key)).lower()
                            # Try direct match first
                            qp_entry = question_paper_lookup.get(norm_key)
                            if not qp_entry:
                                # Fuzzy: try partial matches (e.g. "9a" matches "q9a")
                                for lk, lv in question_paper_lookup.items():
                                    if norm_key in lk or lk in norm_key:
                                        qp_entry = lv
                                        break
                            if qp_entry:
                                ans_data["resolved_question_text"] = qp_entry["question_text"]
                                ans_data["question_paper_marks"]   = qp_entry.get("marks", None)
                                logger.info(
                                    f"  Cross-ref Q{q_key} → '{qp_entry['question_text'][:60]}...'"
                                )
                            else:
                                ans_data["resolved_question_text"] = ""
                                logger.warn(
                                    f"  Cross-ref Q{q_key}: no match in question paper lookup."
                                )
                    return answers
                else:
                    logger.warn(f"Anchors found {list(answers.keys())}, but NONE match the rubric. Falling back to LLM segmentation.")

        # ── FALLBACK: ask Ollama to segment from raw text ────────────────
        logger.warn("No anchors found in OCR — falling back to LLM segmentation.")

        q_list = "\n".join(
            f"  Q{num}: [{r['answer_type']}] {r['question'][:70]}"
            for num, r in rubric.items()
        )

        prompt = f"""You are an expert at reading student exam answer sheets.
Below is OCR text from a student's answer sheet (Hindi, English, or both).
Split it into individual answers for each question.

Questions in this exam:
{q_list}

Student answer sheet text:
\"\"\"
{combined}
\"\"\"

Diagram descriptions found in student's sheet:
\"\"\"
{diagrams if diagrams else "None"}
\"\"\"

Answer boundaries detected:
\"\"\"
{boundaries if boundaries else "Not detected — infer from content"}
\"\"\"

Respond ONLY with valid JSON — no extra text, no markdown.

{{
  "answers": [
    {{
      "q_num": "9a",
      "text_answer": "<student's complete written answer — preserve original wording and mistakes>",
      "diagram_description": "<if student drew a diagram: describe what they drew, what labels they included, if it looks correct>",
      "is_blank": false,
      "answer_quality_hint": "complete"
    }}
  ]
}}

Rules:
- CRITICAL: Extract answers for ALL questions attempted. Do not stop at the first answer.
- These are university engineering exam questions worth 6-7 marks each.
- Preserve student's original wording EXACTLY including spelling mistakes
- For blank answers set is_blank to true and text_answer to "" (ONLY IF the student wrote absolutely nothing. Even a single letter or word is NOT blank.)
- For diagram answers describe what the student drew in detail
- Preserve Hindi text exactly
- If the student answered Q5a and Q5b separately, treat them both as part of Q5's answer
- If you cannot find an answer for a question, still include it in the JSON with is_blank: true
"""

        raw    = self.llm.query(prompt, max_tokens=4000)
        parsed = self.llm.parse_json(raw)

        a_list = []
        if parsed and "answers" in parsed and isinstance(parsed["answers"], list):
            a_list = parsed["answers"]
        else:
            logger.warn("Student segment JSON parse failed or truncated. Attempting regex salvage...")
            # Find all potential answer dictionaries
            blocks = re.findall(r'\{[^{}]*"q_num"[^{}]*"text_answer"[^{}]*\}', raw, re.IGNORECASE | re.DOTALL)
            for block in blocks:
                try:
                    clean_block = block.strip()
                    clean_block = re.sub(r',\s*\}', '}', clean_block)
                    a_obj = json.loads(clean_block)
                    if "q_num" in a_obj:
                        a_list.append(a_obj)
                except Exception as e:
                    logger.warn(f"Regex block parse failed in student segment: {e}")

        if not a_list:
            logger.warn("Segmentation JSON parse failed — using heuristic text split fallback.")
            rubric_keys = [str(k) for k in rubric.keys()] if rubric else []

            if not rubric_keys:
                return {}

            fallback_answers = {}

            # Layer 1: try to find question anchors in the raw combined text
            # e.g. "9a:", "Q9a:", "10b:" etc.
            try:
                anchor_pattern = re.compile(
                    r'(?:^|\n)\s*(?:[Qq]\.?\s*)?(' +
                    '|'.join(re.escape(k) for k in sorted(rubric_keys, key=len, reverse=True)) +
                    r')\s*[:\-\)]\s*',
                    re.IGNORECASE
                )
                parts = anchor_pattern.split(combined)
                # parts layout: [pre_text, q_key1, text1, q_key2, text2, ...]
                if len(parts) >= 3:
                    i = 1
                    while i + 1 < len(parts):
                        q_key = parts[i].strip().lower()
                        q_text = parts[i + 1].strip()
                        matched = next((k for k in rubric_keys if k.lower() == q_key), q_key)
                        fallback_answers[matched] = {
                            "text_answer"        : q_text,
                            "diagram_description": "",
                            "is_blank"           : not bool(q_text),
                            "answer_quality_hint": "heuristic_anchor",
                        }
                        i += 2
            except Exception as e:
                logger.warn(f"Heuristic anchor split failed: {e}")

            if fallback_answers:
                logger.info(f"Heuristic anchor fallback found: Q{sorted(fallback_answers.keys())}")
                return fallback_answers

            # Layer 2: even split — better than giving everything to Q1
            logger.warn("No anchors found either — splitting text evenly across all rubric questions.")
            n = len(rubric_keys)
            chunk_size = max(1, len(combined) // n)
            for idx, q_key in enumerate(rubric_keys):
                chunk_text = combined[idx * chunk_size : (idx + 1) * chunk_size].strip()
                fallback_answers[q_key] = {
                    "text_answer"        : chunk_text,
                    "diagram_description": diagrams if idx == 0 else "",
                    "is_blank"           : not bool(chunk_text),
                    "answer_quality_hint": "even_split_fallback",
                }
            return fallback_answers


        answers = {}
        for item in a_list:
            num = item.get("q_num")
            if num is not None:
                norm_num = re.sub(r'[\s.]', '', str(num)).lower()
                resolved_q = ""
                qp_marks   = None
                if question_paper_lookup:
                    qp_entry = question_paper_lookup.get(norm_num)
                    if not qp_entry:
                        for lk, lv in question_paper_lookup.items():
                            if norm_num in lk or lk in norm_num:
                                qp_entry = lv
                                break
                    if qp_entry:
                        resolved_q = qp_entry["question_text"]
                        qp_marks   = qp_entry.get("marks")
                        logger.info(f"  Cross-ref (fallback) Q{num} → '{resolved_q[:60]}...'")

                answers[str(num).strip()] = {   # always store as str to match rubric keys
                    "text_answer"           : item.get("text_answer", "").strip(),
                    "diagram_description"   : item.get("diagram_description", "").strip(),
                    "is_blank"              : bool(item.get("is_blank", False)),
                    "answer_quality_hint"   : item.get("answer_quality_hint", ""),
                    "resolved_question_text": resolved_q,
                    "question_paper_marks"  : qp_marks,
                }

        logger.info(f"Segmented: answers found for Q{sorted(answers.keys())}")
        return answers


# ══════════════════════════════════════════════════════════════════
#  MODULE 5 — LLM REFERENCE ANSWER GENERATOR
#  YOUR IDEA: LLM generates its own answer for dual-reference eval
# ══════════════════════════════════════════════════════════════════

class LLMReferenceGenerator:
    """
    Generates the LLM's own ideal answer to each question.
    This becomes the second reference in dual-reference evaluation.
    Handles all question types including diagrams.
    """

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def generate(self, question: str, answer_type: str,
                 max_marks: int, subject_hint: str = "") -> dict:
        """
        Returns: {llm_answer, llm_diagram_description, generation_successful}
        """
        diagram_instruction = ""
        if answer_type in ("diagram", "mixed"):
            diagram_instruction = """
Also describe the ideal diagram for this answer:
- What the diagram should show
- All labels that should be present
- All arrows, connections, components
- Any key features the examiner would look for
"""

        prompt = f"""You are an expert teacher for government examinations in India.
A student must answer the following exam question for {max_marks} marks.
{"Subject context: " + subject_hint if subject_hint else ""}

Write the ideal complete answer that a top student should write.
Write it the way a student would write — not too academic, not too simple.
Match the depth to the marks: {max_marks} marks = more detail needed.

Question: {question}
Answer type: {answer_type}
{diagram_instruction}

Respond ONLY with valid JSON — no extra text.

{{
  "llm_answer": "<complete ideal answer text in the most appropriate language>",
  "llm_diagram_description": "<if diagram required: describe the ideal diagram in detail, else empty string>",
  "key_concepts_covered": ["<concept 1>", "<concept 2>", "..."],
  "generation_successful": true
}}"""

        raw    = self.llm.query(prompt, max_tokens=2000)
        parsed = self.llm.parse_json(raw, expect_dict=True)

        if not parsed:
            logger.warn(f"LLM reference generation failed for: {question[:50]}")
            return {
                "llm_answer"              : "",
                "llm_diagram_description" : "",
                "key_concepts_covered"    : [],
                "generation_successful"   : False,
            }

        logger.info(f"LLM reference generated for Q: {question[:40]}...")
        return parsed


# ══════════════════════════════════════════════════════════════════
#  MODULE 6 — DUAL REFERENCE SEMANTIC EVALUATOR
#  YOUR IDEA: student answer vs examiner answer + LLM answer
#  Works for ALL question types including diagrams
# ══════════════════════════════════════════════════════════════════

class DualReferenceEvaluator:
    """
    Evaluates student answer against TWO references:
      1. Examiner's model answer (official ground truth)
      2. LLM's own generated answer (fair second opinion)

    The self-attention mechanism inside Qwen2.5 reads all three
    texts in one context window and computes cross-relationships:
      - student words ↔ examiner words
      - student words ↔ LLM answer words
      - what concepts are present vs missing

    Handles: paragraph, short, diagram, mixed, definition, numerical
    """

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def evaluate(
        self,
        student_answer      : str,
        student_diagram     : str,
        examiner_answer     : str,
        examiner_diagram    : str,
        llm_answer          : str,
        llm_diagram         : str,
        question            : str,
        answer_type         : str,
        rubric_points       : list,
        marking_scheme      : str,
        max_marks           : int,
    ) -> dict:

        # Protect Ollama from massive context explosion
        if len(student_answer) > 6000:
            logger.warn(f"Truncating student answer from {len(student_answer)} to 6000 chars for evaluation.")
            student_answer = student_answer[:6000] + "\n...[TRUNCATED]"

        rubric_str = "\n".join(f"  - {p}" for p in rubric_points)

        # ── Build diagram section only if relevant ─────────────────
        diagram_section = ""
        if answer_type in ("diagram", "mixed") or student_diagram or examiner_diagram:
            diagram_section = f"""
DIAGRAM EVALUATION:
  Examiner's expected diagram: {examiner_diagram or "Not specified"}
  LLM's ideal diagram        : {llm_diagram or "Not specified"}
  Student's drawn diagram    : {student_diagram or "No diagram drawn by student"}
"""

        prompt = f"""You are a highly experienced and fair exam evaluator for government examinations in India.

Evaluate the student's answer using BOTH reference answers below, with STRICT EMPHASIS on the RUBRIC POINTS (Marking Scheme).
If the student's answer contains the specific points mentioned in the RUBRIC POINTS, they MUST be awarded those marks.
Do NOT penalize students for correct answers written in different wording.
Be especially fair for Hindi answers — meaning matters more than exact words.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION: {question}
ANSWER TYPE: {answer_type}
TOTAL MARKS: {max_marks}
MARKING SCHEME: {marking_scheme or "Standard marking"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REFERENCE 1 — Examiner's official answer:
{examiner_answer or "Not provided"}

REFERENCE 2 — Ideal answer (independent assessment):
{llm_answer or "Not provided"}

STUDENT'S ANSWER:
{student_answer or "(No written answer)"}
{diagram_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUBRIC POINTS / MARKING SCHEME ({max_marks} marks total):
{rubric_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond ONLY with valid JSON — no extra text, no markdown.

{{
  "similarity_vs_examiner"  : <float 0.0-1.0, how well student matches reference 1>,
  "similarity_vs_llm"       : <float 0.0-1.0, how well student matches reference 2>,
  "final_semantic_similarity": <float 0.0-1.0, take the highest score based on rubric point coverage>,
  "suggested_marks"         : <integer 0-{max_marks}, strictly based on rubric coverage/marking scheme>,
  "rubric_coverage": {{
    "points_covered": ["<rubric point student addressed>"],
    "points_missed" : ["<rubric point student missed>"]
  }},
  "diagram_evaluation": {{
    "diagram_present"    : <true/false — did student draw a diagram>,
    "diagram_correct"    : <true/false/null — is it correct>,
    "diagram_marks"      : <integer marks for diagram portion, 0 if not applicable>,
    "diagram_feedback"   : "<what was correct/incorrect in the diagram>"
  }},
  "what_student_got_right" : "<specific correct points — be encouraging>",
  "what_student_missed"    : "<specific missing points — be constructive>",
  "feedback"               : "<2-3 sentence constructive feedback in the SAME language as student's answer>",
  "confidence"             : "<high|medium|low — your confidence in this evaluation>",
  "answer_completeness"    : "<complete|partial|minimal|blank>"
}}

Strict rules:
- suggested_marks must be between 0 and {max_marks}
- If student is blank (i.e., NO answer at all): all scores = 0
- If answer type is diagram and no diagram drawn: deduct accordingly
- Be fair — a correct concept expressed differently still deserves marks
- For very short answers (MCQs, fill-in-the-blanks, true/false, one-word), exact or semantic matches get full marks. Do NOT penalize for being brief.
- Feedback must be in the same language the student wrote in
"""

        raw    = self.llm.query(prompt, max_tokens=2000)
        parsed = self.llm.parse_json(raw)

        if not parsed:
            logger.warn("Dual reference evaluation parse failed.")
            return self._fallback(max_marks, rubric_points)

        # ── Clamp and validate all numeric values ──────────────────
        for key in ("similarity_vs_examiner", "similarity_vs_llm", "final_semantic_similarity"):
            parsed[key] = float(max(0.0, min(1.0, parsed.get(key, 0.0))))

        parsed["suggested_marks"] = int(
            max(0, min(max_marks, parsed.get("suggested_marks", 0)))
        )

        logger.log("evaluation_done", {
            "question_preview"   : question[:60],
            "suggested_marks"    : parsed["suggested_marks"],
            "max_marks"          : max_marks,
            "semantic_sim"       : parsed["final_semantic_similarity"],
            "confidence"         : parsed.get("confidence", "?"),
            "completeness"       : parsed.get("answer_completeness", "?"),
        })

        return parsed

    @staticmethod
    def _fallback(max_marks: int, rubric_points: list) -> dict:
        return {
            "similarity_vs_examiner"  : 0.0,
            "similarity_vs_llm"       : 0.0,
            "final_semantic_similarity": 0.0,
            "suggested_marks"         : 0,
            "rubric_coverage"         : {"points_covered": [], "points_missed": rubric_points},
            "diagram_evaluation"      : {
                "diagram_present" : False,
                "diagram_correct" : None,
                "diagram_marks"   : 0,
                "diagram_feedback": "Evaluation failed.",
            },
            "what_student_got_right"  : "",
            "what_student_missed"     : "Evaluation could not be completed.",
            "feedback"                : "Evaluation failed — please re-run.",
            "confidence"              : "low",
            "answer_completeness"     : "unknown",
            "parse_error"             : True,
        }


# ══════════════════════════════════════════════════════════════════
#  MODULE 7 — KEYWORD MATCHER
# ══════════════════════════════════════════════════════════════════

class KeywordMatcher:
    """
    Two-layer keyword matching:
      Layer 1: Direct string match (fast, exact)
      Layer 2: KeyBERT semantic fuzzy match (catches paraphrases)
    Handles Hindi + English.
    CPU only.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model:
            return
        try:
            from keybert import KeyBERT
            self._model = KeyBERT(model=CONFIG["embedding_model"])
            logger.info("KeyBERT loaded.")
        except ImportError:
            logger.warn("keybert not installed. Run: pip install keybert sentence-transformers")

    def score(self, student_text: str, keywords: list, max_marks: int) -> dict:
        if not student_text or not keywords:
            return {
                "keyword_score"   : 0,
                "keywords_found"  : [],
                "keywords_missing": keywords,
                "match_ratio"     : 0.0,
            }

        # Protect KeyBERT/Embedder from null bytes and excessive length
        student_text = student_text.replace('\x00', '')
        if len(student_text) > 4000:
            student_text = student_text[:4000]

        self._load()
        found        = []
        lower_text   = student_text.lower()

        # Layer 1 — direct match
        for kw in keywords:
            if kw.lower() in lower_text:
                found.append(kw)

        # Layer 2 — KeyBERT fuzzy match
        if self._model and student_text.strip():
            try:
                extracted = self._model.extract_keywords(
                    student_text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words=None,
                    top_n=20,
                )
                for ext_kw, confidence in extracted:
                    if confidence < 0.3:
                        continue
                    for rubric_kw in keywords:
                        if rubric_kw not in found:
                            if (ext_kw.lower() in rubric_kw.lower() or
                                    rubric_kw.lower() in ext_kw.lower()):
                                found.append(rubric_kw)
            except Exception as e:
                logger.warn(f"KeyBERT error: {e}")

        found   = list(set(found))
        ratio   = len(found) / len(keywords) if keywords else 0.0
        score   = round(min(ratio * max_marks, max_marks), 2)

        return {
            "keyword_score"   : score,
            "keywords_found"  : found,
            "keywords_missing": [k for k in keywords if k not in found],
            "match_ratio"     : round(ratio, 3),
        }


# ══════════════════════════════════════════════════════════════════
#  MODULE 8 — SCORE FUSION
# ══════════════════════════════════════════════════════════════════

def fuse_scores(
    semantic_result : dict,
    keyword_result  : dict,
    max_marks       : int,
    answer_type     : str,
) -> dict:
    """
    Combines dual-reference semantic score + keyword score.
    Adjusts weights based on answer type.
    """
    # For diagram answers — keyword match matters less
    if answer_type == "diagram":
        alpha, beta = 0.8, 0.2
    elif answer_type == "definition":
        alpha, beta = 0.4, 0.6   # definitions need exact terms
    elif answer_type == "numerical":
        alpha, beta = 0.7, 0.3
    else:
        alpha = CONFIG["alpha"]
        beta  = CONFIG["beta"]

    semantic_sim   = semantic_result.get("final_semantic_similarity", 0.0)
    semantic_marks = semantic_sim * max_marks
    keyword_marks  = keyword_result.get("keyword_score", 0)
    llm_suggested  = semantic_result.get("suggested_marks", 0)

    fused = round(
        min((alpha * semantic_marks) + (beta * keyword_marks), max_marks),
        2,
    )

    # Trust LLM suggestion if it's within 1 mark of formula
    final = int(round(
        llm_suggested if abs(fused - llm_suggested) <= 1 else fused
    ))
    final = max(0, min(max_marks, final))

    return {
        "alpha_used"           : alpha,
        "beta_used"            : beta,
        "semantic_marks"       : round(semantic_marks, 2),
        "keyword_marks"        : round(keyword_marks, 2),
        "fused_marks"          : fused,
        "llm_suggested_marks"  : llm_suggested,
        "final_suggested_marks": final,
    }


# ══════════════════════════════════════════════════════════════════
#  MODULE 9 — CHEATING DETECTOR
# ══════════════════════════════════════════════════════════════════

class CheatingDetector:
    """
    Multilingual sentence embeddings + cosine similarity.
    Flags suspicious answer pairs across all students.
    Also detects if student answer is too similar to model answer
    (possible copying of answer key).
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(CONFIG["embedding_model"])
            logger.info("Cheating detector embedding model loaded.")
        except ImportError:
            logger.warn("sentence-transformers not installed.")

    def analyze(self, all_answers: dict, rubric: dict) -> dict:
        """
        all_answers: {student_id: {q_num: {text_answer, ...}}}
        Returns: {suspicious_pairs, answer_key_copying}
        """
        self._load()
        if not self._model:
            return {"suspicious_pairs": {}, "answer_key_copying": {}}

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        suspicious_pairs  = {}
        answer_key_copying = {}
        student_ids       = list(all_answers.keys())
        all_q_nums        = set()

        for a in all_answers.values():
            all_q_nums.update(a.keys())

        for q_num in sorted(all_q_nums):
            texts = []
            ids   = []
            for sid in student_ids:
                ans = all_answers[sid].get(q_num, {})
                text = ans.get("text_answer", "").strip()
                if text:
                    texts.append(text)
                    ids.append(sid)

            if len(texts) < 2:
                continue

            # Remove null bytes before encoding to prevent C-extension crash
            texts = [t.replace('\x00', '') for t in texts]
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            matrix     = cos_sim(embeddings)
            pairs      = []

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    sim = float(matrix[i][j])
                    if sim >= CONFIG["similarity_threshold"]:
                        pairs.append({
                            "student_1" : ids[i],
                            "student_2" : ids[j],
                            "similarity": round(sim, 4),
                            "severity"  : (
                                "CRITICAL" if sim >= 0.97 else
                                "HIGH"     if sim >= 0.93 else
                                "MEDIUM"   if sim >= 0.90 else
                                "LOW"
                            ),
                        })

            if pairs:
                # Build a human-readable reason for each pair
                for p in pairs:
                    s1 = all_answers[p["student_1"]].get(q_num, {}).get("text_answer", "")[:80]
                    s2 = all_answers[p["student_2"]].get(q_num, {}).get("text_answer", "")[:80]
                    p["reason"] = (
                        f"Cosine similarity = {p['similarity']:.2%}. "
                        f"Both students wrote nearly identical answers for Q{q_num}. "
                        f"Student 1 wrote: '{s1}...' "
                        f"Student 2 wrote: '{s2}...'"
                    )
                suspicious_pairs[f"Q{q_num}"] = pairs

            # Check if student answer matches model answer too closely
            model_ans = rubric.get(q_num, {}).get("model_answer", "")
            if model_ans:
                model_emb = self._model.encode([model_ans], convert_to_numpy=True)
                for i, sid in enumerate(ids):
                    sim = float(cos_sim([embeddings[i]], model_emb)[0][0])
                    if sim >= 0.95:
                        key = f"Q{q_num}"
                        if key not in answer_key_copying:
                            answer_key_copying[key] = []
                        answer_key_copying[key].append({
                            "student_id": sid,
                            "similarity": round(sim, 4),
                            "note"      : "Answer suspiciously similar to model answer",
                        })

        logger.log("cheating_detection_done", {
            "suspicious_pairs" : {k: len(v) for k, v in suspicious_pairs.items()},
            "answer_key_copying": {k: len(v) for k, v in answer_key_copying.items()},
        })

        return {
            "suspicious_pairs"  : suspicious_pairs,
            "answer_key_copying": answer_key_copying,
        }

    def unload(self):
        if self._model:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Cheating detector VRAM freed.")


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

class ExamPipeline:
    """
    Complete commercial-grade exam evaluation pipeline.

    Production usage:
        pipeline = ExamPipeline()

        # Step A — examiner uploads answer key image
        rubric = pipeline.process_answer_key("answer_key.jpg")

        # Step B — each student uploads their sheet
        pipeline.evaluate_student("STU001", rubric, image_path="stu001.jpg")
        pipeline.evaluate_student("STU002", rubric, image_path="stu002.jpg")

        # Step C — cheating detection across all students
        cheat_report = pipeline.detect_cheating(rubric)

        # Step D — save everything
        pipeline.save_results(cheat_report)
    """

    def __init__(self):
        logger.info("Initializing ExamPipeline...")
        self.ocr         = GroqVLM()
        self.llm         = OllamaLLM()
        self.key_parser  = AnswerKeyParser(self.llm)
        self.q_parser    = QuestionPaperParser(self.llm)   # ← NEW
        self.segmenter   = AnswerSegmenter(self.llm)
        self.ref_gen     = LLMReferenceGenerator(self.llm)
        self.evaluator   = DualReferenceEvaluator(self.llm)
        self.kw_matcher  = KeywordMatcher()
        self.cheat_det   = CheatingDetector()

        self._all_student_answers  = {}
        self._all_results          = []
        self._rubric               = {}
        self._llm_references       = {}   # cache LLM answers per question
        self._question_paper_lookup = {}  # ← NEW: {q_num → {question_text, marks}}

        logger.info("Pipeline ready.")

    # ─────────────────────────────────────────────────────────────
    #  STEP A — Process examiner's answer key
    # ─────────────────────────────────────────────────────────────
    def process_answer_key(
        self,
        image_path          : str = None,
        raw_text            : str = None,
        question_paper_text : str = None,   # ← NEW: text from the question paper PDF
    ) -> dict:
        print("\n" + "═" * 60)
        print("  STEP A — Processing Answer Key")
        print("═" * 60)

        if image_path:
            logger.info(f"Answer key image: {image_path}")
            ocr_result = self.ocr.extract(image_path, mode="answer_key")
        elif raw_text is not None:
            ocr_result = {
                "combined_text"       : raw_text,
                "english_text"        : raw_text,
                "hindi_text"          : "",
                "diagram_descriptions": "",
                "boundaries"          : "",
                "full_response"       : raw_text,
            }
        else:
            raise ValueError("Provide image_path or raw_text for answer key.")

        if not ocr_result.get("combined_text", "").strip():
            logger.error("No text extracted from answer key.")
            return {}

        # ── Parse question paper into lookup (if provided) ─────────────────
        if question_paper_text and question_paper_text.strip():
            print("\n  Parsing question paper for cross-reference lookup...")
            self._question_paper_lookup = self.q_parser.parse(question_paper_text)
            print(f"  Question paper lookup: {len(self._question_paper_lookup)} questions indexed.")
            print(f"  Questions indexed: {sorted(self._question_paper_lookup.keys())}")
        else:
            self._question_paper_lookup = {}
            logger.warn("No question paper text provided — cross-reference disabled.")

        rubric = self.key_parser.parse(ocr_result, question_paper_text=question_paper_text or "")

        # ── Patch rubric with question paper data ──────────────────
        # If question paper lookup has marks or better question text,
        # overwrite the rubric — question paper is the ground truth.
        if self._question_paper_lookup:
            questions_dict = rubric.get("questions", {})
            for q_num, q_data in questions_dict.items():
                qp = self._question_paper_lookup.get(q_num)
                if not qp:
                    # try norm match
                    norm = re.sub(r'[Qq.\s]', '', str(q_num)).lower()
                    for lk, lv in self._question_paper_lookup.items():
                        if norm == re.sub(r'[Qq.\s]', '', lk).lower():
                            qp = lv
                            break
                if qp:
                    qp_marks = int(qp.get("marks", 0) or 0)
                    if qp_marks > 0:
                        q_data["max_marks"] = qp_marks
                        logger.info(f"  Q{q_num}: marks patched from question paper → {qp_marks}")
                    if qp.get("question_text") and not q_data.get("question"):
                        q_data["question"] = qp["question_text"]
            # Recompute total, but only use it if it's reasonable.
            # Prevents LLM-hallucinated per-question marks inflating total
            # (e.g. 139 instead of 80).
            recomputed = sum(q["max_marks"] for q in questions_dict.values())
            original   = rubric.get("total_exam_marks", 80)
            if original > 0 and abs(recomputed - original) / original <= 0.20:
                rubric["total_exam_marks"] = recomputed
            else:
                logger.warn(
                    f"Recomputed total marks ({recomputed}) differs significantly "
                    f"from parsed total ({original}). Keeping original total."
                )

        # ── Pre-generate LLM reference answers for all questions ───
        print("\n  Pre-generating LLM reference answers...")
        questions_dict = rubric.get("questions", {})
        for q_num, q_data in questions_dict.items():
            print(f"  Generating reference for Q{q_num}...")
            self._llm_references[q_num] = self.ref_gen.generate(
                question    = q_data["question"],
                answer_type = q_data["answer_type"],
                max_marks   = q_data["max_marks"],
            )

        self._rubric = rubric
        return rubric

    # ─────────────────────────────────────────────────────────────
    #  STEP B — Evaluate one student
    # ─────────────────────────────────────────────────────────────
    def evaluate_student(
        self,
        student_id      : str,
        rubric          : dict,
        image_path      : str  = None,
        raw_text        : str  = None,
        anchored_answers: dict = None,   # ← NEW: pre-computed from PDFProcessor
    ) -> dict:
        print("\n" + "─" * 60)
        print(f"  STEP B — Evaluating Student: {student_id}")
        print("─" * 60)

        logger.log("student_started", {"student_id": student_id})

        # ── OCR student sheet ──────────────────────────────────────
        if image_path:
            logger.info(f"Student sheet: {image_path}")
            ocr_result = self.ocr.extract(image_path, mode="student")
        elif raw_text is not None:
            ocr_result = {
                "combined_text"       : raw_text,
                "english_text"        : raw_text,
                "hindi_text"          : "",
                "diagram_descriptions": "",
                "boundaries"          : "",
                "full_response"       : raw_text,
                # Inject pre-computed anchors from PDFProcessor (multi-page PDF path)
                "anchored_answers"    : anchored_answers or {},
            }
            if anchored_answers:
                logger.info(
                    f"  Injected {len(anchored_answers)} pre-computed anchors "
                    f"from PDF processor: Q{sorted(anchored_answers.keys())}"
                )
        else:
            raise ValueError("Provide image_path or raw_text.")

        if not ocr_result.get("combined_text", "").strip():
            logger.warn(f"No text extracted for {student_id}.")

        # ── Segment answers ────────────────────────────────────────
        questions_dict = rubric.get("questions", {})
        total_exam_marks = rubric.get("total_exam_marks", 0)
        
        answers = self.segmenter.segment(
            ocr_result,
            questions_dict,
            question_paper_lookup=self._question_paper_lookup,   # ← pass lookup
        )

        # Store for cheating detection
        self._all_student_answers[student_id] = answers

        student_result: dict = {
            "student_id"   : student_id,
            "timestamp"    : datetime.now().isoformat(),
            "questions"    : {},
            "total_marks"  : 0,
            "max_total"    : total_exam_marks,   # full paper marks (80)
            "attempted_max": 0,                  # filled after loop
            "ocr_metadata" : {
                "tokens_used": ocr_result.get("tokens_used", 0),
                "model_used" : ocr_result.get("model_used", ""),
            },
        }

        # ── Evaluate each question ─────────────────────────────────
        attempted_max = 0   # sum of max_marks for questions student actually attempted
        for q_num, q_rubric in questions_dict.items():
            student_ans = answers.get(q_num, {})
            text_ans    = student_ans.get("text_answer", "").strip()
            diag_ans    = student_ans.get("diagram_description", "").strip()
            # Recompute is_blank from actual text — never rely on OCR's flag alone
            is_blank    = not bool(text_ans) and not bool(diag_ans)

            print(f"\n  Q{q_num} [{q_rubric['answer_type']}] "
                  f"[{q_rubric['max_marks']} marks]: "
                  f"{q_rubric['question'][:50]}...")

            # ── Use cross-referenced question text if available ────────────
            # resolved_question_text comes from the actual question paper (most accurate).
            # Fall back to the rubric question text if not available.
            resolved_q_text = student_ans.get("resolved_question_text", "").strip()
            effective_question = resolved_q_text if resolved_q_text else q_rubric["question"]

            if resolved_q_text and resolved_q_text != q_rubric["question"]:
                logger.info(
                    f"  Q{q_num}: using cross-referenced question text from question paper."
                )
                print(f"  Q-Paper text: {resolved_q_text[:60]}...")

            if is_blank:
                # Check if this question number even appeared in student's anchors
                # If not in anchored answers at all → student didn't attempt it (optional)
                # If in anchors but text empty → truly blank attempt
                in_anchors = q_num in answers
                if not in_anchors:
                    # Student simply didn't attempt this question — skip entirely
                    logger.info(f"  Q{q_num}: not attempted by student — skipping.")
                    continue
                # Student attempted but wrote nothing meaningful
                print(f"  → Blank/Skipped answer.")
                student_result["questions"][f"Q{q_num}"] = {
                    "question"        : q_rubric["question"],
                    "answer_type"     : q_rubric["answer_type"],
                    "max_marks"       : q_rubric["max_marks"],
                    "student_answer"  : "",
                    "is_blank"        : True,
                    "score_fusion"    : {"final_suggested_marks": 0},
                    "dual_eval"       : {"feedback": "No answer written.", "suggested_marks": 0},
                    "keyword_analysis": {"keywords_found": [], "keywords_missing": q_rubric["keywords"]},
                    "human_marks"     : None,
                    "human_override_reason": None,
                }
                attempted_max += q_rubric["max_marks"]
                continue

            print(f"  Answer preview: {text_ans[:70]}...")

            # Get cached LLM reference answer
            llm_ref = self._llm_references.get(q_num, {})
            # Guard: parse_json may return a list if the LLM response was a JSON
            # array instead of an object — fall back to an empty dict so that
            # subsequent .get() calls don't raise AttributeError.
            if not isinstance(llm_ref, dict):
                logger.warn(
                    f"Q{q_num}: LLM reference was not a dict (got {type(llm_ref).__name__}), "
                    "skipping LLM reference for this question."
                )
                llm_ref = {}

            # Keyword match (CPU)
            kw_result = self.kw_matcher.score(
                text_ans,
                q_rubric["keywords"],
                q_rubric["max_marks"],
            )
            print(f"  Keywords found   : {kw_result['keywords_found']}")

            # Dual reference semantic evaluation
            print(f"  Running dual-reference evaluation...")
            dual_eval = self.evaluator.evaluate(
                student_answer   = text_ans,
                student_diagram  = diag_ans,
                examiner_answer  = q_rubric["model_answer"],
                examiner_diagram = q_rubric.get("diagram_description", ""),
                llm_answer       = llm_ref.get("llm_answer", ""),
                llm_diagram      = llm_ref.get("llm_diagram_description", ""),
                question         = effective_question,           # ← cross-referenced
                answer_type      = q_rubric["answer_type"],
                rubric_points    = q_rubric["rubric_points"],
                marking_scheme   = q_rubric.get("marking_scheme", ""),
                max_marks        = q_rubric["max_marks"],
            )

            print(f"  Sim vs examiner  : {dual_eval.get('similarity_vs_examiner', 0):.2f}")
            print(f"  Sim vs LLM ref   : {dual_eval.get('similarity_vs_llm', 0):.2f}")
            print(f"  Final semantic   : {dual_eval.get('final_semantic_similarity', 0):.2f}")

            # Score fusion
            fusion = fuse_scores(
                semantic_result = dual_eval,
                keyword_result  = kw_result,
                max_marks       = q_rubric["max_marks"],
                answer_type     = q_rubric["answer_type"],
            )
            print(f"  Final marks      : {fusion['final_suggested_marks']}/{q_rubric['max_marks']}")
            print(f"  Confidence       : {dual_eval.get('confidence', '?')}")

            student_result["questions"][f"Q{q_num}"] = {
                "question"              : effective_question,        # ← cross-ref if available
                "rubric_question"       : q_rubric["question"],      # original from answer key
                "resolved_question_text": resolved_q_text,           # from question paper
                "answer_type"           : q_rubric["answer_type"],
                "max_marks"             : q_rubric["max_marks"],
                "student_answer"        : text_ans,
                "student_diagram"       : diag_ans,
                "examiner_answer"       : q_rubric["model_answer"],
                "llm_reference_answer"  : llm_ref.get("llm_answer", ""),
                "is_blank"              : False,
                "keyword_analysis"      : kw_result,
                "dual_eval"             : dual_eval,
                "score_fusion"          : fusion,
                "human_marks"           : None,
                "human_override_reason" : None,
            }

            attempted_max += q_rubric["max_marks"]
            student_result["total_marks"] += fusion["final_suggested_marks"]

        # attempted_max = sum of marks for questions student actually attempted
        student_result["attempted_max"] = attempted_max
        # max_total stays as full exam marks (80) — percentage is always out of 80

        pct = round(
            student_result["total_marks"] / student_result["max_total"] * 100, 1
        ) if student_result["max_total"] and student_result["max_total"] > 0 else 0

        print(f"\n  {student_id}: {student_result['total_marks']}"
              f"/{student_result['max_total']} ({pct}%)")

        logger.log("student_done", {
            "student_id"  : student_id,
            "total_marks" : student_result["total_marks"],
            "max_total"   : student_result["max_total"],
            "percentage"  : pct,
        })

        self._all_results.append(student_result)
        return student_result

    # ─────────────────────────────────────────────────────────────
    #  STEP C — Cheating detection
    # ─────────────────────────────────────────────────────────────
    def detect_cheating(self, rubric: dict) -> dict:
        print("\n" + "═" * 60)
        print("  STEP C — Cheating Detection")
        print("═" * 60)

        if len(self._all_student_answers) < 2:
            print("  Need at least 2 students for cheating detection.")
            return {"suspicious_pairs": {}, "answer_key_copying": {}}

        report = self.cheat_det.analyze(self._all_student_answers, rubric.get("questions", {}))
        self.cheat_det.unload()

        sp = report["suspicious_pairs"]
        ak = report["answer_key_copying"]

        if sp:
            print("\n  SUSPICIOUS PAIRS BETWEEN STUDENTS:")
            for q, pairs in sp.items():
                for p in pairs:
                    print(f"    {q}: {p['student_1']} <-> {p['student_2']} "
                          f"sim={p['similarity']} [{p['severity']}]")
        else:
            print("  No suspicious pairs found between students.")

        if ak:
            print("\n  POSSIBLE ANSWER KEY COPYING:")
            for q, cases in ak.items():
                for c in cases:
                    print(f"    {q}: {c['student_id']} sim={c['similarity']}")
        else:
            print("  No answer key copying detected.")

        return report

    # ─────────────────────────────────────────────────────────────
    #  STEP D — Save results
    # ─────────────────────────────────────────────────────────────
    def save_results(self, cheat_report: dict = None) -> str:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname     = f"evaluation_{timestamp}.json"
        fpath     = os.path.join(CONFIG["output_dir"], fname)

        output = {
            "generated_at"      : datetime.now().isoformat(),
            "system_version"    : "2.0_commercial",
            "llm_model"         : CONFIG["ollama_model"],
            "ocr_model"         : CONFIG["groq_model"],
            "students_evaluated": len(self._all_results),
            "evaluation_results": self._all_results,
            "cheating_report"   : cheat_report or {},
            "config_snapshot"   : {
                k: v for k, v in CONFIG.items()
                if k not in ("groq_api_key",)   # never log API key
            },
        }

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved: {fpath}")
        return fpath


# ══════════════════════════════════════════════════════════════════
#  REPORT PRINTER
# ══════════════════════════════════════════════════════════════════

def print_report(pipeline: ExamPipeline, cheat_report: dict):
    print("\n" + "═" * 60)
    print("  FINAL EVALUATION REPORT")
    print("═" * 60)

    for student in pipeline._all_results:
        pct = round(
            student["total_marks"] / student["max_total"] * 100, 1
        ) if student["max_total"] > 0 else 0

        grade = (
            "A+" if pct >= 90 else "A" if pct >= 80 else
            "B"  if pct >= 70 else "C" if pct >= 60 else
            "D"  if pct >= 50 else "F"
        )

        print(f"\n  Student  : {student['student_id']}")
        print(f"  Total    : {student['total_marks']} / {student['max_total']} "
              f"({pct}%) — Grade: {grade}")
        print(f"  {'─' * 50}")

        for q_label, q in student["questions"].items():
            fusion = q.get("score_fusion", {})
            dual   = q.get("dual_eval", {})
            kw     = q.get("keyword_analysis", {})

            print(f"\n  {q_label} [{q.get('answer_type','?')}]: "
                  f"{q['question'][:50]}...")

            if q.get("is_blank"):
                print("    → BLANK — 0 marks")
                continue

            print(f"    Marks         : {fusion.get('final_suggested_marks', 0)}"
                  f" / {q.get('max_marks', '?')}")
            print(f"    Sim examiner  : {dual.get('similarity_vs_examiner', 0):.2f}")
            print(f"    Sim LLM ref   : {dual.get('similarity_vs_llm', 0):.2f}")
            print(f"    Keywords      : {kw.get('keywords_found', [])}")
            print(f"    Confidence    : {dual.get('confidence', '?')}")
            print(f"    Completeness  : {dual.get('answer_completeness', '?')}")
            fb = dual.get("feedback", "")
            if fb:
                print(f"    Feedback      : {fb[:140]}")

            diag = dual.get("diagram_evaluation", {})
            if diag.get("diagram_present"):
                print(f"    Diagram       : {'Correct' if diag.get('diagram_correct') else 'Needs improvement'}"
                      f" — {diag.get('diagram_feedback','')[:80]}")

    # ── Cheating report ────────────────────────────────────────────
    sp = cheat_report.get("suspicious_pairs", {})
    ak = cheat_report.get("answer_key_copying", {})

    if sp or ak:
        print(f"\n  {'═' * 50}")
        print("  CHEATING ALERTS")
        print(f"  {'═' * 50}")
        for q, pairs in sp.items():
            for p in pairs:
                print(f"  {q}: {p['student_1']} <-> {p['student_2']} "
                      f"({p['similarity']}) [{p['severity']}]")
        for q, cases in ak.items():
            for c in cases:
                print(f"  {q}: {c['student_id']} — possible answer key copy "
                      f"({c['similarity']})")
    else:
        print("\n  No cheating alerts.")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT  — CLI  (no demo data, runs real files directly)
#
#  Usage examples
#  ──────────────
#  Single student (image answer sheet):
#    python exam_evaluator_v2.py \
#        --answer_key  answer_key.jpg \
#        --questions   question_paper.pdf \
#        --student     STU001:sheet1.jpg
#
#  Single student (PDF answer sheet):
#    python exam_evaluator_v2.py \
#        --answer_key  answer_key.pdf \
#        --questions   question_paper.pdf \
#        --student     STU001:sheet1.pdf
#
#  Multiple students:
#    python exam_evaluator_v2.py \
#        --answer_key  answer_key.pdf \
#        --questions   question_paper.pdf \
#        --student STU001:s1.pdf \
#        --student STU002:s2.pdf \
#        --student STU003:s3.jpg
#
#  Combined PDF (all students in one file, 4 pages each):
#    python exam_evaluator_v2.py \
#        --answer_key  answer_key.pdf \
#        --questions   question_paper.pdf \
#        --combined    all_students.pdf \
#        --pages_per   4 \
#        --student_ids STU001,STU002,STU003
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AI Exam Evaluator — evaluate real answer sheets directly",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--answer_key", required=True,
        help="Path to answer key image or PDF (required)",
    )
    parser.add_argument(
        "--questions", default=None,
        help="Path to question paper image or PDF\n"
             "(used for cross-reference: question number → question text)",
    )
    parser.add_argument(
        "--student", action="append", default=[],
        metavar="ID:PATH",
        help="Student ID and answer sheet path, e.g.  STU001:sheet1.pdf\n"
             "Repeat this flag for each student.",
    )
    parser.add_argument(
        "--combined", default=None,
        metavar="PATH",
        help="Combined PDF with all students' sheets in one file",
    )
    parser.add_argument(
        "--pages_per", type=int, default=4,
        metavar="N",
        help="Pages per student in the combined PDF (default: 4)",
    )
    parser.add_argument(
        "--student_ids", default=None,
        metavar="ID1,ID2,...",
        help="Comma-separated student IDs for the combined PDF\n"
             "(auto-names student1, student2, ... if not given)",
    )

    args = parser.parse_args()

    # ── Validate at least one student source provided ──────────────
    if not args.student and not args.combined:
        print("\n  ERROR: Provide at least one --student ID:PATH  or  --combined PATH\n")
        parser.print_help()
        sys.exit(1)

    # ── Validate answer key exists ─────────────────────────────────
    if not os.path.exists(args.answer_key):
        print(f"\n  ERROR: Answer key not found: {args.answer_key}\n")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  ExamAI — Real File Evaluation")
    print("═" * 60)

    # ── Init pipeline ──────────────────────────────────────────────
    pipeline = ExamPipeline()

    # ── Load question paper text if provided ──────────────────────
    q_text = ""
    if args.questions:
        if not os.path.exists(args.questions):
            print(f"  WARN: Question paper not found: {args.questions} — skipping cross-ref")
        else:
            print(f"\n  Loading question paper: {args.questions}")
            if args.questions.lower().endswith(".pdf") and pipeline.ocr:
                try:
                    import fitz, tempfile, time as _time
                    doc = fitz.open(args.questions)
                    for pi in range(len(doc)):
                        page = doc.load_page(pi)
                        # Try native text first
                        native = page.get_text()
                        if native and len(native.strip()) > 30:
                            q_text += native
                        else:
                            pix = page.get_pixmap(dpi=200)
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
                                tp = t.name
                            pix.save(tp)
                            res = pipeline.ocr.extract(tp, mode="questions")
                            q_text += res.get("combined_text", "")
                            os.remove(tp)
                            if pi < len(doc) - 1:
                                _time.sleep(2.0)
                    print(f"  Question paper extracted: {len(q_text)} chars")
                except Exception as e:
                    print(f"  WARN: Could not extract question paper text: {e}")
            else:
                # Image — single OCR call
                res = pipeline.ocr.extract(args.questions, mode="questions")
                q_text = res.get("combined_text", "")
                print(f"  Question paper extracted: {len(q_text)} chars")

    # ── Step A — process answer key ────────────────────────────────
    ak_path = args.answer_key
    processor = None
    if ak_path.lower().endswith(".pdf"):
        try:
            processor = __import__("pdf_processor").PDFProcessor(
                pipeline.ocr, CONFIG["groq_model"]
            )
            ak_result = processor.extract(ak_path, mode="answer_key")
            ak_text   = ak_result["full_text"]
            if q_text:
                ak_text = "QUESTIONS:\n" + q_text + "\n\nMODEL ANSWERS:\n" + ak_text
            rubric = pipeline.process_answer_key(
                raw_text            = ak_text,
                question_paper_text = q_text,
            )
        except ImportError:
            print("  WARN: pdf_processor not found — treating answer key as image")
            rubric = pipeline.process_answer_key(
                image_path          = ak_path,
                question_paper_text = q_text,
            )
    else:
        rubric = pipeline.process_answer_key(
            image_path          = ak_path,
            question_paper_text = q_text,
        )

    if not rubric or not rubric.get("questions"):
        print("\n  ERROR: Could not parse answer key. Check file quality and Ollama/Groq setup.\n")
        sys.exit(1)

    print(f"\n  Rubric ready: {len(rubric['questions'])} questions")

    # ── Build student file map ──────────────────────────────────────
    student_files = {}   # {student_id: path}

    # Individual students via --student
    for entry in args.student:
        if ":" not in entry:
            print(f"  WARN: Skipping malformed --student entry '{entry}' (expected ID:PATH)")
            continue
        sid, spath = entry.split(":", 1)
        sid = sid.strip()
        spath = spath.strip()
        if not os.path.exists(spath):
            print(f"  WARN: Student file not found: {spath} — skipping {sid}")
            continue
        student_files[sid] = spath

    # Combined PDF via --combined
    if args.combined:
        if not os.path.exists(args.combined):
            print(f"  ERROR: Combined PDF not found: {args.combined}")
            sys.exit(1)
        sids = None
        if args.student_ids:
            sids = [s.strip() for s in args.student_ids.split(",") if s.strip()]
        try:
            if processor is None:
                processor = __import__("pdf_processor").PDFProcessor(
                    pipeline.ocr, CONFIG["groq_model"]
                )
            from pdf_processor import MultiStudentPDFSplitter
            splitter = MultiStudentPDFSplitter()
            split_map = splitter.split_by_pages(
                args.combined, args.pages_per, sids,
                output_dir=os.path.join(os.path.dirname(args.combined), "split_sheets"),
            )
            student_files.update(split_map)
            print(f"  Combined PDF split into {len(split_map)} students: {sorted(split_map.keys())}")
        except Exception as e:
            print(f"  ERROR splitting combined PDF: {e}")
            sys.exit(1)

    if not student_files:
        print("\n  ERROR: No valid student files to evaluate.\n")
        sys.exit(1)

    # ── Step B — evaluate each student ────────────────────────────
    for i, (sid, spath) in enumerate(student_files.items()):
        print(f"\n  [{i+1}/{len(student_files)}] Processing: {sid}  ({spath})")

        if spath.lower().endswith(".pdf"):
            if processor is None:
                try:
                    processor = __import__("pdf_processor").PDFProcessor(
                        pipeline.ocr, CONFIG["groq_model"]
                    )
                except ImportError:
                    processor = None

            if processor:
                res = processor.extract(spath, mode="student")
                pipeline.evaluate_student(
                    sid, rubric,
                    raw_text         = res["full_text"],
                    anchored_answers = res.get("anchored_answers", {}),
                )
            else:
                # No pdf_processor — try as image (single-page PDF)
                pipeline.evaluate_student(sid, rubric, image_path=spath)
        else:
            # Image file
            pipeline.evaluate_student(sid, rubric, image_path=spath)

    # ── Step C — cheating detection ────────────────────────────────
    cheat_report = pipeline.detect_cheating(rubric)

    # ── Step D — save & print ──────────────────────────────────────
    saved = pipeline.save_results(cheat_report)
    print(f"\n  Results saved → {saved}")

    print_report(pipeline, cheat_report)
    print("\n  Done.\n")