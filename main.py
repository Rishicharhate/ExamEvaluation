import os
import base64
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_media_type(image_path: str) -> str:
    """Detect media type from file extension."""
    ext = image_path.split(".")[-1].lower()
    format_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return format_map.get(ext, "image/jpeg")


# ── OCR Prompt tuned for handwritten text ────────────────────────────────────
OCR_PROMPT = """You are an expert OCR system specialized in reading handwritten text.
Carefully analyze this handwritten image and extract ALL text visible in it.

Please provide your response in the following structured format:

**ENGLISH TEXT:**
[Extract all handwritten English text exactly as it appears, preserving line breaks]

**HINDI TEXT (हिंदी पाठ):**
[Extract all handwritten Hindi/Devanagari text exactly as it appears, preserving line breaks]

**COMBINED TEXT (Original Layout):**
[Show all text together maintaining the original top-to-bottom order as written]

Important instructions:
- Read cursive, printed, and mixed handwriting carefully
- Preserve original spelling even if there are mistakes
- Do NOT correct grammar or spelling
- Do NOT translate any text
- If a word is unclear, write your best guess with a (?) next to it

If no English text is found, write "No English text detected."
If no Hindi text is found, write "कोई हिंदी पाठ नहीं मिला (No Hindi text detected)."
"""


def parse_response(full_response: str, model: str, tokens: int) -> dict:
    """Parse the VLM response into structured sections."""
    result = {
        "full_response": full_response,
        "english_text": "",
        "hindi_text": "",
        "combined_text": "",
        "model_used": model,
        "tokens_used": tokens,
    }

    if "**ENGLISH TEXT:**" in full_response:
        start = full_response.find("**ENGLISH TEXT:**") + len("**ENGLISH TEXT:**")
        end = full_response.find("**HINDI TEXT")
        result["english_text"] = full_response[start: end if end != -1 else None].strip()

    if "**HINDI TEXT" in full_response:
        start = full_response.find("**HINDI TEXT")
        start = full_response.find("\n", start) + 1
        end = full_response.find("**COMBINED TEXT")
        result["hindi_text"] = full_response[start: end if end != -1 else None].strip()

    if "**COMBINED TEXT" in full_response:
        start = full_response.find("**COMBINED TEXT")
        start = full_response.find("\n", start) + 1
        result["combined_text"] = full_response[start:].strip()

    return result


def extract_text_from_image(image_path: str) -> dict:
    """
    Extract handwritten English and Hindi text from a local image.

    Args:
        image_path : Path to your local image file (jpg, jpeg, png, webp).

    Returns:
        Dictionary with english_text, hindi_text, combined_text, and metadata.
    """

    # Validate file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: '{image_path}'\n"
            f"Make sure the image is in the correct folder."
        )

    print(f"📂 Loading image : {image_path}")
    image_data  = encode_image_to_base64(image_path)
    media_type  = get_media_type(image_path)
    data_url    = f"data:{media_type};base64,{image_data}"

    print("🤖 Sending to Groq VLM...")

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": OCR_PROMPT,
                    },
                ],
            }
        ],
        max_tokens=2048,
        temperature=0.1,  # Low = more accurate OCR
    )

    return parse_response(
        response.choices[0].message.content,
        response.model,
        response.usage.total_tokens,
    )


def process_multiple_images(image_paths: list) -> list:
    """Process a batch of local images."""
    results = []
    for path in image_paths:
        print(f"\n{'─'*50}")
        try:
            result = extract_text_from_image(path)
            results.append({"image": path, "status": "success", **result})
        except Exception as e:
            results.append({"image": path, "status": "error", "error": str(e)})
            print(f"❌ Error: {e}")
    return results


def print_result(result: dict):
    """Pretty-print extraction result."""
    print(f"\n{'='*55}")
    print(f"📌 Model        : {result['model_used']}")
    print(f"📊 Tokens used  : {result['tokens_used']}")
    print(f"\n🔤 ENGLISH TEXT :\n{result['english_text'] or 'No English text detected.'}")
    print(f"\n🔤 HINDI TEXT   :\n{result['hindi_text']   or 'कोई हिंदी पाठ नहीं मिला (No Hindi text detected).'}")
    print(f"\n📄 COMBINED     :\n{result['combined_text'] or '—'}")
    print(f"{'='*55}\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("🚀 Groq VLM — Handwritten Hindi & English OCR")
    print("=" * 55)

    # ─────────────────────────────────────────────────────
    #  👇 CHANGE THIS to your image filename / full path
    # ─────────────────────────────────────────────────────
    IMAGE_PATH = "hindi.jpg"   # same folder as main.py
    # IMAGE_PATH = r"C:\Users\rishi\Pictures\note.png"  # or full path

    result = extract_text_from_image(IMAGE_PATH)
    print_result(result)

    # ── Batch mode: multiple images at once ───────────────
    # images = ["page1.jpg", "page2.jpg", "page3.png"]
    # all_results = process_multiple_images(images)
    # for r in all_results:
    #     print_result(r)