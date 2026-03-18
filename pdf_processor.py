# pdf_processor.py
import os
from PyPDF2 import PdfReader, PdfWriter  # pip install PyPDF2

class PDFProcessor:
    """
    Handles PDF text extraction for questions, answer key, and student PDFs.
    Replace dummy logic with OCR/GROQ API calls as needed.
    """
    def __init__(self, client=None, model_name=None):
        self.client = client          # your OCR/GROQ client
        self.model_name = model_name  # model name if using AI extraction

    def extract(self, pdf_path, mode="student"):
        """
        Extract text and page count from PDF.
        mode: "questions", "answer_key", "student"
        """
        if not os.path.exists(pdf_path):
            return {"full_text": "", "page_count": 0}

        full_text = ""
        page_count = 0
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                full_text += page.extract_text() or ""
            page_count = len(reader.pages)
        except Exception as e:
            print(f"[PDFProcessor] Error reading PDF {pdf_path} natively: {e}")

        # If it's empty, or very short (likely a scan/handwritten PDF), use PyMuPDF + Groq
        if len(full_text.strip()) < 50:
            print(f"[PDFProcessor] Native text empty or too short. Switching to PyMuPDF image extraction for {pdf_path}...")
            full_text = ""  # Reset
            import fitz  # PyMuPDF
            import time
            import tempfile
            
            merged_anchors = {}
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                
                for i in range(page_count):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(dpi=200)  # Render page to an image
                    
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                        
                    pix.save(tmp_path)
                    
                    print(f"  [PDFProcessor] Scanning page {i+1}/{page_count}...")
                    
                    if self.client:
                        # self.client is the GroqVLM wrapper here
                        # Call its extract method
                        try:
                            # If self.client is the 'exam_evaluator_v2.GroqVLM' instance
                            res = self.client.extract(tmp_path, mode=mode)
                            full_text += f"\n--- PAGE {i+1} ---\n"
                            full_text += res.get("combined_text", "")
                            
                            # Merge anchored answers across pages.
                            # If the same question number appears on multiple pages
                            # (answer continues), APPEND the text rather than overwrite.
                            page_anchors = res.get("anchored_answers", {})
                            if page_anchors:
                                for q_key, q_data in page_anchors.items():
                                    if q_key in merged_anchors:
                                        # Append continuation text
                                        existing = merged_anchors[q_key]
                                        cont_text = q_data.get("text_answer", "").strip()
                                        if cont_text:
                                            existing["text_answer"] = (
                                                existing.get("text_answer", "") +
                                                "\n" + cont_text
                                            ).strip()
                                        # Merge diagram descriptions too
                                        cont_diag = q_data.get("diagram_description", "").strip()
                                        if cont_diag and cont_diag.lower() != "none":
                                            existing["diagram_description"] = (
                                                existing.get("diagram_description", "") +
                                                "\n" + cont_diag
                                            ).strip()
                                        existing["is_blank"] = not bool(
                                            existing.get("text_answer", "").strip()
                                        )
                                        print(f"    [PDFProcessor] Q{q_key}: appended page {i+1} continuation text.")
                                    else:
                                        # New question number seen for first time
                                        merged_anchors[q_key] = dict(q_data)
                                        print(f"    [PDFProcessor] Q{q_key}: anchored on page {i+1}.")
                        except Exception as ocr_err:
                            print(f"[PDFProcessor] OCR failed on page {i+1}: {ocr_err}")
                    else:
                        print(f"[PDFProcessor] No OCR client provided. Cannot read image text on page {i+1}.")
                        
                    # Clean up the temporary image
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                        
                    # Delay to avoid hitting Groq's free tier rate limits 
                    # Only sleep if there are more pages left
                    if i < page_count - 1 and self.client:
                        time.sleep(2.5)

            except ImportError:
                print(f"[PDFProcessor] PyMuPDF (fitz) is required for scanned PDFs. Run: pip install PyMuPDF")
            except Exception as e:
                print(f"[PDFProcessor] PyMuPDF extraction failed: {e}")

            return {"full_text": full_text, "page_count": page_count, "anchored_answers": merged_anchors}

        return {"full_text": full_text, "page_count": page_count, "anchored_answers": {}}


class MultiStudentPDFSplitter:
    """
    Splits a combined PDF into individual student PDFs by number of pages.
    """
    def split_by_pages(self, pdf_path, pages_per_student, student_ids=None, output_dir=None):
        """
        pdf_path: path to combined PDF
        pages_per_student: number of pages per student
        student_ids: optional list of student IDs (if None, use generic names)
        output_dir: folder to save split PDFs
        """
        if not os.path.exists(pdf_path):
            return {}

        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
        os.makedirs(output_dir, exist_ok=True)

        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            sfmap = {}

            if student_ids is None:
                # generate generic student IDs
                student_ids = [f"student{i+1}" for i in range((total_pages + pages_per_student -1)//pages_per_student)]

            for i, sid in enumerate(student_ids):
                start = i * pages_per_student
                end   = start + pages_per_student
                if start >= total_pages:
                    break

                writer = PdfWriter()
                for page in reader.pages[start:end]:
                    writer.add_page(page)

                out_path = os.path.join(output_dir, f"{sid}.pdf")
                with open(out_path, "wb") as f:
                    writer.write(f)
                sfmap[sid] = out_path

            return sfmap

        except Exception as e:
            print(f"[MultiStudentPDFSplitter] Error splitting PDF: {e}")
            return {}