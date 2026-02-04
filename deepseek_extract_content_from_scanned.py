import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_PDF = "./input/school-text-ocr-test.pdf"
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "school-text-more-to-do-ocr-tesseract.pdf"
BATCH_SIZE = 10 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCR_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"

print(f"Loading OCR model {OCR_MODEL_ID} on {DEVICE}... (this may take a minute)")
try:
    ocr_processor = AutoProcessor.from_pretrained(OCR_MODEL_ID, trust_remote_code=True)
    # Use float16 on CUDA for memory savings if available; otherwise float32
    model_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    ocr_model = AutoModelForVision2Seq.from_pretrained(
        OCR_MODEL_ID,
        torch_dtype=model_dtype,
        trust_remote_code=True
    ).to(DEVICE)
    ocr_model.eval()
except Exception as e:
    print("Error loading DeepSeek OCR model:", e)
    raise

def ocr_image(image: Image.Image) -> str:
    """
    Runs DeepSeek-OCR-2 on a PIL image and returns extracted text.
    This replaces the previous pytesseract.image_to_string call.
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Processor expects images argument; returns tensors keyed (commonly 'pixel_values')
    inputs = ocr_processor(images=image, return_tensors="pt")
    # Move tensors to DEVICE
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            # generate may accept the same inputs; keep generation conservative
            generated_ids = ocr_model.generate(
                **inputs,
                max_new_tokens=1024,  # reasonable cap per page
                do_sample=False
            )
    except Exception as e:
        # Retry with smaller token limit if generation fails
        print("OCR generation error (retrying with smaller max_new_tokens):", e)
        with torch.no_grad():
            generated_ids = ocr_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

    # Decode to text
    text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def analyze_batch(ocr_texts, start_page_num):
    pages_block = []
    for i, text in enumerate(ocr_texts):
        page_no = start_page_num + i
        pages_block.append(
            f"\n--- PAGE {page_no} ---\n{text.strip()}"
        )

    prompt = f"""
Analyze the following OCR-extracted text from document pages.

Your Goal: Return a list of page numbers that contain the specific section header **'More to do!'**.
RULES:
1. **Look for the text** 'More to do' or 'More to do!'. It usually appears in a distinct bubble, box, or sidebar.
2. **Distinguish Carefully.**
3. Be Cautious with page number: Pages are respective of pdf file which starts with 1. Check the exact page which has structure.
    Return a JSON object with a 'matches' list. Each match must look like this:
    {{
        'page': <int>, 
        'section_detected': string
        'confidence': string 
    }}
PAGES:
{''.join(pages_block)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.2-pro",
            messages=[
                {"role": "system", "content": "You are a strict document classification engine."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        matches = result.get("matches", [])

        valid_pages = []
        for m in matches:
            print(f"  -> Found Page {m['page']}: {m['country_detected']} ({m['doc_type']})")
            valid_pages.append(m["page"])

        return valid_pages

    except Exception as e:
        print(f"OCR batch error at page {start_page_num}: {e}")
        return []

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Input file not found: {INPUT_PDF}")
        return

    print("Converting PDF to images (300 DPI)...")
    images = convert_from_path(INPUT_PDF, dpi=300)
    total_pages = len(images)

    identified_pages = []

    print(f"Running OCR + Analysis on {total_pages} pages...")

    for i in range(0, total_pages, BATCH_SIZE):
        batch_images = images[i:i + BATCH_SIZE]
        start_page = i + 1

        print(f"Processing pages {start_page}-{start_page + len(batch_images) - 1}")

        ocr_texts = []
        for img in batch_images:
            text = ocr_image(img)
            ocr_texts.append(text)

        pages = analyze_batch(ocr_texts, start_page)
        identified_pages.extend(pages)

    final_pages = sorted(set(identified_pages))
    print(f"\nFinal Identified Pages: {final_pages}")

    if final_pages:
        reader = PdfReader(INPUT_PDF)
        writer = PdfWriter()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for p in final_pages:
            writer.add_page(reader.pages[p - 1])

        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        with open(output_path, "wb") as f:
            writer.write(f)

        print(f"✅ Extracted PDF saved to: {output_path}")
    else:
        print("No matching pages found.")

if __name__ == "__main__":
    main()
