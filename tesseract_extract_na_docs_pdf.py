import os, json
import json
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter
import pytesseract
from PIL import Image

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_PDF = "./input/anyline-sample-scan-book-ocr.pdf"
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "anyline-sample-scan-book-ocr-tesseract.pdf"
BATCH_SIZE = 10

def ocr_image(image: Image.Image) -> str:
    """
    Runs Tesseract OCR on a PIL image.
    """
    return pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 6"
    )

def analyze_batch(ocr_texts, start_page_num):
    """
    Sends OCR TEXT (not images) to OpenAI.
    """
    pages_block = []
    for i, text in enumerate(ocr_texts):
        page_no = start_page_num + i
        pages_block.append(
            f"\n--- PAGE {page_no} ---\n{text.strip()[:4000]}"
        )

    prompt = f"""
Analyze the following OCR-extracted text from document pages.

GOAL:
Return ONLY pages that contain **North American Government IDs**
(United States, Canada, Mexico).

RULES:
1. Country name must be explicitly visible.
2. Exclude IDs from Europe, Asia, Africa, Middle East, etc.
3. Exclude non-ID documents (meters, barcodes, license plates, etc).

Return JSON ONLY in this format:
{{
  "matches": [
    {{ "page": int, "country_detected": string, "doc_type": string }}
  ]
}}

PAGES:
{''.join(pages_block)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict document classification engine."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
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
