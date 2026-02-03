import os
import io
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_PDF = "./input/anyline-sample-scan-book-ocr.pdf"
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "anyline-sample-scan-book-ocr-na-id-only.pdf"
BATCH_SIZE = 10  # Reduced to 10 for higher accuracy

def encode_image_to_base64(image):
    """Converts a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_batch(images, start_page_num):
    """
    Sends a BATCH of images to OpenAI.
    Forces the AI to identify the 'Country' before declaring a match.
    """
    user_content = [
        {
            "type": "text",
            "text": (
                f"Analyze these {len(images)} images (Pages {start_page_num} to {start_page_num + len(images) - 1}).\n"
                "Your Goal: Return a list of page numbers containing **North American Government IDs** "
                "(USA, Canada, Mexico) ONLY.\n\n"
                "RULES:\n"
                "1. **Read the Country/State Name** on the card first.\n"
                "2. **Exclude** IDs from Countries of Other continents.\n"
                "3. **Exclude** everything other than IDs.\n\n"
                "Return a JSON object with a 'matches' list. Each match must look like this:\n"
                "{ 'page': int, 'country_detected': string, 'doc_type': string }"
            )
        }
    ]

    for i, img in enumerate(images):
        current_page = start_page_num + i
        base64_img = encode_image_to_base64(img)
        
        # Insert explicit text label before the image
        user_content.append({
            "type": "text",
            "text": f"--- PAGE {current_page} ---"
        })
        
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"}
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise Document Forensic Analyst."},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result = json.loads(response.choices[0].message.content)
        matches = result.get("matches", [])
        
        # Log the AI's reasoning for debugging
        valid_pages = []
        for m in matches:
            print(f"  -> Found Page {m['page']}: {m['country_detected']} ({m['doc_type']})")
            valid_pages.append(m['page'])
            
        return valid_pages

    except Exception as e:
        print(f"Error in batch {start_page_num}: {e}")
        return []

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: Input file not found at {INPUT_PDF}")
        return

    print("Converting PDF to images (300 DPI)...")
    # 300 DPI is required to read the small text on Drivers Licenses
    all_images = convert_from_path(INPUT_PDF, dpi=300)
    total_pages = len(all_images)
    
    identified_pages = []
    
    print(f"Starting analysis of {total_pages} pages (Batch Size: {BATCH_SIZE})...")
    
    for i in range(0, total_pages, BATCH_SIZE):
        batch_images = all_images[i : i + BATCH_SIZE]
        start_page = i + 1
        print(f"Scanning pages {start_page}-{start_page + len(batch_images) - 1}...")
        
        pages_in_batch = analyze_batch(batch_images, start_page)
        identified_pages.extend(pages_in_batch)

    # Deduplicate and Sort
    final_pages = sorted(list(set(identified_pages)))
    print(f"\nFinal Identified Pages: {final_pages}")

    if final_pages:
        reader = PdfReader(INPUT_PDF)
        writer = PdfWriter()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        for p_num in final_pages:
            if 1 <= p_num <= total_pages:
                writer.add_page(reader.pages[p_num - 1])
        
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        with open(output_path, "wb") as f:
            writer.write(f)
        print(f"Success! extracted PDF saved to: {output_path}")
    else:
        print("No matching pages found.")

if __name__ == "__main__":
    main()