import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter
import pytesseract

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIGURATION
INPUT_PDF = "./input/school-text-ocr-test.pdf"
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "school-text-more-to-do-ocr-tesseract.pdf"
BATCH_SIZE = 10 

def extract_text_from_image(image):
    """
    Uses Tesseract OCR to extract text from a PIL image.
    """
    try:
        # standard config for block of text
        text = pytesseract.image_to_string(image, lang='eng') 
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def analyze_text_batch(page_text_map):
    """
    Sends EXTRACTED TEXT (not images) to OpenAI.
    """
    # Construct a clean text payload for the AI
    # We create a structured string that clearly delineates pages
    context_str = ""
    for page_num, text in page_text_map.items():
        clean_text = text.replace('\n', ' ').strip()
        # Limit text length per page to avoid token limits if OCR is messy
        context_str += f"--- PAGE {page_num} START ---\n{clean_text[:3000]}\n--- PAGE {page_num} END ---\n\n"

    system_prompt = (
        "You are a text filter assistant. You will receive OCR text from a textbook. "
        "Your task is to identify which pages contain the specific section header: 'More to do'."
    )

    user_prompt = (
        f"Analyze the following extracted text from {len(page_text_map)} pages.\n\n"
        "**TARGET:** Find pages that contain the distinct header **'More to do!'** or **'More to do'**.\n"

        f"{context_str}\n\n"
        "**OUTPUT FORMAT:**\n"
        "Return strictly a JSON object with a 'matches' list:\n"
        "{ \"matches\": [ { \"page\": int, \"snippet_found\": string } ] }"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5.2-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        matches = result.get("matches", [])
        
        valid_pages = []
        for m in matches:
            print(f"  -> Found Page {m['page']}: \"{m.get('snippet_found', 'N/A')}\"")
            valid_pages.append(m['page'])
            
        return valid_pages

    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return []

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: Input file not found at {INPUT_PDF}")
        return

    print("Step 1: Converting PDF to images...")
    # 300 DPI ensures Tesseract can read small headers clearly
    all_images = convert_from_path(INPUT_PDF, dpi=300)
    total_pages = len(all_images)
    
    identified_pages = []
    
    print(f"Step 2: Starting OCR & Analysis of {total_pages} pages...")
    
    for i in range(0, total_pages, BATCH_SIZE):
        batch_images = all_images[i : i + BATCH_SIZE]
        start_page = i + 1
        end_page = start_page + len(batch_images) - 1
        
        print(f"\nProcessing Batch: Pages {start_page}-{end_page}")
        
        # 1. Local OCR Extraction
        page_text_map = {}
        for idx, img in enumerate(batch_images):
            curr_page = start_page + idx
            print(f"  - OCR Scanning Page {curr_page}...", end="\r")
            text = extract_text_from_image(img)
            page_text_map[curr_page] = text
        print(f"  - OCR Complete for batch. Sending text to GPT-4o...")

        # 2. AI Text Analysis
        pages_in_batch = analyze_text_batch(page_text_map)
        identified_pages.extend(pages_in_batch)

    # Deduplicate and Sort
    final_pages = sorted(list(set(identified_pages)))
    print(f"\nFinal Identified Pages: {final_pages}")

    # 3. PDF Extraction
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
        print(f"Success! Extracted PDF saved to: {output_path}")
    else:
        print("No matching pages found.")

if __name__ == "__main__":
    main()