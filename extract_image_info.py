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

INPUT_PDF = "./input/school-text-ocr-test.pdf"
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "school-text-more-to-do-only.pdf"
BATCH_SIZE = 10 

def encode_image_to_base64(image):
    """Converts a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_batch(images, start_page_num):
    """
    Sends a BATCH of images to OpenAI.
    Forces the AI to identify pages containing the specific header 'More to do'.
    """
    user_content = [
        {
            "type": "text",
            "text": (
                f"Analyze these {len(images)} images (Pages {start_page_num} to {start_page_num + len(images) - 1}).\n"
                "Your Goal: Identify every page that contains a picture or illustration of a HILL (or mountain), a TREE, or a HOUSE (or building structure).\n\n"
                "RULES:\n"
                "1. Scan the background and foreground of all illustrations and diagrams.\n"
                "2. If an object is found, specify which of the three (Hill, Tree, House) was detected.\n"
                "3. Be specific: A 'Hill' includes mountains; a 'House' includes schools or residential buildings; a 'Tree' includes any clearly defined woody plant.\n"
                "Return a JSON object with a 'matches' list. Each match must look like this:"
                "{ 'page': int, 'objects_detected': ['tree', 'house', 'hill'], 'description': 'Short description of where the object appears on the page' }"
            )
        }
    ]

    for i, img in enumerate(images):
        current_page = start_page_num + i
        base64_img = encode_image_to_base64(img)
        
        # Insert explicit text label before the image to help the model track page numbers
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
                {"role": "system", "content": "You are a Visual Content Analyst specializing in textbook illustrations. Your goal is to identify specific geographical and man-made objects within page images with high precision."},
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
            print(f"  -> Found Page {m['page']}: {m['objects_detected']} (Description: {m.get('description', 'N/A')})")
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
            # Ensure page number is within valid range
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