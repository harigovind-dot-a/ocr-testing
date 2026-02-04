import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader, PdfWriter
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_PDF = "./input/school-text-ocr-test.pdf"
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "school-text-more-to-do-only.pdf"

def extract_pages(input_pdf, matches):
    reader = PdfReader(input_pdf)

    total_pages = len(reader.pages)

    # Collect page numbers safely
    pages = sorted({
        m["page"]
        for m in matches
        if 1 <= m["page"] <= total_pages
    })

    if not pages:
        print("No valid pages to extract.")
        return
    
    writer = PdfWriter()

    for page_num in pages:
        writer.add_page(reader.pages[page_num - 1])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"✅ Extracted PDF saved to: {output_path}")

PROMPT = """
Analyze the provided document.
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
"""

def main():
    # Upload PDF once
    file = client.files.create(
        file=open(INPUT_PDF, "rb"),
        purpose="user_data"
    )

    # Single Responses API call
    response = client.responses.create(
        model="gpt-4o",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file.id},
                {"type": "input_text", "text": PROMPT}
            ]
        }],
        text={
            "format": {"type": "json_object"}
        },
    )

    result = json.loads(response.output_text)
    print(json.dumps(result, indent=2))
    matches = result.get("matches", [])
    extract_pages(INPUT_PDF, matches)

if __name__ == "__main__":
    main()