import os, json
from dotenv import load_dotenv
from mistralai import Mistral
from openai import OpenAI
from pypdf import PdfReader, PdfWriter

load_dotenv()

INPUT_PDF = "./input/anyline-sample-scan-book-ocr.pdf"
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "result.pdf"

mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_mistral_markdown(pdf_path):
    print(f"1. Uploading '{pdf_path}' to Mistral OCR...")
    
    try:
        with open(pdf_path, "rb") as f:
            uploaded_file = mistral.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": f,
                },
                purpose="ocr",
            )
        
        print(f"2. Processing OCR with Mistral (File ID: {uploaded_file.id})...")
        
        signed_url = mistral.files.get_signed_url(file_id=uploaded_file.id)
        
        ocr_response = mistral.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=False
        )
        
        full_markdown = ""
        # Mistral OCR returns a list of pages
        for i, page in enumerate(ocr_response.pages):
            page_num = i + 1
            # Mistral returns .markdown string for the page
            page_content = page.markdown
            full_markdown += f"\n--- PAGE {page_num} ---\n{page_content}\n"
            
        print("   OCR Complete.")
        return full_markdown, len(ocr_response.pages)

    except Exception as e:
        print(f"Error in Mistral OCR: {e}")
        return None, 0

def analyze_with_openai(markdown_text, total_pages):
    """
    Sends the Mistral Markdown to OpenAI with the specific NA ID prompt.
    Ref: gpt-5.2-extract_na_docs.py prompt logic 
    """
    print("3. Sending OCR Markdown to OpenAI for NA ID Analysis...")

    prompt = f"""
Analyze the following OCR-extracted text from document pages.
Your Goal: Return a list of page numbers containing **North American Government IDs**
[USA, Canada, Mexico] ONLY.

RULES:
1. **Read the Country/State Name** on the card first.
2. **Exclude** IDs from Countries of Other continents.
3. ID's can include passports, driving license, visas of North American, green card, other IDs.
4. **Exclude** everything other than IDs.
5. Be Cautious with page number: Pages are respective of pdf file which starts with 1. Check the exact page which has structure.

Return a JSON object with a 'matches' list. Each match must look like this:
    {{
        'page': <int>, 
        'country_detected': string, 
        'doc_type': string 
    }}

PAGES CONTENT:
{markdown_text}
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a strict document classification engine."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        matches = result.get("matches", [])

        valid_pages = []
        print("\nAnalysis Matches:")
        for m in matches:
            # Validate page number is within range
            if 1 <= m['page'] <= total_pages:
                print(f"  -> Found Page {m['page']}: {m['country_detected']} ({m['doc_type']})")
                valid_pages.append(m["page"])
            else:
                print(f"  -> Ignored Invalid Page {m['page']}")

        return sorted(list(set(valid_pages)))

    except Exception as e:
        print(f"Error in OpenAI Analysis: {e}")
        return []

def split_and_save_pdf(original_pdf, pages_to_keep, output_dir, output_filename):
    if not pages_to_keep:
        print("No matching pages found to extract.")
        return

    print(f"4. Extracting {len(pages_to_keep)} pages to new PDF...")
    
    reader = PdfReader(original_pdf)
    writer = PdfWriter()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for p in pages_to_keep:
        try:
            writer.add_page(reader.pages[p - 1])
        except IndexError:
            print(f"  Warning: Page {p} out of range, skipping.")

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"✅ Extracted PDF saved to: {output_path}")

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Input file not found: {INPUT_PDF}")
        return

    # Step 1: Mistral OCR
    markdown_text, total_pages = get_mistral_markdown(INPUT_PDF)
    
    if not markdown_text:
        return

    # Step 2: OpenAI Analysis
    target_pages = analyze_with_openai(markdown_text, total_pages)

    # Step 3: PDF Split
    split_and_save_pdf(INPUT_PDF, target_pages, OUTPUT_DIR, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()