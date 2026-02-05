import os, base64
from dotenv import load_dotenv
from mistralai import Mistral
load_dotenv()

INPUT_PDF = "./input/anyline-sample-scan-book-ocr.pdf"
OUTPUT_DIR = "./output/md_with_img_id"
OUTPUT_FILENAME = "anyline-sample-scan-book-ocr.md"

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
uploaded_file = client.files.upload(
    file = {
        "file_name" : "school-text-ocr-test.pdf",
        "content" : open(INPUT_PDF, "rb")
    },
    purpose="ocr"
)
file_url = client.files.get_signed_url(file_id=uploaded_file.id)

response = client.ocr.process(
    model='mistral-ocr-latest',
    document={
        "type" : "document_url",
        "document_url" : file_url.url
    },
    include_image_base64=True
)

def data_uri_to_bytes(data_uri):
    _, encoded = data_uri.split(',', 1)
    return base64.b64decode(encoded)

def export_image(image):
    parsed_image = data_uri_to_bytes(image.image_base64)
    image_path = os.path.join(OUTPUT_DIR, image.id)
    with open(image_path, "wb") as file:
        file.write(parsed_image)

md_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
with open(md_path, "w", encoding="utf-8") as f:
    for page_num, page in enumerate(response.pages, start=1):
        f.write(f"\n\n----PDF PAGE {page_num}----\n\n")
        f.write(page.markdown)
        for image in page.images:
            export_image(image)