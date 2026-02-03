import os
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
INPUT_PDF = "./input/school-text-ocr-test.pdf"  # Your PDF path
CONFIDENCE_THRESHOLD = 0.20  # Minimum 20% confidence to consider it a match

# These are the "prompts" the AI checks against the image.
# We include "a page of text" as a 'negative' class so it doesn't force a match on text-only pages.
LABELS = [
    "a photo or drawing of a hill or mountain",
    "a photo or drawing of a tree",
    "a photo or drawing of a house or building",
    "a page of text document" 
]

def main():
    print("1. Loading CLIP Model (this downloads ~600MB on the first run)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"2. Converting PDF '{INPUT_PDF}' to images...")
    try:
        # Convert all pages to images (200 DPI is usually enough for object detection)
        images = convert_from_path(INPUT_PDF, dpi=200)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return

    print(f"3. Analyzing {len(images)} pages for objects...")
    print("-" * 60)

    for i, image in enumerate(images):
        page_num = i + 1
        
        # Prepare inputs for the model
        inputs = processor(
            text=LABELS, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )

        # Run the model (no_grad disables training mode to save memory)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get probabilities (softmax makes them add up to 100%)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        # Check results
        print(f"Page {page_num}:")
        found_match = False
        
        for idx, label in enumerate(LABELS):
            score = probs[idx].item()
            
            # We ignore the "text document" label for the final report
            if "text document" not in label:
                # If score is higher than threshold, we flag it
                if score > CONFIDENCE_THRESHOLD:
                    print(f"  [FOUND] {label} ({score:.1%})")
                    found_match = True
        
        if not found_match:
            print("  (No significant objects found)")
        print("-" * 60)

if __name__ == "__main__":
    main()