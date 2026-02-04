import torch
from pdf2image import convert_from_path
from transformers import CLIPProcessor, CLIPModel

INPUT_PDF = "./input/school-text-ocr-test.pdf"
CONFIDENCE_THRESHOLD = 0.70 

LABELS = [
    "a photo or drawing of a hill or mountain",
    "a photo or drawing of a tree",
    "a photo or drawing of a house or building",
]

def main():
    print("1. Loading CLIP Model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"2. Converting PDF '{INPUT_PDF}' to images...")
    try:
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

        print(f"Page {page_num}:")
        found_match = False
        
        for idx, label in enumerate(LABELS):
            score = probs[idx].item()
            
            if score > CONFIDENCE_THRESHOLD:
                print(f"  [FOUND] {label} ({score:.1%})")
                found_match = True
        
        if not found_match:
            print("  (No significant objects found)")
        print("-" * 60)

if __name__ == "__main__":
    main()