import torch
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForCausalLM

INPUT_PDF = "./input/school-text-ocr-test.pdf"
PAGE_OFFSET = 0 

TARGET_KEYWORDS = ["tree", "hill", "mountain", "house", "building", "home", "cottage"]

def main():
    print("1. Loading Florence-2 Model (Microsoft's best document VLM)...")
    model_id = 'microsoft/Florence-2-base'
    
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print(f"2. Converting PDF to images...")
    try:
        images = convert_from_path(INPUT_PDF, dpi=200)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return

    print(f"3. Analyzing {len(images)} pages...")
    print("-" * 60)

    for i, image in enumerate(images):
        page_num = i + 1 + PAGE_OFFSET
        
        # Florence-2 Prompt Task: "Describe this image in detail"
        prompt = "Find out building, tree, hill in image if exists"

        inputs = processor(text=prompt, images=image, return_tensors="pt")

        # Generate description
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
                use_cache=False
            )
        
        # Decode the result
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # specific post-processing for Florence-2
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        description = parsed_answer['<MORE_DETAILED_CAPTION>'].lower()

        # Check if any of our target keywords exist in the description
        found_keywords = [word for word in TARGET_KEYWORDS if word in description]
        
        if found_keywords:
            # Deduplicate keywords found
            found_unique = list(set(found_keywords))
            print(f"Page {page_num}: [MATCH] Found {', '.join(found_unique)}")
            # Optional: Print the full description to see what it saw
            # print(f"  AI Saw: \"{description[:100]}...\"")
        else:
            print(f"Page {page_num}: ...", end='\r')

    print("\n" + "-" * 60)
    print("Analysis Complete.")

if __name__ == "__main__":
    main()