from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_dir = "fine_tuned_kid_explainer"  # Directory where the model was saved
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Set pad_token_id if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set the model to generate better outputs
model.eval()

def generate_explanation(concept):
    """Generates an explanation for a given concept using the fine-tuned model."""
    input_text = f"Explain for an 8-year-old kid about {concept}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate output with revised settings
    output = model.generate(
        **inputs, 
        max_length=80, 
        pad_token_id=tokenizer.pad_token_id, 
        do_sample=True, 
        top_p=0.8, 
        top_k=40, 
        temperature=0.7, 
        repetition_penalty=1.2,
        num_return_sequences=1
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test with sample concepts
test_concepts = ["Uplink", "Energy"]
for concept in test_concepts:
    explanation = generate_explanation(concept)
    print(f"\nConcept: {concept}")
    print(f"Kid-Friendly Explanation: {explanation}")
