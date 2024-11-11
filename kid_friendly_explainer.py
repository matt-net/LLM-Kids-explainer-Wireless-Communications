from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import json
from torch.utils.data import Dataset

class KidFriendlyExplainerDataset(Dataset):
    """Custom dataset class to handle kid-friendly explanation data for fine-tuning."""
    
    def __init__(self, data_path, tokenizer, max_length=70):
        # Load and process data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        self.data = [{"input_text": f"Explain for a 8 year old kid about {item['Concept']}", 
                      "output_text": item["Explanation"]} for item in raw_data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item["input_text"], padding="max_length", truncation=True, max_length=self.max_length)
        labels = self.tokenizer(item["output_text"], padding="max_length", truncation=True, max_length=self.max_length)

        # Use -100 to ignore padding in labels
        labels["input_ids"] = [label if label != self.tokenizer.pad_token_id else -100 for label in labels["input_ids"]]
        inputs["labels"] = labels["input_ids"]
        return {key: torch.tensor(val) for key, val in inputs.items()}

class FineTuneKidExplainer:
    """Class to fine-tune a language model to generate kid-friendly explanations."""

    def __init__(self, model_name="gpt2", data_path="kid_explanations.json"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not already defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Initialize the model and resize embeddings if new tokens were added
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.data_path = data_path
        self.dataset = KidFriendlyExplainerDataset(data_path, self.tokenizer)

    def fine_tune(self, output_dir="fine_tuned_kid_explainer", epochs=3, batch_size=8):
        """Fine-tunes the model on kid-friendly explanations."""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no"  # No evaluation dataset required
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
        )
        
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model fine-tuned and saved to {output_dir}")

    def generate_explanation(self, concept):
        """Generates an explanation for a given concept using the fine-tuned model."""
        input_text = f"Explain for a 7-year old kid about {concept}"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=70)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
