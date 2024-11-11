from kid_friendly_explainer import FineTuneKidExplainer

# Initialize the fine-tuning class
explainer = FineTuneKidExplainer(model_name="gpt2", data_path="kid_explanations.json")

# Fine-tune the model
explainer.fine_tune(epochs=100, batch_size=8)

# Test the model on a sample concept
print(explainer.generate_explanation("Bandwidth"))
