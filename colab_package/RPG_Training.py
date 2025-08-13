# RPG Level Generator Training Script
# Upload this to Google Colab and run each cell

# ================================
# CELL 1: Setup and Installation
# ================================
!pip install -q transformers datasets peft accelerate bitsandbytes

import torch
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from google.colab import drive

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ================================
# CELL 2: Mount Drive and Load Data
# ================================
# Mount Google Drive
drive.mount('/content/drive')

# Navigate to your uploaded folder (adjust path as needed)
import os
os.chdir('/content/drive/MyDrive/colab_package')  # Adjust this path

# Load training data
def load_training_data():
    """Load the processed training data"""
    with open('train_data.json', 'r') as f:
        train_data = json.load(f)
    
    with open('val_data.json', 'r') as f:
        val_data = json.load(f)
    
    with open('dataset_stats.json', 'r') as f:
        stats = json.load(f)
    
    print(f"Dataset loaded:")
    print(f"   Training examples: {len(train_data)}")
    print(f"   Validation examples: {len(val_data)}")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Models used: {stats['models_used']}")
    
    return train_data, val_data, stats

train_data, val_data, dataset_stats = load_training_data()

# ================================
# CELL 3: Model and Tokenizer Setup
# ================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def setup_model_and_tokenizer():
    print(f"Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer

model, tokenizer = setup_model_and_tokenizer()

# ================================
# CELL 4: Data Preparation
# ================================
def format_training_example(example):
    """Format example for instruction following"""
    instruction = example["instruction"]
    output = example["output"]
    
    # Chat format for TinyLlama
    formatted = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    return formatted

def prepare_dataset(train_data, val_data, tokenizer):
    """Prepare dataset for training"""
    print("Preparing datasets...")
    
    # Format examples
    train_texts = [format_training_example(ex) for ex in train_data]
    val_texts = [format_training_example(ex) for ex in val_data]
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    
    # Tokenize
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=1024,
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    train_dataset = train_dataset.map(tokenize_function, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, remove_columns=["text"])
    
    print(f"Datasets prepared:")
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Val: {len(val_dataset)} examples")
    print(f"   Sample length: {len(train_dataset[0]['input_ids'])} tokens")
    
    return train_dataset, val_dataset

train_dataset, val_dataset = prepare_dataset(train_data, val_data, tokenizer)

# ================================
# CELL 5: LoRA Configuration
# ================================
def setup_lora_config():
    """Setup LoRA for efficient fine-tuning"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,                    # Rank - higher = more parameters
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.1,        # Dropout for regularization
        target_modules=[         # TinyLlama specific modules
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    return lora_config

# Apply LoRA to model
lora_config = setup_lora_config()
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# ================================
# CELL 6: Training Setup
# ================================
def setup_training_args():
    """Setup training arguments"""
    return TrainingArguments(
        output_dir="./rpg_model_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,      # Small batch due to memory
        gradient_accumulation_steps=8,       # Effective batch size = 8
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=25,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # Disable wandb for simplicity
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

training_args = setup_training_args()

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ================================
# CELL 7: Training
# ================================
# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print("Starting training...")
print("This will take 20-30 minutes on Colab T4 GPU")

# Train the model
trainer.train()

print("Training completed!")

# ================================
# CELL 8: Save Model
# ================================
# Save the trained model
output_dir = "./rpg_model_final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")

# ================================
# CELL 9: Test Generation
# ================================
def test_generation():
    """Test the trained model"""
    test_prompt = """Generate a dungeon RPG level with these exact specifications:
- Dimensions: 15 x 10
- Difficulty: medium
- Theme: dungeon

Return ONLY a JSON object with this exact structure:
{
    "width": 15,
    "height": 10,
    "difficulty": "medium",
    "theme": "dungeon",
    "player_spawn": {"x": <int>, "y": <int>},
    "enemies": [
        {"x": <int>, "y": <int>, "type": "basic"}
    ],
    "terrain_map": [
        ["B", "B", "B", "..."],
        ["B", ".", ".", "..."]
    ]
}"""

    # Format for generation
    formatted_prompt = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    assistant_start = generated_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    response = generated_text[assistant_start:].strip()
    
    print("Generated RPG Level:")
    print(response[:500] + "..." if len(response) > 500 else response)
    
    return response

# Test the model
generated_level = test_generation()

# ================================
# CELL 10: Download Model
# ================================
# Zip the model for download
!zip -r rpg_model_final.zip rpg_model_final/

from google.colab import files
files.download('rpg_model_final.zip')

print("Model downloaded! You can now use it locally.")
