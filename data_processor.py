import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

def load_collected_data() -> List[Dict[str, Any]]:
    """Load all collected data files"""
    data_dir = Path("data/collected")
    
    if not data_dir.exists():
        print("❌ No collected data found. Run collect_data.py first.")
        return []
    
    all_examples = []
    
    for file_path in data_dir.glob("*.json"):
        print(f"📁 Loading {file_path.name}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_examples.extend(data)
    
    return all_examples

def filter_valid_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter only valid examples"""
    valid_examples = [ex for ex in examples if ex.get("valid", False)]
    print(f"📊 Filtered {len(valid_examples)} valid examples from {len(examples)} total")
    return valid_examples

def convert_to_training_format(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert examples to training format"""
    training_data = []
    
    for example in examples:
        if not example.get("valid", False):
            continue
        
        # Create input-output pair
        params = example["parameters"]
        input_text = f"Generate a {params['theme']} RPG level with dimensions {params['width']}x{params['height']} and {params['difficulty']} difficulty."
        
        # Use the original response as output
        output_text = example["response"]
        
        training_example = {
            "input": input_text,
            "output": output_text,
            "model": example.get("model", "unknown"),
            "parameters": params
        }
        
        training_data.append(training_example)
    
    return training_data

def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
    """Split data into train and validation sets"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def save_processed_data(train_data: List[Dict], val_data: List[Dict], stats: Dict):
    """Save processed data"""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    with open(output_dir / "train_data.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Save validation data
    with open(output_dir / "val_data.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Save statistics
    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"💾 Saved to {output_dir}")

def analyze_dataset(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze dataset statistics"""
    models = [ex.get("model", "unknown") for ex in examples]
    difficulties = [ex["parameters"]["difficulty"] for ex in examples if "parameters" in ex]
    themes = [ex["parameters"]["theme"] for ex in examples if "parameters" in ex]
    sizes = [f"{ex['parameters']['width']}x{ex['parameters']['height']}" for ex in examples if "parameters" in ex]
    
    stats = {
        "total_examples": len(examples),
        "models": dict(Counter(models)),
        "difficulties": dict(Counter(difficulties)),
        "themes": dict(Counter(themes)),
        "sizes": dict(Counter(sizes))
    }
    
    return stats

def main():
    print("🔄 Processing training data...")
    
    # Load all collected data
    all_examples = load_collected_data()
    
    if not all_examples:
        return
    
    # Filter valid examples
    valid_examples = filter_valid_examples(all_examples)
    
    if len(valid_examples) < 10:
        print("⚠️ Warning: Very few valid examples. Consider collecting more data.")
    
    # Convert to training format
    training_data = convert_to_training_format(valid_examples)
    print(f"📝 Converted {len(training_data)} training examples")
    
    # Split data
    train_data, val_data = split_data(training_data)
    print(f"📈 Split: {len(train_data)} train, {len(val_data)} validation")
    
    # Analyze dataset
    stats = analyze_dataset(valid_examples)
    
    # Save processed data
    save_processed_data(train_data, val_data, stats)
    
    # Print statistics
    print("\n📊 DATASET STATISTICS:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Models: {stats['models']}")
    print(f"   Difficulties: {stats['difficulties']}")
    print(f"   Themes: {stats['themes']}")
    print(f"   Sizes: {stats['sizes']}")
    
    print("✅ Data processing complete!")
    print(f"\n🎯 Ready for training with {len(train_data)} examples!")

if __name__ == "__main__":
    main()