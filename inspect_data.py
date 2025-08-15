import json
import sys
import re
from pathlib import Path

def inspect_training_example(file_path: str, example_index: int = 0):
    """Inspect a specific training example in detail"""
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if example_index >= len(data):
        print(f"❌ Example {example_index} not found. File has {len(data)} examples.")
        return
    
    example = data[example_index]
    
    print(f"📋 INSPECTING EXAMPLE {example_index + 1}")
    print("="*60)
    
    # Show parameters
    params = example.get("parameters", {})
    print(f"🎮 Parameters:")
    print(f"   Size: {params.get('width')}x{params.get('height')}")
    print(f"   Difficulty: {params.get('difficulty')}")
    print(f"   Theme: {params.get('theme')}")
    print(f"   Model: {example.get('model')}")
    print(f"   Valid: {example.get('valid', 'Unknown')}")
    
    # Show raw response (first 300 chars)
    raw_response = example.get("response", "")
    print(f"\n📝 Raw Response (first 300 chars):")
    print(f"   {raw_response[:300]}...")
    
    # Try to parse and show JSON structure
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            print(f"\n✅ Parsed JSON Structure:")
            for key, value in parsed.items():
                if key == "terrain_map":
                    if isinstance(value, list) and len(value) > 0:
                        print(f"   {key}: {len(value)} rows x {len(value[0]) if value else 0} cols")
                        # Show first 3 rows
                        for i, row in enumerate(value[:3]):
                            print(f"      Row {i}: {row}")
                        if len(value) > 3:
                            print(f"      ... ({len(value)-3} more rows)")
                    else:
                        print(f"   {key}: {value}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ No valid JSON found")
    
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")

def show_file_summary(file_path: str):
    """Show summary of a data file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    valid_count = sum(1 for ex in data if ex.get("valid", False))
    
    print(f"📁 FILE: {Path(file_path).name}")
    print(f"   Total examples: {len(data)}")
    print(f"   Valid examples: {valid_count}")
    print(f"   Success rate: {valid_count/len(data)*100:.1f}%")
    
    if data:
        print(f"   Model: {data[0].get('model', 'Unknown')}")

if __name__ == "__main__":
    # Auto-find latest data files
    data_dirs = ["data/collected", "data/collected_v2"]
    
    files_found = []
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            files_found.extend(list(dir_path.glob("*.json")))
    
    if not files_found:
        print("❌ No data files found. Run collect_data.py first.")
        exit(1)
    
    if len(sys.argv) > 1:
        # Specific file and example
        file_path = sys.argv[1]
        example_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        inspect_training_example(file_path, example_index)
    else:
        # Show summary of all files
        print("📊 DATA FILES SUMMARY:")
        print("="*50)
        for file_path in files_found:
            show_file_summary(str(file_path))
        
        if files_found:
            print(f"\n💡 To inspect specific example:")
            print(f"   python inspect_data.py {files_found[0]} 0")