import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import RPGDataCollector
from src.llm_clients import OpenAIClient, GeminiClient
from config import OPENAI_API_KEY, GEMINI_API_KEY, SAMPLES_PER_MODEL

def main():
    print("🚀 Starting RPG Level Data Collection")
    
    # Initialize clients
    clients = {}
    
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-key-here":
        clients["gpt4"] = OpenAIClient(OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
    
    if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-key-here":
        clients["gemini"] = GeminiClient(GEMINI_API_KEY)
        print("✅ Gemini client initialized")
    
    if not clients:
        print("❌ No API keys configured! Please set OPENAI_API_KEY or GEMINI_API_KEY in config.py")
        return
    
    # Collect data
    collector = RPGDataCollector(clients)
    
    try:
        print(f"📊 Collecting {SAMPLES_PER_MODEL} samples per model...")
        data = collector.collect_data(SAMPLES_PER_MODEL)
        
        # Summary
        print("\n📊 Collection Summary:")
        total_valid = 0
        total_examples = 0
        
        for model, examples in data.items():
            valid_count = sum(1 for ex in examples if ex.get("valid", False))
            total_valid += valid_count
            total_examples += len(examples)
            success_rate = (valid_count/len(examples))*100 if examples else 0
            print(f"  {model}: {valid_count}/{len(examples)} valid ({success_rate:.1f}%)")
        
        overall_rate = (total_valid/total_examples)*100 if total_examples else 0
        print(f"\n🎯 Overall: {total_valid}/{total_examples} valid ({overall_rate:.1f}%)")
        
        print("✅ Data collection complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Collection interrupted by user")
        print("💾 Partial data has been saved")

if __name__ == "__main__":
    main()