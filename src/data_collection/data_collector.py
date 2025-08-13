import json
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from .llm_client import LLMClient
from ..game_generation.level_generator import LevelGenerator

class DataCollector:
    def __init__(self, llm_clients: Dict[str, LLMClient], output_dir: str = "data/raw_generations"):
        self.llm_clients = llm_clients
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generators = {
            name: LevelGenerator(client) 
            for name, client in llm_clients.items()
        }
    
    def collect_training_data(self, 
                            num_samples: int = 100,
                            models_to_use: List[str] = None) -> Dict[str, List[Dict]]:
        """Collect training data from multiple LLM models"""
        
        if models_to_use is None:
            models_to_use = list(self.llm_clients.keys())
        
        all_data = {}
        
        for model_name in models_to_use:
            print(f"\nCollecting data from {model_name}...")
            model_data = []
            generator = self.generators[model_name]
            
            for i in range(num_samples):
                print(f"Generating sample {i+1}/{num_samples}")
                
                # Vary the parameters for diversity
                width = 15 + (i % 3) * 5  # 15, 20, 25
                height = 10 + (i % 3) * 5  # 10, 15, 20
                difficulty = ["easy", "medium", "hard"][i % 3]
                theme = ["dungeon", "forest", "castle", "cave"][i % 4]
                
                try:
                    level, response = generator.generate_level(
                        width=width, 
                        height=height, 
                        difficulty=difficulty, 
                        theme=theme
                    )
                    
                    # Create training example
                    training_example = {
                        "input_prompt": response.prompt,
                        "output": response.content,
                        "model": model_name,
                        "parameters": {
                            "width": width,
                            "height": height,
                            "difficulty": difficulty,
                            "theme": theme
                        },
                        "metadata": {
                            "timestamp": response.timestamp,
                            "tokens_used": response.tokens_used,
                            "cost": response.cost_estimate,
                            "validation_passed": level.validate_level()
                        }
                    }
                    
                    model_data.append(training_example)
                    
                except Exception as e:
                    print(f"Error generating sample {i+1}: {e}")
                    continue
            
            all_data[model_name] = model_data
            
            # Save data for this model
            self._save_model_data(model_name, model_data)
        
        return all_data
    
    def _save_model_data(self, model_name: str, data: List[Dict]):
        """Save collected data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_data_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} examples to {filepath}")