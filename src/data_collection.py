import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.llm_clients import LLMClient
from src.game_objects import MapTiles, Position
import time

class RPGDataValidator:
    def __init__(self):
        self.validation_rules = {
            "required_fields": ["width", "height", "walls", "enemies", "player_spawn"],  # Changed to player_spawn
            "valid_terrain_chars": ["B", ".", "P", "E", " "],
        }
    
    def validate_raw_response(self, response: str) -> Tuple[Optional[Dict], List[str]]:
        """Try to extract and validate JSON from raw LLM response"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None, ["No JSON object found in response"]
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data, []
            
        except json.JSONDecodeError as e:
            return None, [f"JSON parsing error: {str(e)}"]
        except Exception as e:
            return None, [f"Unexpected error: {str(e)}"]
    
    def validate_map(self, map_data: Dict, expected_width: int, expected_height: int) -> Tuple[bool, List[str]]:
        """Validate a map data structure"""
        errors = []
        
        # Check required fields
        for field in self.validation_rules["required_fields"]:
            if field not in map_data:
                errors.append(f"Missing field: {field}")
        
        if errors:
            return False, errors
        
        # Basic validation
        if map_data.get("width") != expected_width:
            errors.append(f"Width mismatch")
        if map_data.get("height") != expected_height:
            errors.append(f"Height mismatch")
        
        # Check if player_spawn has x and y
        player_spawn = map_data.get("player_spawn", {})
        if not isinstance(player_spawn, dict) or "x" not in player_spawn or "y" not in player_spawn:
            errors.append("Invalid player_spawn format")
        
        return len(errors) == 0, errors

class RPGDataCollector:
    def __init__(self, llm_clients: Dict[str, LLMClient], output_dir: str = "data/collected"):
        self.clients = llm_clients
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validator = RPGDataValidator()
        
    def create_map_prompt(self, width: int, height: int) -> str:
        return f"""
Generate a tilemap for a game level with these exact specifications:
- Dimensions: {width} x {height}
- All edges should be walls
- Place ONE player (P) and multiple enemies (E)
- Place some walls inside the level
- Player should be able to reach all enemies
- Player should be near the center

Return ONLY a JSON object with this exact structure:
{{
    "width": {width},
    "height": {height},
    "player_spawn": {{"x": <int>, "y": <int>}},
    "enemies": [
        {{"x": <int>, "y": <int>}},
        {{"x": <int>, "y": <int>}}
    ],
    "walls": [
        {{"x": <int>, "y": <int>}},
        {{"x": <int>, "y": <int>}}
    ]
}}

Rules:
1. All edge positions must be walls
2. Place 3-8 enemies randomly
3. Make sure all areas are reachable from player position
4. Use coordinates where (0,0) is top-left
5. Player should be roughly in the center area
"""

    def collect_data(self, samples_per_model: int = 50) -> Dict[str, List]:
        """Collect training data from all models"""
        all_data = {}
        
        for model_name, client in self.clients.items():
            print(f"\n🔄 Collecting data from {model_name}...")
            model_data = []
            
            for i in range(samples_per_model):
                # Random map sizes for variety
                width, height = random.choice([(20, 15), (25, 20), (15, 12)])
                
                prompt = self.create_map_prompt(width, height)
                
                print(f"  Sample {i+1}/{samples_per_model}: {width}x{height} map")
                
                # Generate with retry logic
                response = self._generate_with_retry(client, prompt)
                if response:
                    # Try to parse and validate
                    map_tiles = self._parse_response(response, width, height)
                    
                    training_example = {
                        "prompt": prompt,
                        "response": response,
                        "model": model_name,
                        "parameters": {"width": width, "height": height},
                        "valid": map_tiles is not None,
                        "timestamp": time.time()
                    }
                    
                    model_data.append(training_example)
                    
                    # Small delay to be nice to APIs
                    time.sleep(1)
            
            all_data[model_name] = model_data
            
            # Save each model's data
            self._save_data(model_name, model_data)
            print(f"✅ Collected {len(model_data)} samples from {model_name}")
        
        return all_data
    
    def _generate_with_retry(self, client: LLMClient, prompt: str, max_retries: int = 3) -> str:
        """Generate with retry logic"""
        for attempt in range(max_retries):
            response = client.generate(prompt)
            if response:
                return response
            print(f"    Retry {attempt + 1}/{max_retries}")
            time.sleep(2)
        return None
    
    def _parse_response(self, response: str, width: int, height: int):
        """Try to parse LLM response into MapTiles with validation"""
        
        # Use validator to parse JSON
        parsed_data, parse_errors = self.validator.validate_raw_response(response)
        
        if not parsed_data:
            print(f"    Parse error: {parse_errors[0] if parse_errors else 'Unknown'}")
            return None
        
        # Validate the structure
        is_valid, validation_errors = self.validator.validate_map(parsed_data, width, height)
        
        if not is_valid:
            print(f"    Validation error: {validation_errors[0] if validation_errors else 'Unknown'}")
            return None
        
        try:
            # Create MapTiles object
            map_tiles = MapTiles(
                width=parsed_data["width"],
                height=parsed_data["height"],
                player_spawn=Position(parsed_data["player_spawn"]["x"], parsed_data["player_spawn"]["y"]),  # Changed to player_spawn
                enemies=[Position(e["x"], e["y"]) for e in parsed_data["enemies"]],
                walls=[Position(w["x"], w["y"]) for w in parsed_data["walls"]]
            )
            
            return map_tiles if map_tiles.validate() else None
            
        except Exception as e:
            print(f"    MapTiles creation error: {e}")
            return None
    
    def _save_data(self, model_name: str, data: List):
        """Save collected data"""
        timestamp = int(time.time())
        filename = self.output_dir / f"{model_name}_data_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📁 Saved to {filename}")