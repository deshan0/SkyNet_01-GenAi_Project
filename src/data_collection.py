import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .llm_clients import OpenAIClient, GeminiClient
from .game_objects import GameLevel, Position, Enemy

class RPGDataValidator:
    def __init__(self):
        self.validation_rules = {
            "required_fields": ["width", "height", "difficulty", "theme", "player_spawn", "enemies", "terrain_map"],
            "valid_difficulties": ["easy", "medium", "hard"],
            "valid_themes": ["dungeon", "forest", "castle", "cave"],
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
    
    def validate_level(self, level_data: Dict, expected_width: int, expected_height: int) -> Tuple[bool, List[str]]:
        """Validate a level data structure"""
        errors = []
        
        # Check required fields
        for field in self.validation_rules["required_fields"]:
            if field not in level_data:
                errors.append(f"Missing field: {field}")
        
        if errors:
            return False, errors
        
        # Basic validation
        if level_data.get("width") != expected_width:
            errors.append(f"Width mismatch")
        if level_data.get("height") != expected_height:
            errors.append(f"Height mismatch")
        if level_data.get("difficulty") not in self.validation_rules["valid_difficulties"]:
            errors.append(f"Invalid difficulty")
        
        return len(errors) == 0, errors

class RPGDataCollector:
    def __init__(self, llm_clients: Dict[str, any], output_dir: str = "data/collected"):
        self.clients = llm_clients
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validator = RPGDataValidator()
        
    def _create_prompt(self, width: int, height: int, difficulty: str, theme: str) -> str:
        """Create a detailed prompt for RPG level generation"""
        return f"""Generate a {theme} RPG level with these exact specifications:
- Dimensions: {width} x {height}
- Difficulty: {difficulty}
- Theme: {theme}

Return ONLY a JSON object with this EXACT structure:
{{
    "width": {width},
    "height": {height},
    "difficulty": "{difficulty}",
    "theme": "{theme}",
    "player_spawn": {{"x": <int>, "y": <int>}},
    "enemies": [
        {{"x": <int>, "y": <int>, "type": "basic"}}
    ],
    "terrain_map": [
        ["B", "B", "B", "B", "..."],
        ["B", ".", ".", ".", "..."],
        ["B", ".", "P", ".", "..."],
        ["B", ".", ".", "E", "..."],
        ["B", "B", "B", "B", "..."]
    ]
}}

Rules:
- B = walls (all edges must be walls)
- . = empty floor
- P = player spawn (exactly one, must match player_spawn coordinates)
- E = enemies (multiple allowed, must match enemies list)
- The terrain_map must be COMPLETE with all {height} rows and {width} columns
- All positions must be within bounds (0 to {width-1}, 0 to {height-1})"""

    def _parse_response(self, response: str, width: int, height: int, difficulty: str, theme: str) -> Optional[GameLevel]:
        """Try to parse LLM response into GameLevel with validation"""
        
        # Use validator instead of manual parsing
        parsed_data, parse_errors = self.validator.validate_raw_response(response)
        
        if not parsed_data:
            print(f"    Parse error: {parse_errors[0] if parse_errors else 'Unknown'}")
            return None
        
        # Validate the structure
        is_valid, validation_errors = self.validator.validate_level(parsed_data, width, height)
        
        if not is_valid:
            print(f"    Validation error: {validation_errors[0] if validation_errors else 'Unknown'}")
            return None
        
        try:
            # Create GameLevel
            level = GameLevel(
                width=parsed_data["width"],
                height=parsed_data["height"],
                difficulty=parsed_data["difficulty"],
                theme=parsed_data["theme"],
                player_spawn=Position(parsed_data["player_spawn"]["x"], parsed_data["player_spawn"]["y"]),
                enemies=[Enemy(Position(e["x"], e["y"]), e.get("type", "basic")) for e in parsed_data["enemies"]],
                walls=[],
                terrain_map=parsed_data["terrain_map"]
            )
            
            return level if level.validate() else None
            
        except Exception as e:
            print(f"    GameLevel creation error: {e}")
            return None

    def _generate_with_retry(self, client, prompt: str) -> str:
        """Generate with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.generate(prompt)
                return response
            except Exception as e:
                print(f"    API error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise

    def collect_data(self, num_samples: int) -> Dict[str, List[Dict]]:
        """Collect training data from all available models"""
        all_data = {}
        
        for model_name, client in self.clients.items():
            print(f"\n🔄 Collecting data from {model_name}...")
            model_data = []
            
            for i in range(num_samples):
                # Random parameters
                width, height = random.choice([(15, 10), (20, 15), (25, 20)])
                difficulty = random.choice(["easy", "medium", "hard"])
                theme = random.choice(["dungeon", "forest", "castle", "cave"])
                
                print(f"  Sample {i+1}/{num_samples}: {width}x{height} {difficulty} {theme}")
                
                try:
                    # Generate
                    prompt = self._create_prompt(width, height, difficulty, theme)
                    response = self._generate_with_retry(client, prompt)
                    
                    # Parse and validate
                    level = self._parse_response(response, width, height, difficulty, theme)
                    
                    # Store result
                    example = {
                        "model": model_name,
                        "parameters": {"width": width, "height": height, "difficulty": difficulty, "theme": theme},
                        "prompt": prompt,
                        "response": response,
                        "parsed_level": level.to_dict() if level else None,
                        "valid": level is not None,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    model_data.append(example)
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    # Store failed example
                    example = {
                        "model": model_name,
                        "parameters": {"width": width, "height": height, "difficulty": difficulty, "theme": theme},
                        "prompt": prompt if 'prompt' in locals() else "",
                        "response": "",
                        "parsed_level": None,
                        "valid": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    model_data.append(example)
            
            # Save model data
            timestamp = int(time.time())
            filename = f"{model_name}_data_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            valid_count = sum(1 for ex in model_data if ex["valid"])
            print(f"📁 Saved to {filepath}")
            print(f"✅ Collected {valid_count}/{len(model_data)} valid samples from {model_name}")
            
            all_data[model_name] = model_data
        
        return all_data