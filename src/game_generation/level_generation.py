from typing import List, Dict, Any
import json
from .game_objects import GameLevel, Position, Player, Enemy
from ..data_collection.llm_client import LLMClient, LLMResponse

class LevelGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.generation_history: List[Dict[str, Any]] = []
    
    def generate_level(self, 
                      width: int = 20, 
                      height: int = 15, 
                      difficulty: str = "medium",
                      theme: str = "dungeon") -> tuple[GameLevel, LLMResponse]:
        """Generate a level and return both the level and the LLM response for training data"""
        
        prompt = self._create_level_prompt(width, height, difficulty, theme)
        
        response = self.llm_client.generate(
            prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        try:
            # Parse the response into a GameLevel object
            level = self._parse_response_to_level(response.content, width, height)
            
            # Store this generation for training data
            self._store_generation_data(prompt, response, level)
            
            return level, response
            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            # Return a fallback level
            return self._create_fallback_level(width, height), response
    
    def _create_level_prompt(self, width: int, height: int, difficulty: str, theme: str) -> str:
        return f"""
Generate a {theme} RPG level with the following specifications:
- Dimensions: {width}x{height}
- Difficulty: {difficulty}
- Theme: {theme}

Requirements:
1. All edges must be walls (B)
2. Exactly ONE player spawn point (P) near the center
3. Multiple enemies (E) placed strategically
4. Internal walls for interesting layout
5. All areas must be reachable by the player
6. Return as JSON with this structure:

{{
    "width": {width},
    "height": {height},
    "player_spawn": {{"x": int, "y": int}},
    "enemies": [{{"x": int, "y": int, "type": "basic"}}],
    "walls": [{{"start_x": int, "start_y": int, "end_x": int, "end_y": int}}],
    "terrain_map": [["B", ".", ".", ...], ["B", ".", "E", ...], ...]
}}

Make the level challenging but fair for {difficulty} difficulty.
"""
    
    def _parse_response_to_level(self, response: str, width: int, height: int) -> GameLevel:
        """Parse LLM JSON response into GameLevel object"""
        # Extract JSON from response (handle cases where LLM adds extra text)
        json_str = self._extract_json_from_response(response)
        data = json.loads(json_str)
        
        # Create Player object
        player = Player(
            spawn_position=Position(
                x=data["player_spawn"]["x"],
                y=data["player_spawn"]["y"]
            )
        )
        
        # Create Enemy objects
        enemies = [
            Enemy(
                position=Position(x=enemy["x"], y=enemy["y"]),
                enemy_type=enemy.get("type", "basic")
            )
            for enemy in data["enemies"]
        ]
        
        return GameLevel(
            width=width,
            height=height,
            player=player,
            enemies=enemies,
            walls=[],  # We'll extract from terrain_map
            terrain_map=data["terrain_map"]
        )
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON part from LLM response"""
        start = response.find('{')
        end = response.rfind('}') + 1
        return response[start:end]
    
    def _store_generation_data(self, prompt: str, response: LLMResponse, level: GameLevel):
        """Store generation data for training"""
        generation_data = {
            "prompt": prompt,
            "response": response.content,
            "model_used": response.model_name,
            "timestamp": response.timestamp,
            "tokens_used": response.tokens_used,
            "cost": response.cost_estimate,
            "parsed_level": level.model_dump(),
            "validation_passed": level.validate_level()
        }
        self.generation_history.append(generation_data)
    
    def _create_fallback_level(self, width: int, height: int) -> GameLevel:
        """Create a simple fallback level if LLM parsing fails"""
        # Create a simple level programmatically
        terrain_map = [['B'] * width for _ in range(height)]
        
        # Add floor space
        for y in range(1, height-1):
            for x in range(1, width-1):
                terrain_map[y][x] = '.'
        
        # Add player in center
        player_x, player_y = width//2, height//2
        terrain_map[player_y][player_x] = 'P'
        
        # Add some enemies
        enemies = []
        for i in range(3):
            ex, ey = (i+1)*3, (i+1)*2
            if ex < width-1 and ey < height-1:
                terrain_map[ey][ex] = 'E'
                enemies.append(Enemy(position=Position(x=ex, y=ey)))
        
        return GameLevel(
            width=width,
            height=height,
            player=Player(spawn_position=Position(x=player_x, y=player_y)),
            enemies=enemies,
            walls=[],
            terrain_map=terrain_map
        )