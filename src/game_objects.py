from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Position:
    x: int
    y: int
    
    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y}

@dataclass
class Enemy:
    position: Position
    enemy_type: str = "basic"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.position.x,
            "y": self.position.y,
            "type": self.enemy_type
        }

@dataclass
class GameLevel:
    width: int
    height: int
    difficulty: str
    theme: str
    player_spawn: Position
    enemies: List[Enemy]
    walls: List[Position]
    terrain_map: List[List[str]]
    
    def validate(self) -> bool:
        """Basic validation of the game level"""
        try:
            # Check dimensions
            if self.width <= 0 or self.height <= 0:
                return False
            
            # Check terrain map dimensions
            if len(self.terrain_map) != self.height:
                return False
            
            if any(len(row) != self.width for row in self.terrain_map):
                return False
            
            # Check player spawn is within bounds
            if not (0 <= self.player_spawn.x < self.width and 0 <= self.player_spawn.y < self.height):
                return False
            
            # Check enemies are within bounds
            for enemy in self.enemies:
                if not (0 <= enemy.position.x < self.width and 0 <= enemy.position.y < self.height):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "difficulty": self.difficulty,
            "theme": self.theme,
            "player_spawn": self.player_spawn.to_dict(),
            "enemies": [enemy.to_dict() for enemy in self.enemies],
            "terrain_map": self.terrain_map
        }