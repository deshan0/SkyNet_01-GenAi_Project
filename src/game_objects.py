from dataclasses import dataclass
from typing import List

@dataclass
class Position:
    x: int
    y: int

@dataclass
class MapTiles:
    width: int
    height: int
    walls: List[Position]
    enemies: List[Position]
    player_spawn: Position  # Changed from player_pos to player_spawn
    
    def validate(self) -> bool:
        """Basic validation for the map"""
        # Check if player is within bounds
        if not (0 <= self.player_spawn.x < self.width and 0 <= self.player_spawn.y < self.height):
            return False
        
        # Check if all walls are within bounds
        for wall in self.walls:
            if not (0 <= wall.x < self.width and 0 <= wall.y < self.height):
                return False
        
        # Check if all enemies are within bounds
        for enemy in self.enemies:
            if not (0 <= enemy.x < self.width and 0 <= enemy.y < self.height):
                return False
        
        return True