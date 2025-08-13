from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class TerrainType(Enum):
    WALL = "B"
    FLOOR = "."
    PLAYER_SPAWN = "P"
    ENEMY_SPAWN = "E"

class Position(BaseModel):
    x: int
    y: int

class Enemy(BaseModel):
    position: Position
    enemy_type: str = "basic"
    health: int = 100
    damage: int = 20

class Player(BaseModel):
    spawn_position: Position
    health: int = 100

class Wall(BaseModel):
    start_pos: Position
    end_pos: Position

class GameLevel(BaseModel):
    width: int
    height: int
    player: Player
    enemies: List[Enemy]
    walls: List[Wall]
    terrain_map: List[List[str]]
    
    def to_string_map(self) -> List[str]:
        """Convert to string representation like your existing maps"""
        return [''.join(row) for row in self.terrain_map]
    
    def validate_level(self) -> bool:
        """Basic validation - we'll expand this"""
        # Check if player exists
        if not self.player:
            return False
        
        # Check if enemies exist
        if len(self.enemies) == 0:
            return False
        
        # Check boundaries
        if (self.player.spawn_position.x >= self.width or 
            self.player.spawn_position.y >= self.height):
            return False
        
        return True