from typing import Dict, Any, Optional
import time
import json
from .local_model_manager import LocalModelManager
from .llm_clients import OpenAIClient, GeminiClient

class ModelRouter:
    def __init__(self, 
                 openai_key: Optional[str] = None,
                 gemini_key: Optional[str] = None,
                 local_model_path: str = "models/rpg_model_final"):
        
        # Initialize local model
        self.local_manager = LocalModelManager(local_model_path)
        
        # Initialize API clients
        self.api_clients = {}
        if openai_key:
            self.api_clients["openai"] = OpenAIClient(openai_key)
        if gemini_key:
            self.api_clients["gemini"] = GeminiClient(gemini_key)
        
        # Routing configuration
        self.confidence_threshold = 0.7  # Confidence threshold for using local model
        self.cost_per_api_call = 0.002   # Rough cost estimate
        
        print(f"🧠 ModelRouter initialized")
        print(f"📱 Local model: Available")
        print(f"🌐 API models: {list(self.api_clients.keys())}")
    
    def generate_level(self, 
                      width: int, 
                      height: int, 
                      difficulty: str, 
                      theme: str,
                      strategy: str = "auto") -> Dict[str, Any]:
        """
        Generate RPG level using smart routing
        
        Strategies:
        - "auto": Smart routing based on confidence
        - "local": Force local model
        - "api": Force API model (if available)
        - "compare": Generate with both for comparison
        """
        
        if strategy == "local":
            return self._generate_local(width, height, difficulty, theme)
        
        elif strategy == "api":
            return self._generate_api(width, height, difficulty, theme)
        
        elif strategy == "compare":
            return self._generate_compare(width, height, difficulty, theme)
        
        else:  # auto
            return self._generate_auto(width, height, difficulty, theme)
    
    def _generate_local(self, width: int, height: int, difficulty: str, theme: str) -> Dict[str, Any]:
        """Generate using local fine-tuned model"""
        print("🤖 Using local fine-tuned model...")
        
        result = self.local_manager.generate_level(width, height, difficulty, theme)
        
        return {
            **result,
            "strategy_used": "local",
            "cost": 0.0,
            "confidence": self._calculate_confidence(result)
        }
    
    def _generate_api(self, width: int, height: int, difficulty: str, theme: str) -> Dict[str, Any]:
        """Generate using API model"""
        if not self.api_clients:
            print("⚠️ No API clients available, falling back to local")
            return self._generate_local(width, height, difficulty, theme)
        
        # Use first available API client
        client_name = list(self.api_clients.keys())[0]
        client = self.api_clients[client_name]
        
        print(f"🌐 Using API model: {client_name}")
        
        prompt = self._create_api_prompt(width, height, difficulty, theme)
        
        start_time = time.time()
        response = client.generate(prompt)
        generation_time = time.time() - start_time
        
        # Parse response
        level_data = self._parse_api_response(response)
        
        return {
            "raw_response": response,
            "parsed_level": level_data,
            "generation_time": generation_time,
            "model_used": f"api_{client_name}",
            "success": level_data is not None,
            "strategy_used": "api",
            "cost": self.cost_per_api_call,
            "confidence": 0.9 if level_data else 0.1
        }
    
    def _generate_auto(self, width: int, height: int, difficulty: str, theme: str) -> Dict[str, Any]:
        """Smart routing: try local first, fallback to API if confidence is low"""
        
        # Try local model first
        local_result = self._generate_local(width, height, difficulty, theme)
        confidence = local_result["confidence"]
        
        print(f"🎯 Local model confidence: {confidence:.2f}")
        
        if confidence >= self.confidence_threshold:
            print("✅ Using local model result (high confidence)")
            return {**local_result, "strategy_used": "auto_local"}
        
        elif self.api_clients:
            print("⚠️ Low confidence, trying API model...")
            api_result = self._generate_api(width, height, difficulty, theme)
            return {
                **api_result, 
                "strategy_used": "auto_api",
                "local_confidence": confidence,
                "fallback_reason": "low_confidence"
            }
        
        else:
            print("⚠️ Low confidence but no API available, using local result")
            return {**local_result, "strategy_used": "auto_local_only"}
    
    def _generate_compare(self, width: int, height: int, difficulty: str, theme: str) -> Dict[str, Any]:
        """Generate with both local and API models for comparison"""
        print("🔍 Generating with both models for comparison...")
        
        local_result = self._generate_local(width, height, difficulty, theme)
        
        if self.api_clients:
            api_result = self._generate_api(width, height, difficulty, theme)
            
            return {
                "strategy_used": "compare",
                "local_result": local_result,
                "api_result": api_result,
                "comparison": {
                    "local_confidence": local_result["confidence"],
                    "api_confidence": api_result["confidence"],
                    "cost_saved": api_result["cost"],
                    "speed_difference": api_result["generation_time"] - local_result["generation_time"]
                }
            }
        else:
            return {
                "strategy_used": "compare_local_only",
                "local_result": local_result,
                "api_result": None,
                "comparison": {"note": "No API models available"}
            }
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for local model result"""
        if not result["success"]:
            return 0.1
        
        confidence = 0.5  # Base confidence
        
        # Bonus for successful JSON parsing
        if result["parsed_level"]:
            confidence += 0.3
            
            level = result["parsed_level"]
            
            # Check required fields
            required_fields = ["width", "height", "difficulty", "theme", "player_spawn", "enemies", "terrain_map"]
            for field in required_fields:
                if field in level:
                    confidence += 0.02
            
            # Check terrain map completeness
            if "terrain_map" in level and isinstance(level["terrain_map"], list):
                if len(level["terrain_map"]) > 3:  # Not just placeholder
                    confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _create_api_prompt(self, width: int, height: int, difficulty: str, theme: str) -> str:
        """Create prompt for API models"""
        return f"""Generate a {theme} RPG level with these exact specifications:
- Dimensions: {width} x {height}
- Difficulty: {difficulty}
- Theme: {theme}

Return ONLY a JSON object with this exact structure:
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
        ["B", "B", "B", "..."],
        ["B", ".", ".", "..."]
    ]
}}"""
    
    def _parse_api_response(self, response: str) -> Optional[Dict]:
        """Parse API response"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                return None
            
            json_str = response[start:end]
            return json.loads(json_str)
            
        except Exception as e:
            print(f"⚠️ API response parse error: {e}")
            return None