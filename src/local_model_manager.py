import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

class LocalModelManager:
    def __init__(self, model_path: str = "models/rpg_model_final"):
        self.model_path = Path(model_path)
        self.base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 LocalModelManager initialized")
        print(f"📁 Model path: {self.model_path}")
        print(f"💻 Device: {self.device}")
        
    def load_model(self):
        """Load the fine-tuned model"""
        if self.model is not None:
            print("✅ Model already loaded")
            return
        
        print("🔄 Loading fine-tuned model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Load fine-tuned weights
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_level(self, 
                      width: int, 
                      height: int, 
                      difficulty: str, 
                      theme: str,
                      max_tokens: int = 800,
                      temperature: float = 0.7) -> Dict[str, Any]:
        """Generate RPG level using the fine-tuned model"""
        
        if self.model is None:
            self.load_model()
        
        # Create prompt
        prompt = self._create_prompt(width, height, difficulty, theme)
        
        # Format for chat
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        start_time = time.time()
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        assistant_start = generated_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        response = generated_text[assistant_start:].strip()
        
        # Try to parse JSON
        level_data = self._parse_response(response)
        
        return {
            "raw_response": response,
            "parsed_level": level_data,
            "generation_time": generation_time,
            "model_used": "local_fine_tuned",
            "success": level_data is not None
        }
    
    def _create_prompt(self, width: int, height: int, difficulty: str, theme: str) -> str:
        """Create prompt for level generation"""
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
}}

Rules:
1. All edges must be walls (B)
2. Player spawn (P) should be marked in terrain_map
3. Enemies (E) should be marked in terrain_map
4. Use: B=wall, .=floor, P=player, E=enemy"""
    
    def _parse_response(self, response: str) -> Optional[Dict]:
        """Try to parse JSON from response"""
        try:
            # Extract JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                return None
            
            json_str = response[start:end]
            return json.loads(json_str)
            
        except Exception as e:
            print(f"⚠️ JSON parse error: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "device": self.device,
            "base_model": self.base_model_name,
            "parameters": "12.6M trainable (LoRA)"
        }