import openai
import time
from typing import Optional

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response with error handling"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    timeout=60
                )
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"    OpenAI error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise
        
        raise Exception(f"OpenAI failed after {max_retries} attempts")

class GeminiClient:
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response with error handling"""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                print(f"    Gemini error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise
        
        raise Exception(f"Gemini failed after {max_retries} attempts")