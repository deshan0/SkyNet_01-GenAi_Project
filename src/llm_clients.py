import openai
import google.generativeai as genai
from abc import ABC, abstractmethod
import time
import json

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano"):
        openai.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str) -> str:
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

class GeminiClient(LLMClient):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None