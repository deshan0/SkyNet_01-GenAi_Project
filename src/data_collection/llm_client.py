from abc import ABC, abstractmethod
from typing import Dict, Any, List
import openai
import anthropic
import google.generativeai as genai
from dataclasses import dataclass
import json
import time

@dataclass
class LLMResponse:
    content: str
    model_name: str
    prompt: str
    timestamp: float
    tokens_used: int
    cost_estimate: float

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=self.model,
            prompt=prompt,
            timestamp=time.time(),
            tokens_used=response.usage.total_tokens,
            cost_estimate=self._calculate_cost(response.usage.total_tokens)
        )
    
    def _calculate_cost(self, tokens: int) -> float:
        # Rough cost estimation for GPT-4
        return tokens * 0.00003  # $0.03 per 1K tokens

# Similar classes for AnthropicClient and GeminiClient...