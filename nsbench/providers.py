"""One OpenAI-compatible client; OpenRouter covers nearly every hosted model. Keys via env only."""
import os
from openai import OpenAI


def client(base="openrouter"):
    if base == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def chat(model, messages, base="openrouter", max_tokens=16000, temperature=0.0):
    r = client(base).chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    u = r.usage
    return r.choices[0].message.content or "", (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
