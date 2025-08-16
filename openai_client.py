
from langchain_openai import ChatOpenAI
from config.settings import Config

class OpenAIClient:
    def __init__(self):
        self.client = ChatOpenAI(api_key=Config.OPENAI_API_KEY, model_name=Config.OPENAI_MODEL)
        self.model = Config.OPENAI_MODEL

    def generate_response(self, messages, temperature=0.7, max_tokens=1000):
        """Generate response using OpenAI GPT-4o"""
        try:
            response = self.client.invoke(
                messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."

    def generate_structured_response(self, messages, response_format=None):
        """Generate structured response for routing decisions"""
        try:
            response = self.client.invoke(
                messages,
                temperature=0.1,  # Lower temperature for more consistent routing
            )
            return response.content
        except Exception as e:
            print(f"Error generating structured response: {e}")
            return None
