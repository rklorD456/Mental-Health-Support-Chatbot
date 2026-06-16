class TranslatorService:
    def __init__(self, llm_client, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name

    def to_english(self, text: str) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "translate the input text into English. output ONLY the final English translation."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] translation to english failed: {e}")
            return text

    def from_english(self, text: str, target_language: str) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": f"translate the following mental health response into: '{target_language}'. output ONLY the translation."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] translation from english failed: {e}")
            return text