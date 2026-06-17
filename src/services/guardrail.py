import re

class GuardrailService:
    def __init__(self):
        # Common prompt injection patterns and jailbreak signatures
        self.injection_patterns = [
            r"ignore\s+(any|all)?\s*previous\s+instructions",
            r"system\s+override",
            r"forget\s+(your|the)?\s*rules",
            r"you\s+are\s+now\s+a",
            r"disregard\s+above",
            r"bypass\s+restrictions",
            r"dan\s+mode",
            r"jailbreak",
            r"acting\s+as\s+a\s+terminal",
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.injection_patterns]

    def is_safe(self, text: str) -> bool:
        """Return True if the text is clean, False if an injection pattern is found."""
        clean_text = text.strip()
        
        for pattern in self.compiled_patterns:
            if pattern.search(clean_text):
                return False
                
        return True