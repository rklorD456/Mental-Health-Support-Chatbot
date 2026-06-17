import re
import unicodedata


class GuardrailService:
    def __init__(self):
        # --- Prompt injection & jailbreak signatures ---
        self.injection_patterns = [
            # Original patterns
            r"ignore\s+(any|all)?\s*previous\s+instructions",
            r"system\s+override",
            r"forget\s+(your|the)?\s*rules",
            r"you\s+are\s+now\s+a",
            r"disregard\s+above",
            r"bypass\s+restrictions",
            r"dan\s+mode",
            r"jailbreak",
            r"acting\s+as\s+a\s+terminal",

            # Role/persona hijacking
            r"pretend\s+(you\s+are|to\s+be)",
            r"roleplay\s+as",
            r"act\s+as\s+(if\s+you\s+(are|were)|a)",
            r"simulate\s+(a\s+)?(different|unrestricted|evil)\s+(ai|model|assistant)",
            r"your\s+(true|real|hidden)\s+(self|personality|purpose)",

            # Instruction smuggling via delimiter tricks
            r"---+\s*(new\s+)?instructions",
            r"<<<+.*?(instructions|prompt|system).*?>>>+",
            r"\[INST\]",
            r"<\|system\|>",
            r"<\|user\|>",

            # Goal/context override
            r"new\s+(prime\s+)?directive",
            r"override\s+(all\s+)?(prior|previous|your)\s+(instructions|rules|constraints)",
            r"(ignore|disregard|forget)\s+(all\s+)?(prior|previous|your|the|above)\s+(instructions|rules|prompt|context|constraints)",
            r"(your|the)\s+(only|new|real|actual)\s+(goal|purpose|job|task|instruction)\s+(now\s+)?is",
            r"your\s+(only|new|real|actual)\s+(goal|purpose|job|task)\s+now",

            # Developer/debug mode tricks
            r"developer\s+mode",
            r"sudo\s+mode",
            r"god\s+mode",
            r"maintenance\s+mode",
            r"(enable|activate|enter|switch\s+to)\s+.{0,20}(mode|persona|personality)",

            # Payload exfiltration
            r"(print|repeat|output|reveal|show|display)\s+(your\s+)?(system\s+)?(prompt|instructions|rules|context)",
            r"what\s+(are\s+your|is\s+your\s+system)\s+(instructions|prompt|rules)",
        ]

        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in self.injection_patterns
        ]

        # --- Hard limits (length / structure) ---
        self.max_input_length = 2000   # characters
        self.max_line_count   = 40     # unusually structured inputs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_safe(self, text: str) -> bool:
        """Return True if the text is safe, False if a threat is detected.

        Runs four independent checks in order of cost (cheapest first):
          1. Length / structural limits
          2. Unicode normalisation + homoglyph collapse
          3. Whitespace-collapse normalisation
          4. Regex pattern matching
        """
        if not isinstance(text, str) or not text.strip():
            return True  # empty input is not an injection

        if not self._check_limits(text):
            return False

        normalised = self._normalise(text)

        if self._matches_any_pattern(normalised):
            return False

        return True

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_limits(self, text: str) -> bool:
        """Reject inputs that are suspiciously long or highly structured."""
        if len(text) > self.max_input_length:
            return False
        if text.count("\n") > self.max_line_count:
            return False
        return True

    def _normalise(self, text: str) -> str:
        """Collapse tricks that try to defeat pattern matching.

        Steps:
          - NFKC unicode normalisation (resolves homoglyphs like ｉ→i, ０→0)
          - Strip zero-width / invisible characters
          - Collapse repeated whitespace and punctuation separators
            (e.g. "i g n o r e" → "ignore", "i.g.n.o.r.e" → "ignore")
        """
        # 1. Unicode normalise
        text = unicodedata.normalize("NFKC", text)

        # 2. Remove zero-width and other invisible characters
        text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]", "", text)

        # 3. Collapse letter-by-letter spacing tricks: "i g n o r e" / "i.g.n.o.r.e" → "ignore"
        text = re.sub(r"\b(\w)([\s.\-_,]+\w){3,}\b", lambda m: re.sub(r"[\s.\-_,]", "", m.group()), text)

        # 4. Collapse any remaining runs of whitespace to a single space
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _matches_any_pattern(self, text: str) -> bool:
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        return False