import re

class ArabicCleaner:
    def __init__(self):
        # Arabic diacritics + Qur’anic symbols (tashkeel, sukun, maddah, etc.)
        self.diacritics = re.compile(
            r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0670\u08D4-\u08E1\u08E3-\u08FF]'
        )

    def clean(self, text: str) -> str:
        # Remove diacritics and Qur’anic symbols
        text = self.diacritics.sub('', text)

        # Remove RTL control character (U+200F)
        text = text.replace('\u200f', '')

        # Normalize characters
        text = re.sub(r'[إأآٱا]', 'ا', text)  # all alifs
        text = re.sub(r'ى', 'ي', text)        # alef maqsura → ya
        text = re.sub(r'ة', 'ه', text)        # ta marbuta → ha
        text = re.sub(r'ؤ', 'و', text)        # waw hamza → waw
        text = re.sub(r'ئ', 'ي', text)        # ya hamza → ya

        # Fix extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
