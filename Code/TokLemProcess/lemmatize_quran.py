import os
import json
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from tqdm import tqdm

class QuranLemmatizer:
    def __init__(self):
        # Initialize with enhanced special cases
        self.SPECIAL_LEMMAS = {
            "الرحمن": "رحم",
            "الرحيم": "رحم",
            "العلمين": "علم",
            "ليضلون": "ضل",
            "الشيطين": "شيطان",
            "الظلمت": "ظلمة",
        }
        
        try:
            self.mle = MLEDisambiguator.pretrained()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MLE disambiguator: {str(e)}")

    def lemmatize_token(self, token):
        """Enhanced lemmatization with special case priority"""
        if token in self.SPECIAL_LEMMAS:
            return self.SPECIAL_LEMMAS[token]
        
        try:
            analyses = self.mle.disambiguate([token])
            if analyses and analyses[0].analyses:
                return analyses[0].analyses[0].analysis['lex']
            return token
        except Exception:
            return token  # Fallback to original token

    def process_verse(self, verse_text):
        """Process a single verse with proper tokenization"""
        tokens = simple_word_tokenize(verse_text)
        return {
            "tokens": tokens,
            "lemmas": [self.lemmatize_token(t) for t in tokens]
        }

def get_file_paths():
    """Handle path resolution cross-platform"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "..", "..", "CleanedData", "quran_cleaned_arabic.json")
    output_path = os.path.join(current_dir, "..", "..", "CleanedData", "quran_lemmatized_enhanced.json")
    
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return input_path, output_path

def validate_input_file(input_path):
    """Validate input JSON structure"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Invalid Quran JSON format: Expected list of Surahs")
        return data

def process_quran():
    try:
        input_path, output_path = get_file_paths()
        quran_data = validate_input_file(input_path)
        lemmatizer = QuranLemmatizer()
        
        enhanced_quran = []
        
        for surah in tqdm(quran_data, desc="Processing Surahs"):
            processed_surah = {
                "surahName": surah["surahName"],
                "verses": []
            }
            
            for verse in tqdm(surah["verses"], desc=f"Processing {surah['surahName']}", leave=False):
                verse_data = lemmatizer.process_verse(verse)
                processed_surah["verses"].append({
                    "original": verse,
                    "tokens": verse_data["tokens"],
                    "lemmas": verse_data["lemmas"],
                    "surah": surah["surahName"],  # Added context
                    "verseNumber": surah["verses"].index(verse) + 1  # Add verse number
                })
            
            enhanced_quran.append(processed_surah)
        
        # Save with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_quran, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Success! Processed {len(enhanced_quran)} Surahs")
        print(f"Saved to: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    process_quran()