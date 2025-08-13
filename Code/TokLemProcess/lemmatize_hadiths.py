import os
import json
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from tqdm import tqdm  # Progress bar

# Initialize MLE disambiguator
try:
    mle = MLEDisambiguator.pretrained()
except Exception as e:
    print("Error loading MLE disambiguator:", str(e))
    exit(1)

# Special case handling for Hadiths
HADITH_SPECIAL_LEMMAS = {
    "رسول": "رسل",
    "صلي": "صلى",
    "عليه": "على",
    "وسلم": "سلم"
}

def lemmatize_batch(tokens):
    """Hadith-specific lemmatization"""
    analyses = mle.disambiguate(tokens)
    return [
        HADITH_SPECIAL_LEMMAS.get(token,
        a.analyses[0].analysis['lex'] if a.analyses else token)
        for a, token in zip(analyses, tokens)
    ]

def process_hadiths():
    # Path configuration
    current_dir = os.path.dirname(__file__)
    input_path = os.path.abspath(os.path.join(current_dir, "..", "..", "CleanedData", "bukhari_all_arabic_cleaned.json"))
    output_path = os.path.abspath(os.path.join(current_dir, "..", "..", "CleanedData", "hadiths_lemmatized.json"))

    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    # Verify input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    # Load Hadiths data
    with open(input_path, "r", encoding="utf-8") as f:
        hadiths_data = json.load(f)

    lemmatized_hadiths = []
    
    # Process with progress bar
    for hadith in tqdm(hadiths_data, desc="Processing Hadiths"):
        tokens = simple_word_tokenize(hadith["cleaned_arabic"])
        lemmas = lemmatize_batch(tokens)
        
        processed_hadith = {
            "book_id": hadith["book_id"],
            "book_title_ar": hadith["book_title_ar"],
            "chapter_title_ar": hadith["chapter_title_ar"],
            "original_text": hadith["cleaned_arabic"],
            "tokens": tokens,
            "lemmas": lemmas
        }
        
        lemmatized_hadiths.append(processed_hadith)

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lemmatized_hadiths, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Successfully processed {len(lemmatized_hadiths)} hadiths")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    process_hadiths()