import requests
import json
import os
from Cleaner_Arabic import ArabicCleaner

def fetch_and_clean_quran(output_dir="CleanedData", output_file="quran_cleaned_arabic.json"):
    URL = "https://quranapi.pages.dev/api/arabic1.json"  # Arabic with tashkeel
    response = requests.get(URL)

    if response.status_code != 200:
        print(f"❌ Failed to fetch Quran: {response.status_code}")
        return

    data = response.json()
    cleaner = ArabicCleaner()
    result = []

    for surah in data:
        name = surah['surahNameArabic']
        verses = surah['translation']  # this is a list of ayah texts

        cleaned_verses = [cleaner.clean(v) for v in verses]

        result.append({
            "surahName": name,
            "verses": cleaned_verses
        })

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(result)} surahs cleaned and saved to {output_path}")
