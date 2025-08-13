import requests
from bs4 import BeautifulSoup
import json
import os
from Cleaner_Arabic import ArabicCleaner

def scrape_bukhari_books(book_ids, output_dir="../CleanedData", output_file="bukhari_all_arabic_cleaned.json"):
    base_url = "https://sunnah.com/bukhari/"
    headers = {"User-Agent": "Mozilla/5.0"}
    cleaner = ArabicCleaner()
    all_hadiths = []

    for book_id in book_ids:
        url = f"{base_url}{book_id}"
        print(f"\n📖 Scraping book page: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"❌ Failed to load {url} (HTTP {response.status_code})")
                continue

            soup = BeautifulSoup(response.content, "html.parser")

            # ✅ Arabic book title
            book_title_div = soup.find("div", class_="book_page_colindextitle")
            book_title_ar = book_title_div.get_text(strip=True) if book_title_div else f"كتاب رقم {book_id}"
            # Remove English suffix if present (e.g., "كتاب الإيمان2Belief")
            book_title_ar = ''.join(filter(lambda c: not c.isascii() or c.isspace(), book_title_ar)).strip()

            hadith_blocks = soup.find_all("div", class_="actualHadithContainer")

            for block in hadith_blocks:
                # ✅ Arabic hadith text
                arabic_tag = block.find("div", class_="arabic_hadith_full")
                if not arabic_tag:
                    continue
                arabic_text = arabic_tag.get_text(strip=True)
                cleaned_text = cleaner.clean(arabic_text)

                # ✅ Arabic chapter title only
                chapter_title_ar = ""
                chapter_block = block.find_previous("div", class_="chapter")
                if chapter_block:
                    chapter_ar = chapter_block.find("div", class_="arabicchapter")
                    if chapter_ar:
                        chapter_title_ar = chapter_ar.get_text(strip=True)

                all_hadiths.append({
                    "book_id": book_id,
                    "book_title_ar": book_title_ar,
                    "chapter_title_ar": chapter_title_ar,
                    "cleaned_arabic": cleaned_text
                })

                print(f"✅ Hadith saved from book {book_id}")

        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, output_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_hadiths, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Finished: {len(all_hadiths)} hadiths saved to {path}")

if __name__ == "__main__":
    books_to_scrape = list(range(1, 98))  # Sahih al-Bukhari book IDs
    scrape_bukhari_books(books_to_scrape)
