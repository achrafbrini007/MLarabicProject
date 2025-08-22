from search_engine import ArabicSearchEngine
from pathlib import Path

MIN_SCORE_QURAN = 0.5
MIN_SCORE_HADITH = 0.5
MAX_TEXT_LEN = 200  # truncate long lines for console readability

def truncate(s: str, n: int = MAX_TEXT_LEN) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def print_quran_hits(hits, min_score=MIN_SCORE_QURAN):
    if not hits:
        return
    print("\n📖 Quran Matches:")
    shown_any = False
    for i, m in enumerate(hits, 1):
        score = float(m.get("score", 0.0))
        if score < min_score:
            continue
        text = truncate(m.get("text", ""))
        meta = m.get("metadata", {}) or {}
        citation = meta.get("citation")
        # Fallbacks (shouldn’t be needed if engine filled citation)
        if not citation:
            sname = meta.get("surah_name")
            s = meta.get("surah")
            a = meta.get("ayah")
            citation = f"Qur'an {sname} ({s}):{a}" if sname else f"Qur'an {s}:{a}"
        print(f"{i}. {text}")
        print(f"   → {citation}")
        print(f"   → Relevance: {score:.3f}")
        shown_any = True
    if not shown_any:
        print("   (no items above threshold)")

def print_hadith_hits(hits, min_score=MIN_SCORE_HADITH):
    if not hits:
        return
    print("\n📚 Hadith Matches:")
    shown_any = False
    for i, m in enumerate(hits, 1):
        score = float(m.get("score", 0.0))
        if score < min_score:
            continue
        text = truncate(m.get("text", ""))
        meta = m.get("metadata", {}) or {}
        # Show only the book title (Arabic); engine supplies this in 'citation'
        citation = meta.get("book_title_ar") or meta.get("citation") or "صحيح البخاري"
        print(f"{i}. {text}")
        print(f"   → {citation}")
        print(f"   → Relevance: {score:.3f}")
        shown_any = True
    if not shown_any:
        print("   (no items above threshold)")

def main():
    try:
        print("Initializing Arabic Search Engine...")
        config_path = Path(__file__).parent.parent / "config.json"
        print(f"Looking for config at: {config_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        engine = ArabicSearchEngine(str(config_path))
        print("Engine ready. Running sample searches...\n")

        queries = [
            "الرحمة في الإسلام",
            "الصبر على المصائب",
            "فضل الزكاة والصدقة",
        ]

        for query in queries:
            print(f"\nSearch Query: '{query}'")
            print("=" * 60)
            results = engine.search(query, search_type="both", top_k=5)

            if not results:
                print("No results found")
                print("=" * 60)
                continue

            print_quran_hits(results.get("quran", []), min_score=MIN_SCORE_QURAN)
            print_hadith_hits(results.get("hadith", []), min_score=MIN_SCORE_HADITH)
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
