from search_engine import ArabicSearchEngine
import json
from pathlib import Path

def main():
    try:
        print("Initializing Arabic Search Engine...")
        
        # ✅ FIXED: Go up one level to AraBERTPipeline folder
        config_path = Path(__file__).parent.parent / "config.json"
        print(f"Looking for config at: {config_path}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        engine = ArabicSearchEngine(str(config_path))
        print("Engine ready. Running sample searches...\n")
        
        queries = [
            "الرحمة في الإسلام",
            "الصبر على المصائب", 
            "فضل الزكاة والصدقة"
        ]
        
        for query in queries:
            print(f"\nSearch Query: '{query}'")
            print("=" * 60)
            
            results = engine.search(query, search_type="both", top_k=5)
            
            if not results:
                print("No results found")
                continue
                
            # Quran results - LOWERED THRESHOLD from 0.7 to 0.5
            if "quran" in results and results["quran"]:
                print("\n📖 Quran Matches:")
                for idx, match in enumerate(results["quran"], 1):
                    if match["score"] > 0.5:  # ✅ CHANGED FROM 0.7 to 0.5
                        print(f"{idx}. {match['text']}")
                        print(f"   → Surah: {match['metadata']['surah']}")
                        print(f"   → Relevance: {match['score']:.3f}")
                    else:
                        print(f"{idx}. [FILTERED] Score too low: {match['score']:.3f}")
            
            # Hadith results - LOWERED THRESHOLD from 0.6 to 0.5
            if "hadith" in results and results["hadith"]:
                print("\n📚 Hadith Matches:")
                for idx, match in enumerate(results["hadith"], 1):
                    if match["score"] > 0.5:  # ✅ CHANGED FROM 0.6 to 0.5
                        hadith_text = match["text"]
                        if len(hadith_text) > 120:
                            hadith_text = hadith_text[:117] + "..."
                        print(f"{idx}. {hadith_text}")
                        print(f"   → Book: {match['metadata']['book']}")
                        print(f"   → Chapter: {match['metadata']['chapter']}")
                        print(f"   → Relevance: {match['score']:.3f}")
                    else:
                        print(f"{idx}. [FILTERED] Score too low: {match['score']:.3f}")
            
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()