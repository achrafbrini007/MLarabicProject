from fastapi import FastAPI, Query
from search_engine import ArabicSearchEngine
import json

app = FastAPI()
search_engine = ArabicSearchEngine("config.json")

@app.get("/search/")
async def semantic_search(
    query: str = Query(..., min_length=3),
    search_type: str = Query("both", regex="^(quran|hadith|both)$"),
    top_k: int = Query(5, ge=1, le=20)
):
    return search_engine.search(query, search_type, top_k)

@app.get("/verse/{surah_idx}/{verse_idx}")
async def get_verse_details(surah_idx: int, verse_idx: int):
    with open(search_engine.config["data_paths"]["quran"], 'r', encoding='utf-8') as f:
        quran = json.load(f)
    return {
        "verse": quran[surah_idx]["verses"][verse_idx],
        "context": {
            "previous": quran[surah_idx]["verses"][verse_idx-1] if verse_idx > 0 else None,
            "next": quran[surah_idx]["verses"][verse_idx+1] if verse_idx < len(quran[surah_idx]["verses"])-1 else None
        }
    }