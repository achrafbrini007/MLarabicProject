import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
from typing import List, Dict

class ArabicSearchEngine:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.model = SentenceTransformer(
            self.config["model_settings"]["model_name"],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.quran_data = self._load_data("quran")
        self.hadith_data = self._load_data("hadiths")
        
        print(f"Loaded {len(self.quran_data)} surahs and {len(self.hadith_data)} hadiths")
        
        self.quran_embeddings = self._precompute_embeddings(self.quran_data)
        self.hadith_embeddings = self._precompute_embeddings(self.hadith_data)

    def _load_data(self, data_type: str) -> List[Dict]:
        project_root = Path("C:/Users/pc/Desktop/PFAarabicProject")
        
        file_patterns = {
            "quran": ["quran_lemmatized_enhanced.json", "quran_lemmatized.json"],
            "hadiths": ["hadiths_lemmatized.json"]
        }
        
        for filename in file_patterns[data_type]:
            path = project_root / "CleanedData" / filename
            print(f"Trying {data_type} file: {path}")
            if path.exists():
                print(f"Found {data_type} file: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        data_dir = project_root / "CleanedData"
        if data_dir.exists():
            available_files = list(data_dir.glob("*.json"))
            print(f"Available files in {data_dir}: {available_files}")
        
        raise FileNotFoundError(f"No {data_type} file found in {data_dir}")

    def _precompute_embeddings(self, data: List[Dict]) -> np.ndarray:
        texts = []
        for item in data:
            if "verses" in item:
                texts.extend([v["original"] for v in item["verses"]])
            else:
                texts.append(item["original_text"])
        
        print(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.config["model_settings"]["batch_size"],
            show_progress_bar=True
        )
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def expand_islamic_query(self, query):
        """Enhanced Islamic query expansion"""
        islamic_expansion = {
            "الرحمة": "الرحمن الرحيم رحمة واسعة المغفرة العفو",
            "الصبر": "الصابرين المصائب البلاء الاحتساب الثواب الأجر الصمود التحمل",
            "الزكاة": "الصدقة الفقراء المساكين المحتاجين الإنفاق التطوع",
            "المصائب": "البلاء المصائب الشدائد الصبر الاحتساب الثواب المحن",
            "الإسلام": "الإيمان التوحيد المسلمون الدين الله الرسول",
            "الفضل": "الثواب الأجر الخير البركة الفضائل المكافأة",
            "الصدقة": "التبرع العطاء المساعدة الفقراء المحتاجين البر",
            "في": "", "على": "", "من": "", "ال": "", "وا": ""
        }
        
        expanded = query
        for term, expansion in islamic_expansion.items():
            if term in query:
                expanded += " " + expansion
        return expanded.strip()

    def search(self, query: str, search_type: str = "both", top_k: int = 10) -> Dict:
        # Auto-expand the query
        original_query = query
        query = self.expand_islamic_query(query)
        print(f"Expanded query: '{query}'")
        
        query_embedding = self.model.encode(query)
        results = {}
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if search_type in ("quran", "both"):
            scores = np.dot(self.quran_embeddings, query_embedding)
            scores = np.clip(scores, -1.0, 1.0)
            top_indices = np.argsort(scores)[-top_k*5:][::-1]
            results["quran"] = []
            
            for idx in top_indices:
                verse_count = 0
                for surah_idx, surah in enumerate(self.quran_data):
                    if idx < verse_count + len(surah["verses"]):
                        verse_idx = idx - verse_count
                        verse_text = surah["verses"][verse_idx]["original"]
                        
                        # Relaxed filter: from 4 to 3 words
                        if len(verse_text.split()) < 3:
                            continue
                            
                        # Boost for relevance
                        boosted_score = float(scores[idx])
                        query_words = original_query.split()
                        for word in query_words:
                            if word in verse_text and len(word) > 2:
                                boosted_score = min(1.0, boosted_score * 2.0)
                                break
                        
                        # Extra boost for expanded terms
                        expanded_terms = query.split()
                        for term in expanded_terms:
                            if term in verse_text and len(term) > 2:
                                boosted_score = min(1.0, boosted_score * 1.3)
                                break
                        
                        results["quran"].append({
                            "text": verse_text,
                            "score": boosted_score,
                            "metadata": {
                                "surah": surah["surahName"],
                                "verse_index": verse_idx
                            }
                        })
                        break
                    verse_count += len(surah["verses"])
            
            results["quran"].sort(key=lambda x: x["score"], reverse=True)
            results["quran"] = results["quran"][:top_k]
        
        if search_type in ("hadith", "both"):
            scores = np.dot(self.hadith_embeddings, query_embedding)
            scores = np.clip(scores, -1.0, 1.0)
            top_indices = np.argsort(scores)[-top_k*3:][::-1]
            results["hadith"] = []
            
            for idx in top_indices:
                hadith_text = self.hadith_data[idx]["original_text"]
                
                boosted_score = float(scores[idx])
                query_words = original_query.split()
                for word in query_words:
                    if word in hadith_text and len(word) > 2:
                        boosted_score = min(1.0, boosted_score * 1.8)
                        break
                
                expanded_terms = query.split()
                for term in expanded_terms:
                    if term in hadith_text and len(term) > 2:
                        boosted_score = min(1.0, boosted_score * 1.2)
                        break
                
                results["hadith"].append({
                    "text": hadith_text,
                    "score": boosted_score,
                    "metadata": {
                        "book": self.hadith_data[idx]["book_title_ar"],
                        "chapter": self.hadith_data[idx]["chapter_title_ar"]
                    }
                })
            
            results["hadith"].sort(key=lambda x: x["score"], reverse=True)
            results["hadith"] = results["hadith"][:top_k]
        
        return results