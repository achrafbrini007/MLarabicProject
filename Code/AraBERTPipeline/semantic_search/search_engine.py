import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any

class ArabicSearchEngine:
    def __init__(self, config_path: str):
        with open(config_path, encoding='utf-8') as f:
            self.config = json.load(f)

        self.model = SentenceTransformer(
            self.config["model_settings"]["model_name"],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.quran_data  = self._load_data("quran")
        self.hadith_data = self._load_data("hadiths")

        print(f"Loaded {len(self.quran_data)} Quran items and {len(self.hadith_data)} Hadith items")

        # Flatten to texts + metas, then embed
        (self.quran_texts,
         self.quran_metas)  = self._flatten_quran(self.quran_data)

        (self.hadith_texts,
         self.hadith_metas) = self._flatten_hadith(self.hadith_data)

        self.quran_embeddings  = self._embed_and_normalize(self.quran_texts, "quran")
        self.hadith_embeddings = self._embed_and_normalize(self.hadith_texts, "hadith")

    # ---------- IO ----------
    def _load_data(self, data_type: str) -> List[Dict]:
        project_root = Path("C:/Users/pc/Desktop/PFAarabicProject")
        file_patterns = {
            "quran":   ["quran_cleaned_arabic.json", "quran_ceaned_arabic.json"],
            "hadiths": ["bukhari_all_arabic_cleaned.json", "hadiths_lemmatized.json"]
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

    # ---------- Flatteners (build parallel texts+metas) ----------
    def _flatten_quran(self, data: List[Union[Dict, str]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Your Qur’an file looks like:
        [
          {"surahName": "الفاتحة", "verses": ["بسم الله...", ...]},
          {"surahName": "البقرة",  "verses": [...]},
          ...
        ]
        We derive surah number from list index (1-based) and ayah from verse index (1-based).
        """
        texts, metas = [], []
        if not data:
            return texts, metas

        # Case A: flat list of strings (fallback)
        if isinstance(data[0], str):
            for i, t in enumerate(data):
                t = (t or "").strip()
                if not t:
                    continue
                meta = {
                    "source": "quran",
                    "surah": None,
                    "surah_name": None,
                    "ayah": i + 1,
                    "citation": f"Qur'an (index {i})"
                }
                texts.append(t)
                metas.append(meta)
            return texts, metas

        # Case B: list of dicts with surahName + verses
        for s_idx, surah_obj in enumerate(data, start=1):  # 1-based surah number
            surah_name = (surah_obj.get("surahName")
                          or surah_obj.get("name")
                          or surah_obj.get("surah_name_ar"))
            verses = surah_obj.get("verses") or []
            if not isinstance(verses, list):
                verses = [verses]

            for a_idx, v in enumerate(verses, start=1):  # 1-based ayah number
                verse_text = (v or "").strip() if isinstance(v, str) else str(v).strip()
                if not verse_text or len(verse_text.split()) < 2:
                    continue

                citation = f"Qur'an {surah_name} ({s_idx}):{a_idx}" if surah_name else f"Qur'an {s_idx}:{a_idx}"
                meta = {
                    "source": "quran",
                    "surah": s_idx,
                    "surah_name": surah_name,
                    "ayah": a_idx,
                    "citation": citation
                }
                texts.append(verse_text)
                metas.append(meta)

        return texts, metas

    def _flatten_hadith(self, data: List[Union[Dict, str]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Your Bukhari items look like dicts with:
           book_id, book_title_ar, chapter_title_ar, cleaned_arabic
           We will use cleaned_arabic as text and citation = book_title_ar (only)."""
        texts, metas = [], []
        if not data:
            return texts, metas

        # Case A: flat strings (fallback)
        if isinstance(data[0], str):
            for i, t in enumerate(data):
                t = (t or "").strip()
                if not t:
                    continue
                meta = {
                    "source": "hadith",
                    "collection": "Bukhari",
                    "book_title_ar": None,
                    "book_id": None,
                    "citation": "صحيح البخاري"
                }
                texts.append(t)
                metas.append(meta)
            return texts, metas

        # Case B: dicts with book_title_ar / cleaned_arabic
        for h in data:
            text = (h.get("cleaned_arabic")
                    or h.get("original_text")
                    or h.get("text")
                    or h.get("hadith_text")
                    or "").strip()
            if not text or len(text.split()) < 2:
                continue

            book_title_ar = h.get("book_title_ar")  # e.g., "كتاب الزكاة"
            book_id = h.get("book_id")

            meta = {
                "source": "hadith",
                "collection": "Bukhari",
                "book_title_ar": book_title_ar,
                "book_id": book_id,
                # Display only the Arabic book title as you requested
                "citation": book_title_ar or "صحيح البخاري"
            }
            texts.append(text)
            metas.append(meta)

        return texts, metas

    # ---------- Embeddings ----------
    def _embed_and_normalize(self, texts: List[str], label: str) -> np.ndarray:
        print(f"\nDebug: Computing embeddings for {label}… ({len(texts)} texts)")
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=self.config["model_settings"]["batch_size"],
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # L2 normalize
        return emb.astype(np.float32)

    # ---------- Query expansion ----------
    def expand_islamic_query(self, query: str) -> str:
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
        expanded = [query]
        for term, exp in islamic_expansion.items():
            if term in query and exp:
                expanded.append(exp)
        return " ".join(expanded).strip()

    # ---------- Search ----------
    def search(self, query: str, search_type: str = "both", top_k: int = 10) -> Dict:
        original_query = query
        expanded_query = self.expand_islamic_query(query)
        print(f"Search Query: '{original_query}'")
        print("="*60)
        print(f"Original query: '{original_query}'")
        print(f"Expanded query: '{expanded_query}'")

        qv = self.model.encode([expanded_query], convert_to_numpy=True)[0]
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        results = {}

        # ---- Quran ----
        if search_type in ("quran", "both") and len(self.quran_embeddings) > 0:
            scores = np.clip(self.quran_embeddings @ qv, -1.0, 1.0)
            top_idx = np.argsort(scores)[-top_k*5:][::-1]

            hits = []
            for idx in top_idx:
                verse_text = self.quran_texts[idx]
                meta = dict(self.quran_metas[idx])  # copy

                # Boosting (light)
                boosted = float(scores[idx])
                for w in original_query.split():
                    if len(w) > 2 and w in verse_text:
                        boosted = min(1.0, boosted * 2.0)
                        break
                for t in expanded_query.split():
                    if len(t) > 2 and t in verse_text:
                        boosted = min(1.0, boosted * 1.3)
                        break

                meta.setdefault("citation", self._format_citation(meta))
                hits.append({"text": verse_text, "score": boosted, "metadata": meta})

            hits.sort(key=lambda x: x["score"], reverse=True)
            results["quran"] = hits[:top_k]

        # ---- Hadith ----
        if search_type in ("hadith", "both") and len(self.hadith_embeddings) > 0:
            scores = np.clip(self.hadith_embeddings @ qv, -1.0, 1.0)
            top_idx = np.argsort(scores)[-top_k*3:][::-1]

            hits = []
            for idx in top_idx:
                hadith_text = self.hadith_texts[idx]
                meta = dict(self.hadith_metas[idx])

                boosted = float(scores[idx])
                for w in original_query.split():
                    if len(w) > 2 and w in hadith_text:
                        boosted = min(1.0, boosted * 1.8)
                        break
                for t in expanded_query.split():
                    if len(t) > 2 and t in hadith_text:
                        boosted = min(1.0, boosted * 1.2)
                        break

                meta.setdefault("citation", self._format_citation(meta))
                hits.append({"text": hadith_text, "score": boosted, "metadata": meta})

            hits.sort(key=lambda x: x["score"], reverse=True)
            results["hadith"] = hits[:top_k]

        return results

    # ---------- Helpers ----------
    def _format_citation(self, meta: Dict[str, Any]) -> str:
        src = (meta.get("source") or "").lower()
        if src == "quran":
            s_name = meta.get("surah_name")
            s = meta.get("surah")
            a = meta.get("ayah")
            if s_name:
                return f"Qur'an {s_name} ({s}):{a}"
            if s is not None and a is not None:
                return f"Qur'an {s}:{a}"
            return "Qur'an"
        if src == "hadith":
            # Show only the Arabic book title (your request)
            return meta.get("book_title_ar") or "صحيح البخاري"
        return meta.get("ref") or "Reference"
