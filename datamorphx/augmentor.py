from typing import List, Optional
import random

try:
    from nltk.corpus import wordnet
except ImportError:
    wordnet = None


class Augmentor:
    """
    datamorphx — Smart text augmentation for NLP and LLM fine-tuning.

    Supported strategies:
      - "semantic_variation": basic heuristic rewrites
      - "synonym": uses WordNet (if installed)
      - "embedding_paraphrase": placeholder for semantic similarity rewrites
    """

    def __init__(
        self,
        model: Optional[str] = None,
        similarity_cutoff: float = 0.8,
        strategy: str = "semantic_variation",
    ):
        self.model = model
        self.similarity_cutoff = similarity_cutoff
        self.strategy = strategy

    def expand(self, texts: List[str]) -> List[str]:
        if self.strategy == "synonym":
            return [self._synonym_replace(t) for t in texts]
        elif self.strategy == "embedding_paraphrase":
            return [self._embedding_paraphrase(t) for t in texts]
        else:
            return [self._semantic_variation(t) for t in texts]

    def _semantic_variation(self, text: str) -> str:
        if "can't" in text:
            return text.replace("can't", "cannot")
        elif "hate" in text:
            return text.replace("hate", "strongly dislike")
        return text + " (augmented)"
    
    def _synonym_replace(self, text: str) -> str:
        import spacy
        if not wordnet:
            return text + " [WordNet not available]"

    # Load spaCy model once
        nlp = spacy.load("en_core_web_sm")
        print("✅ Loaded model:", nlp.meta["name"])
        words = text.split()
        out = []
        for w in words:
            syns = wordnet.synsets(w)
            if syns:
                lemmas = [l.name().replace("_", " ") for s in syns for l in s.lemmas()]
                lemmas = [l for l in lemmas if l.lower() != w.lower()]
            # Check semantic similarity with spaCy
                w_doc = nlp(w)
                filtered = []
                for l in lemmas:
                    l_doc = nlp(l)
                    if w_doc.similarity(l_doc) > 0.5:
                        filtered.append(l)
                if filtered:
                    out.append(random.choice(filtered))
                    continue
            out.append(w)
        return " ".join(out)



    def _embedding_paraphrase(self, text: str) -> str:
        # Placeholder for future embedding-based augmentation.
        return text + " [embedding paraphrase placeholder]"
