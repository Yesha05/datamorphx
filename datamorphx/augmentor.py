from typing import List, Optional

class Augmentor:
    """
    Minimal Augmentor stub.
    Real implementation will call LLMs / embeddings and filter by similarity.
    """
    def __init__(self, model: Optional[str] = None, similarity_cutoff: float = 0.8):
        self.model = model
        self.similarity_cutoff = similarity_cutoff

    def expand(self, texts: List[str], labels: Optional[List[int]] = None, strategy: str = "semantic_variation") -> List[str]:
        out = []
        for t in texts:
            if "can't" in t:
                out.append(t.replace("can't", "cannot"))
            elif "hate" in t:
                out.append(t.replace("hate", "strongly dislike"))
            else:
                out.append(t + " (augmented)")
        return out
