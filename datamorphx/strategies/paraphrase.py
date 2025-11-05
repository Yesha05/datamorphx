from sentence_transformers import SentenceTransformer, util

_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def paraphrase_augment(text):
    # simple heuristic: find semantically similar variants
    candidates = [text, text.replace("can't", "cannot"), text.replace("do", "perform")]
    embeddings = _model.encode(candidates, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(embeddings, embeddings)[0]
    best_idx = sims[1:].argmax().item() + 1
    return candidates[best_idx]
