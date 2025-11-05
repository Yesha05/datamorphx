import random
import re
from typing import List
import spacy
from nltk.corpus import wordnet
from wordfreq import zipf_frequency

nlp = spacy.load("en_core_web_sm")
WORD_RE = re.compile(r"^[A-Za-z\-']+$")

# map spaCy POS to WordNet POS
POS_MAP = {
    "NOUN": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "ADJ": wordnet.ADJ,
    "ADV": wordnet.ADV,
}


def _get_wordnet_pos(spacy_pos: str):
    return POS_MAP.get(spacy_pos, None)


def synonym_augment_pos(text: str) -> str:
    doc = nlp(text)
    out_tokens: List[str] = []

    for token in doc:
        w = token.text
        # skip pronouns, punctuation, short tokens
        if token.is_stop or token.is_punct or len(w) <= 1 or token.pos_ == "PRON":
            out_tokens.append(w)
            continue

        wn_pos = _get_wordnet_pos(token.pos_)
        if not wn_pos:
            out_tokens.append(w)
            continue

        synsets = wordnet.synsets(w, pos=wn_pos)
        if not synsets:
            out_tokens.append(w)
            continue

        # collect candidate lemmas with filtering and scoring by frequency
        candidates = set()
        for s in synsets:
            for l in s.lemmas():
                cand = l.name().replace("_", " ")
                if " " in cand:  # avoid multi-word
                    continue
                if not WORD_RE.match(cand):
                    continue
                if cand.lower() == w.lower():
                    continue
                candidates.add(cand)

        # score by corpus frequency (zipf) and pick the most common candidate above threshold
        scored = []
        for c in candidates:
            freq = zipf_frequency(c, "en")  # higher is more common; ~1-7 range
            scored.append((freq, c))

        if not scored:
            out_tokens.append(w)
            continue

        # prefer high-frequency synonyms
        scored.sort(reverse=True)
        best_freq, best_cand = scored[0]
        # require some minimal commonness to avoid obscure senses
        if best_freq >= 2.5 and random.random() < 0.4:
            out_tokens.append(best_cand)
        else:
            out_tokens.append(w)

    return "".join([t.whitespace_ + t.text if hasattr(t, "whitespace_") else t for t in out_tokens])
