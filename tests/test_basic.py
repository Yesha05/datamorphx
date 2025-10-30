from datamorph import Augmentor

def test_expand():
    aug = Augmentor()
    out = aug.expand(["I can't do this."])
    assert "cannot" in out[0] or "(augmented)" in out[0]
