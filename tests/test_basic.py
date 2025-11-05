from datamorphx import Augmentor  

def test_synonym_strategy_runs():
    aug = Augmentor(strategy="synonym")
    out = aug.expand(["I can't do this."])
    assert isinstance(out, list)
