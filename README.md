# datamorphx

[![PyPI version](https://img.shields.io/pypi/v/datamorphx.svg)](https://pypi.org/project/datamorphx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

### Smart text augmentation for NLP and fine-tuning

**datamorphx** helps expand your text datasets with meaningful variations — think of it as a lightweight text blender for model training or prompt tuning.

---

### 📦 Installation

```bash
pip install datamorphx

from datamorphx import Augmentor

augmentor = Augmentor()
results = augmentor.expand(["I can't do this."])
print(results)

