# datamorphx/cli.py
import argparse
from datamorphx import Augmentor

def main():
    parser = argparse.ArgumentParser(description="datamorphx text augmentation CLI")
    parser.add_argument("--strategy", type=str, default="semantic_variation", help="augmentation strategy")
    parser.add_argument("--text", type=str, required=True, help="input text to augment")
    args = parser.parse_args()

    aug = Augmentor(strategy=args.strategy)
    result = aug.expand([args.text])
    print(result[0])

if __name__ == "__main__":
    main()
