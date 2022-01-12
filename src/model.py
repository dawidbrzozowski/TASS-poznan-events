from pathlib import Path
from typing import List

import click
import torch
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer

from src.io import load_json, write_json

_MODEL_USED = "allegro/herbert-base-cased"


class BERTEncoder:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_USED)
        self._model = BertModel.from_pretrained(_MODEL_USED, output_hidden_states=True)
        self._model.eval()
        torch.no_grad()

    def generate_sentence_embedding(self, texts: List[str]) -> List[List[float]]:
        embeds = []
        for text in tqdm(texts, desc="Generating embeddings for given texts..."):
            encoding = self._tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            preds = self._model(**encoding)
            last_layer = preds["hidden_states"][-1]
            # last_four_layers = preds["hidden_states"][-4:]
            # cat_hidden_states = torch.cat(last_four_layers, dim=-1)
            # sentence_emb = torch.mean(cat_hidden_states, dim=1).squeeze()
            sentence_emb = torch.mean(last_layer, dim=1).squeeze()
            embeds.append(sentence_emb.tolist())
        return embeds


@click.command(
    help="Generate embeddings for a directory."
    " The embeddings will be based on the description"
    " and the name of the event concatenated."
)
@click.option(
    "--input_dir",
    "-i",
    type=Path,
    required=True,
    help="Path to the input directory, from which JSON files will be read."
)
def main(input_dir: Path):
    enc = BERTEncoder()
    for file_path in tqdm(
            [
                path
                for path in list(input_dir.iterdir())
                if not path.stem.endswith("embeddings")
            ]
    ):
        day_data = load_json(file_path)
        names_and_descriptions = [
            f"{event['name']} {event['description']}"
            for event in day_data
        ]
        embeds = enc.generate_sentence_embedding(names_and_descriptions)
        write_json(
            data=embeds,
            out_path=file_path.parent / f"{file_path.stem}-embeddings.json"
        )


if __name__ == '__main__':
    main()
