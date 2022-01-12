from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.io import load_json
from tqdm import tqdm

POSSIBLE_CATEGORIES = [
    "Wydarzenia międzynarodowe",
    "Inne",
    "Kultura i sztuka",
    "Konferencje, spotkania i wykłady",
    "Targi",
    "Rozrywka",
    "Biznes",
    "Seniorzy",
    "Wydarzenia online/zdalne",
    "Czerwiec 56 - Czarny Czwartek",
    "Sport",
    "Wystawy",
    "Imprezy w CK Zamek",
    "Koncerty",
    "dostępne dla osób z niepełnosprawnościami",
    "Zdrowie",
    "Oświata",
    "Dziecko",
]

_DATA_DIRS = [Path("data/2021/3"), Path("data/2021/6"), Path("data/2021/9"), Path("data/2021/12")]


@dataclass(frozen=True)
class Event:
    event_id: int
    name: str
    description: str
    category: str
    embedding: np.ndarray


def load_events_from_dir(dir_path: Path) -> List[Event]:
    events: List[Event] = []
    for i in range(1, find_max_day_in_dir(dir_path) + 1):
        event_data = load_json(dir_path / f"{i}.json")
        embedding_data = load_json(dir_path / f"{i}-embeddings.json")
        for event, event_embedding in zip(event_data, embedding_data):
            events.append(
                Event(
                    event_id=event["event_id"],
                    name=event["name"],
                    description=event["description"],
                    category=event["category"],
                    embedding=np.array(event_embedding)
                )
            )
    return events


def load_all_events() -> List[Event]:
    events: List[Event] = []
    for data_dir in tqdm(_DATA_DIRS, desc="Loading data from dirs..."):
        events.extend(load_events_from_dir(data_dir))
    return events


def find_max_day_in_dir(dir_path: Path) -> int:
    days = [int(fp.stem) for fp in dir_path.iterdir() if not fp.stem.endswith("embeddings")]
    return max(days)
