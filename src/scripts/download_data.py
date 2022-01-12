from pathlib import Path
from typing import Optional

import click
import requests
from tqdm import tqdm
from click import INT
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

from src.date import get_number_of_days_in_month
from src.io import write_json

_BASE_URL = "https://www.poznan.pl/mim/public/ws-information/?co=getDayEvent&date="
_HTTP_OK = 200
_TAG_PREFIX = "./{http://sit-wlkp.eu/xmlrepo/events.xsd}"


def _prepare_url(year: int, month: int, day: int) -> str:
    date = f"{year}-{month}-{day}"
    return _BASE_URL + date


@click.command(help="Download events data from a given month using Poznan API")
@click.option(
    "-m",
    "--month",
    type=INT,
    required=True,
    help="Month as integer value between 1-12.",
)
@click.option(
    "-y",
    "--year",
    type=INT,
    required=True,
    help="Year as integer.",
)
@click.option(
    "-o",
    "--output-dir",
    type=Path,
    required=False,
    help="Output dir to store events.",
)
def main(month: INT, year: INT, output_dir: Optional[Path]):
    for day in tqdm(range(1, get_number_of_days_in_month(year, month) + 1)):
        url = _prepare_url(year, month, day)
        xml_data = requests.get(url)
        if xml_data.status_code != _HTTP_OK:
            raise requests.HTTPError(
                f"Could not retrieve data for {year}/{month}/{day}"
            )
        tree_root = ET.ElementTree(ET.fromstring(xml_data.content.decode())).getroot()
        events = tree_root.findall(f"{_TAG_PREFIX}event")
        events_as_dicts = [
            {
                "event_id": event.find(f"{_TAG_PREFIX}event_id").text,
                "name": BeautifulSoup(
                    event.find(f"{_TAG_PREFIX}event_version")
                    .find(f"{_TAG_PREFIX}version")
                    .find(f"{_TAG_PREFIX}evtml_name")
                    .text,
                    "html.parser",
                ).get_text(strip=True, separator=" "),
                "description": BeautifulSoup(
                    event.find(f"{_TAG_PREFIX}event_version")
                    .find(f"{_TAG_PREFIX}version")
                    .find(f"{_TAG_PREFIX}evtml_desc")
                    .text,
                    "html.parser",
                ).get_text(strip=True, separator=" "),
                "category": event.find(f"{_TAG_PREFIX}category").text,
            }
            for event in events
        ]

        path = (
            Path(output_dir) / f"{day}.json"
            if output_dir
            else Path(f"data/{year}/{month}/{day}.json")
        )
        write_json(events_as_dicts, out_path=path)


if __name__ == "__main__":
    main()
