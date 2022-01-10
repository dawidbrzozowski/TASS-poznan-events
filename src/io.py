
import json
from pathlib import Path
from typing import Optional, Union

Pathlike = Union[str, Path]
JSONlike = Optional[Union[str, int, float, bool, list, dict]]
_DEFAULT_INDENT = 4


def write_json(data: JSONlike, out_path: Pathlike, **kwargs):
    """
    Writes a JSONlike object to a file.
    :param out_path: path for the output file.
    :param data: data to be stored.
    :param kwargs: additional keyword arguments for json.dump function.
    """
    if "indent" not in kwargs:
        kwargs["indent"] = _DEFAULT_INDENT

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with out_path.open("w+", encoding="utf-8") as write_file:
        json.dump(data, write_file, **kwargs)


def load_json(in_path: Pathlike) -> JSONlike:
    """
    Reads a JSONlike object from a file.
    :param in_path: path for the input file.
    :return: JSONlike object.
    """
    with Path(in_path).open(encoding="utf-8") as f:
        return json.load(f)
