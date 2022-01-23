from src.io import load_json, write_json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import click


def sum_entropy_in_ranges(step: float, entropies: Dict[int, float]):
    """
    For every range sums the number of entropies that are in this range.
    It helps to estimate the distribution of entropies
    :param entropies: a collection of entropies for every label
    :param step: a range for which entropies will be summed
    :return: summed entropies for every step
    """

    sums = defaultdict(lambda: 0)
    for entropy in entropies.values():
        bucket_no = int(entropy / step)
        sums[bucket_no] += 1

    return sums


def calculate_entropy_distribution(entropies_cnt: Dict[int, int]):
    """
    Receving entropies_cnt calculates what percent of entropies
    were in a given range. Finds max_bucket id and fills every bucket
    from 0 to max_bucket_id with % of entropies that are in given bucket
    :param entropies_cnt: sums of entropies in given range
    :return:
    """

    max_bucket = max(entropies_cnt) + 1
    total_cnt = sum(entropies_cnt.values())
    dist = [0.0] * max_bucket
    for key, value in entropies_cnt.items():
        dist[key] = value / total_cnt

    return dist


def print_dist(data: List[float]):
    for idx, item in enumerate(data):
        formatted_item = "{:.2f}".format(item * 100)
        str_val = str(formatted_item)
        str_val = str_val.replace(".", ",")
        print(f'{str_val}%')


@click.command(
    help="Prepares entropy raport"
)
@click.option(
    "-i",
    "--input_file",
    type=Path,
    required=True,
    help="Path to file containing entropy for every cluster"
)
@click.option(
    "-s",
    "--step",
    type=float,
    default=0.5,
    required=True,
    help="Path to file containing entropy for every cluster"
)
def main(input_file: Path, step: float):
    entropies = load_json(input_file)
    sums = sum_entropy_in_ranges(step, entropies)
    dist = calculate_entropy_distribution(sums)
    print_dist(dist)


if __name__ == '__main__':
    main()
