from typing import Any, Dict
from math import log2


def entropy(class_counts: Dict[Any, int]):
    """
    Calculate entropy for a dictionary containing the number of elements.
    :param class_counts: A dictionary containing the number of occurances of each class.
    :return: The entropy value for a given class count.
    """
    elements_number = sum(class_counts.values())
    return -sum(
        class_counts[category]/elements_number*log2(class_counts[category]/elements_number)
        for category in class_counts
    )
