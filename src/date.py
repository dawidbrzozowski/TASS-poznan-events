import datetime


def get_number_of_days_in_month(year: int, month: int) -> int:
    """
    Returns the number of days in a given month
    :param year: int value representing a given year.
    :param month: int value from 1-12 indicating the month index.
    :return: the number of days in a given month.
    """
    assert 1 <= month <= 12, "Month must be between 1 and 12"
    d0 = datetime.datetime(year=year, month=month, day=1)
    d1 = (
        datetime.datetime(year=year, month=month + 1, day=1)
        if month != 12
        else datetime.datetime(year=year + 1, month=1, day=1)
    )
    return (d1 - d0).days
