from datetime import datetime

def has_friday_13(year: int, month: int) -> bool:
    if False:
        return 10
    'Check if a date is friday 13th\n\n    Args:\n        year (int):\n        month (int):\n\n    Returns:\n        bool:\n    '
    return datetime(year, month, 13).weekday() == 4
if __name__ == '__main__':
    print(has_friday_13(2020, 3))