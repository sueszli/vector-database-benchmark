import datetime

def is_friday_thirteen(year: int, month: int) -> bool:
    if False:
        while True:
            i = 10
    try:
        if not 1 <= year <= 9999:
            raise ValueError('El año debe estar en el rango entre 1 y 9999')
        elif not 1 <= month <= 12:
            raise ValueError('El mes debe estar entre el 1 y 12')
        date_13th = datetime.date(year, month, 13)
        return date_13th.weekday() == 4
    except ValueError as ve:
        return (False, str(ve))
        return False
print(is_friday_thirteen(2023, 10))
print(is_friday_thirteen(2023, 1))
print(is_friday_thirteen(-5, 12))
print(is_friday_thirteen(2022, 8))
print(is_friday_thirteen(2022, 2))
print(is_friday_thirteen(2023, 14))