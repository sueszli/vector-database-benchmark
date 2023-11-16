"""
 * Crea una función que sea capaz de detectar si existe un viernes 13 en el mes y el año indicados.
 * - La función recibirá el mes y el año y retornará verdadero o falso.
"""
import calendar

def friday(month: int, year: int):
    if False:
        print('Hello World!')
    if calendar.weekday(year, month, 13) == 4:
        return True
    else:
        return False
print(friday(5, 2022))
print(friday(1, 2500))