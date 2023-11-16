from datetime import date

def ageCalculator(years, months, days):
    if False:
        i = 10
        return i + 15
    age_day = 0
    age_months = 0
    age_year = 0
    today_day = int(today.strftime('%d'))
    today_month = int(today.strftime('%m'))
    today_year = int(today.strftime('%y'))
    if today_day < day:
        today_day += 31
        age_day = today_day - days
    else:
        age_day = today_day - days
    if today_month < months:
        today_month += 12
        age_months = today_month - months
    else:
        age_months = today_month - months
    age_year = today_year - years
    print(f'your age of today is :{today_year}-{today_month}-{today_day}')
today = date.today()
print("today's date is:", today)
birthDate = input('Enter your birth date in YYYY-MM-DD format:')
(year, month, day) = map(int, birthDate.split('-'))
if month > 12 or day > 31 or year < int(today.strftime('%y')):
    print('invalid date')
    exit()
print('your date of birth is:', birthDate)
ageCalculator(year, month, day)