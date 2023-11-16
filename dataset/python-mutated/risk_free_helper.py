YIELD_CURVE_TENORS = {0: '0S', 30: '1M', 60: '2M', 90: '3M', 180: '6M', 270: '9M', 365: '1Y', 365 * 2: '2Y', 365 * 3: '3Y', 365 * 4: '4Y', 365 * 5: '5Y', 365 * 6: '6Y', 365 * 7: '7Y', 365 * 8: '8Y', 365 * 9: '9Y', 365 * 10: '10Y', 365 * 15: '15Y', 365 * 20: '20Y', 365 * 30: '30Y', 365 * 40: '40Y', 365 * 50: '50Y'}
YIELD_CURVE_DURATION = sorted(YIELD_CURVE_TENORS.keys())

def get_tenor_for(start_date, end_date):
    if False:
        print('Hello World!')
    duration = (end_date - start_date).days
    tenor = 0
    for t in YIELD_CURVE_DURATION:
        if duration >= t:
            tenor = t
        else:
            break
    return YIELD_CURVE_TENORS[tenor]

def get_tenors_for(start_date, end_date):
    if False:
        for i in range(10):
            print('nop')
    return [YIELD_CURVE_TENORS[t] for t in YIELD_CURVE_DURATION if (end_date - start_date).days >= t]