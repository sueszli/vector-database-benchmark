from datetime import datetime
from typing import Any, Callable, List, TypeVar, Union
from faker import Faker
f = Faker()
Faker.seed(0)
T = TypeVar('T')
_END_DATETIME = datetime(2023, 1, 1, 0, 0, 0, 0)

def one_of(*callables: Callable[[], Any]) -> Any:
    if False:
        return 10
    return f.random.choice(callables)()

def list_of(callable: Callable[[], T], max_length: int) -> List[T]:
    if False:
        print('Hello World!')
    return [callable() for _ in range(f.random_int(max=max_length))]

def lax_int(*args: Any, **kwargs: Any) -> Union[int, float, str]:
    if False:
        return 10
    return f.random.choice((int, float, str))(f.random_int(*args, **kwargs))

def lax_float(*args: Any, **kwargs: Any) -> Union[int, float, str]:
    if False:
        i = 10
        return i + 15
    return f.random.choice((int, float, str))(f.pyfloat(*args, **kwargs))

def time_seconds() -> int:
    if False:
        i = 10
        return i + 15
    dt = f.date_time(end_datetime=_END_DATETIME)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt - midnight).total_seconds()

def time_microseconds() -> float:
    if False:
        while True:
            i = 10
    return float(time_seconds()) + f.random_int(max=999999) * 1e-06

def time_string() -> str:
    if False:
        i = 10
        return i + 15
    return f.time()

def lax_time() -> Union[int, float, str]:
    if False:
        return 10
    return one_of(time_seconds, time_microseconds, time_string)

def date_string() -> str:
    if False:
        print('Hello World!')
    return f.date(end_datetime=_END_DATETIME).format('%Y-%m-%d')

def datetime_timestamp() -> int:
    if False:
        while True:
            i = 10
    dt = f.date_time(end_datetime=_END_DATETIME)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt - midnight).total_seconds()

def datetime_microseconds() -> float:
    if False:
        print('Hello World!')
    return float(datetime_timestamp()) + f.random_int(max=999999) * 1e-06

def datetime_str() -> str:
    if False:
        return 10
    return f.date_time(end_datetime=_END_DATETIME).isoformat()

def lax_datetime() -> Union[int, float, str]:
    if False:
        for i in range(10):
            print('nop')
    return one_of(datetime_timestamp, datetime_microseconds, datetime_str)

def blog() -> dict:
    if False:
        return 10
    return {'type': 'blog', 'title': f.text(max_nb_chars=40), 'post_count': lax_int(), 'readers': lax_int(), 'avg_post_rating': lax_float(min_value=0, max_value=5), 'url': f.url()}

def social_profile() -> dict:
    if False:
        print('Hello World!')
    return {'type': 'profile', 'username': f.user_name(), 'join_date': date_string(), **one_of(facebook_profile, twitter_profile, linkedin_profile)}

def facebook_profile() -> dict:
    if False:
        i = 10
        return i + 15
    return {'network': 'facebook', 'friends': lax_int()}

def twitter_profile() -> dict:
    if False:
        i = 10
        return i + 15
    return {'network': 'twitter', 'followers': lax_int()}

def linkedin_profile() -> dict:
    if False:
        i = 10
        return i + 15
    return {'network': 'linkedin', 'connections': min(f.random_int(), 500)}

def website() -> dict:
    if False:
        while True:
            i = 10
    return one_of(blog, social_profile)

def person() -> dict:
    if False:
        i = 10
        return i + 15
    return {'id': f.uuid4(), 'name': f.name(), 'height': str(f.pydecimal(min_value=1, max_value=2, right_digits=2)), 'entry_created_date': date_string(), 'entry_created_time': lax_time(), 'entry_updated_at': lax_datetime(), 'websites': list_of(website, max_length=5)}

def person_data(length: int) -> List[dict]:
    if False:
        return 10
    return [person() for _ in range(length)]