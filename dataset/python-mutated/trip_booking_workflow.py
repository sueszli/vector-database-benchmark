from typing import List, Tuple, Optional
import ray
from ray import workflow

def make_request(*args) -> None:
    if False:
        for i in range(10):
            print('nop')
    return '-'.join(args)

@ray.remote
def generate_request_id():
    if False:
        for i in range(10):
            print('nop')
    import uuid
    return uuid.uuid4().hex

@ray.remote
def book_car(request_id: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    car_reservation_id = make_request('book_car', request_id)
    return car_reservation_id

@ray.remote
def book_hotel(request_id: str, *deps) -> str:
    if False:
        for i in range(10):
            print('nop')
    hotel_reservation_id = make_request('book_hotel', request_id)
    return hotel_reservation_id

@ray.remote
def book_flight(request_id: str, *deps) -> str:
    if False:
        i = 10
        return i + 15
    flight_reservation_id = make_request('book_flight', request_id)
    return flight_reservation_id

@ray.remote
def book_all(car_req_id: str, hotel_req_id: str, flight_req_id: str) -> str:
    if False:
        while True:
            i = 10
    car_res_id = book_car.bind(car_req_id)
    hotel_res_id = book_hotel.bind(hotel_req_id, car_res_id)
    flight_res_id = book_flight.bind(hotel_req_id, hotel_res_id)

    @ray.remote
    def concat(*ids: List[str]) -> str:
        if False:
            return 10
        return ', '.join(ids)
    return workflow.continuation(concat.bind(car_res_id, hotel_res_id, flight_res_id))

@ray.remote
def handle_errors(car_req_id: str, hotel_req_id: str, flight_req_id: str, final_result: Tuple[Optional[str], Optional[Exception]]) -> str:
    if False:
        while True:
            i = 10
    (result, error) = final_result

    @ray.remote
    def wait_all(*deps) -> None:
        if False:
            print('Hello World!')
        pass

    @ray.remote
    def cancel(request_id: str) -> None:
        if False:
            print('Hello World!')
        make_request('cancel', request_id)
    if error:
        return workflow.continuation(wait_all.bind(cancel.bind(car_req_id), cancel.bind(hotel_req_id), cancel.bind(flight_req_id)))
    else:
        return result
if __name__ == '__main__':
    car_req_id = generate_request_id.bind()
    hotel_req_id = generate_request_id.bind()
    flight_req_id = generate_request_id.bind()
    saga_result = book_all.options(**workflow.options(catch_exceptions=True)).bind(car_req_id, hotel_req_id, flight_req_id)
    final_result = handle_errors.bind(car_req_id, hotel_req_id, flight_req_id, saga_result)
    print(workflow.run(final_result))