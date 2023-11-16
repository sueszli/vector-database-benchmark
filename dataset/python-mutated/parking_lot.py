from abc import ABCMeta, abstractmethod
from enum import Enum

class VehicleSize(Enum):
    MOTORCYCLE = 0
    COMPACT = 1
    LARGE = 2

class Vehicle(metaclass=ABCMeta):

    def __init__(self, vehicle_size, license_plate, spot_size):
        if False:
            while True:
                i = 10
        self.vehicle_size = vehicle_size
        self.license_plate = license_plate
        self.spot_size
        self.spots_taken = []

    def clear_spots(self):
        if False:
            return 10
        for spot in self.spots_taken:
            spot.remove_vehicle(self)
        self.spots_taken = []

    def take_spot(self, spot):
        if False:
            return 10
        self.spots_taken.append(spot)

    @abstractmethod
    def can_fit_in_spot(self, spot):
        if False:
            return 10
        pass

class Motorcycle(Vehicle):

    def __init__(self, license_plate):
        if False:
            i = 10
            return i + 15
        super(Motorcycle, self).__init__(VehicleSize.MOTORCYCLE, license_plate, spot_size=1)

    def can_fit_in_spot(self, spot):
        if False:
            print('Hello World!')
        return True

class Car(Vehicle):

    def __init__(self, license_plate):
        if False:
            for i in range(10):
                print('nop')
        super(Car, self).__init__(VehicleSize.COMPACT, license_plate, spot_size=1)

    def can_fit_in_spot(self, spot):
        if False:
            print('Hello World!')
        return spot.size in (VehicleSize.LARGE, VehicleSize.COMPACT)

class Bus(Vehicle):

    def __init__(self, license_plate):
        if False:
            i = 10
            return i + 15
        super(Bus, self).__init__(VehicleSize.LARGE, license_plate, spot_size=5)

    def can_fit_in_spot(self, spot):
        if False:
            i = 10
            return i + 15
        return spot.size == VehicleSize.LARGE

class ParkingLot(object):

    def __init__(self, num_levels):
        if False:
            for i in range(10):
                print('nop')
        self.num_levels = num_levels
        self.levels = []

    def park_vehicle(self, vehicle):
        if False:
            for i in range(10):
                print('nop')
        for level in self.levels:
            if level.park_vehicle(vehicle):
                return True
        return False

class Level(object):
    SPOTS_PER_ROW = 10

    def __init__(self, floor, total_spots):
        if False:
            print('Hello World!')
        self.floor = floor
        self.num_spots = total_spots
        self.available_spots = 0
        self.spots = []

    def spot_freed(self):
        if False:
            return 10
        self.available_spots += 1

    def park_vehicle(self, vehicle):
        if False:
            while True:
                i = 10
        spot = self._find_available_spot(vehicle)
        if spot is None:
            return None
        else:
            spot.park_vehicle(vehicle)
            return spot

    def _find_available_spot(self, vehicle):
        if False:
            print('Hello World!')
        'Find an available spot where vehicle can fit, or return None'
        pass

    def _park_starting_at_spot(self, spot, vehicle):
        if False:
            print('Hello World!')
        'Occupy starting at spot.spot_number to vehicle.spot_size.'
        pass

class ParkingSpot(object):

    def __init__(self, level, row, spot_number, spot_size, vehicle_size):
        if False:
            return 10
        self.level = level
        self.row = row
        self.spot_number = spot_number
        self.spot_size = spot_size
        self.vehicle_size = vehicle_size
        self.vehicle = None

    def is_available(self):
        if False:
            i = 10
            return i + 15
        return True if self.vehicle is None else False

    def can_fit_vehicle(self, vehicle):
        if False:
            while True:
                i = 10
        if self.vehicle is not None:
            return False
        return vehicle.can_fit_in_spot(self)

    def park_vehicle(self, vehicle):
        if False:
            i = 10
            return i + 15
        pass

    def remove_vehicle(self):
        if False:
            return 10
        pass