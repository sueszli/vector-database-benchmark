"""
/* Reto #43: Simulador de clima
 *  Dificultad: Fácil | Publicación: 30/10/23 | Corrección: 13/11/23
 *  Enunciado: 
 
 * Crea una función que simule las condiciones climáticas (temperatura y probabilidad de lluvia)
 * de un lugar ficticio al pasar un número concreto de días según estas reglas:
 * - La temperatura inicial y el % de probabilidad de lluvia lo define el usuario.
 * - Cada día que pasa:
 *   - 10% de posibilidades de que la temperatura aumente o disminuya 2 grados.
 *   - Si la temperatura supera los 25 grados, la probabilidad de lluvia al día 
 *     siguiente aumenta en un 20%.
 *   - Si la temperatura baja de 5 grados, la probabilidad de lluvia al día 
 *     siguiente disminuya en un 20%.
 *   - Si llueve (100%), la temperatura del día siguiente disminuye en 1 grado.
 * - La función recibe el número de días de la predicción y muestra la temperatura
 *   y si llueve durante todos esos días.
 * - También mostrará la temperatura máxima y mínima de ese periodo y cuántos días va a llover.
 
 */
 """
import random

class WheatherSimulator:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.days_predicted = self.get_days()
        self.intial_temperature = self.get_temperature()
        self.initial_rain_probability = self.get_rain_probability()
        self.days_rain = 0
        self.temperatures = []
        print('\nEmpezemos con la predicción')
        self.predict_wheather(1, self.intial_temperature, self.initial_rain_probability)

    def get_days(self):
        if False:
            for i in range(10):
                print('nop')
        days = input('\nCuántos días quieres que dure la predicción? (introduce un número entero mayor que 0): ')
        try:
            int(days)
            return int(days)
        except Exception:
            print('Parámetro incorrecto. Por favor, introduce una cantidad válida')
            return self.get_days()

    def get_temperature(self):
        if False:
            for i in range(10):
                print('nop')
        temp = input('Cuál es la temperatura inicial? (introduce un número entero mayor que 0): ')
        try:
            int(temp)
            return int(temp)
        except Exception:
            print('Parámetro incorrecto. Por favor, introduce un parámetro válido')
            return self.get_temperature()

    def get_rain_probability(self):
        if False:
            for i in range(10):
                print('nop')
        rain_prob = input('Cuál es la probabilidad inicial de lluvia? (introduce un número decimal entre cero y uno, siendo 1 máxima posibilidad y 0 imposible): ')
        try:
            float(rain_prob)
        except Exception:
            print('Parámetro incorrecto. Por favor, introduce un parámetro válido')
            return self.get_rain_probability()
        if float(rain_prob) <= 1:
            return float(rain_prob)
        else:
            print('Parámetro incorrecto. Por favor, introduce un parámetro válido')
            return self.get_rain_probability()

    def predict_wheather(self, day_number: int, temp: int, rain_prob: int):
        if False:
            for i in range(10):
                print('nop')
        if day_number <= self.days_predicted:
            change_of_temperature = [False, False]
            next_day_data = [temp, rain_prob]
            if random.randint(1, 10) == 1:
                (change_of_temperature[0], change_of_temperature[1]) = (True, random.randint(0, 1))
                if change_of_temperature[1] == 0:
                    temp -= 2
                else:
                    temp += 2
            self.temperatures.append(temp)
            print(f'\nDía {day_number}:\n')
            print(f'Temperatura: {temp}')
            if random.randint(0, 9) in [number for number in range(int(rain_prob * 10))]:
                self.days_rain += 1
                print('Hoy lloverá.')
                next_day_data[0] += 1
            else:
                print('Hoy no lloverá')
            if temp > 25:
                next_day_data[1] += 0.2
            if temp < 5:
                next_day_data[1] -= 0.2
            self.predict_wheather(day_number + 1, next_day_data[0], next_day_data[1])
        else:
            print(f'\nLa temperatura mínima durante la predicción será {min(self.temperatures)} grados y la máxima, de {max(self.temperatures)} grados.')
            print(f'En total, lloverá {self.days_rain} días.')
if __name__ == '__main__':
    print('\nHola, soy un simulador del clima. Puedo predecir el clima en Termolandia, un país muy lejano. \nSolo dime las condiciones iniciales y el número de días que quieres predecir.')
    while True:
        WheatherSimulator()
        input('\nPresiona Enter para empezar una nueva predicción.\n')