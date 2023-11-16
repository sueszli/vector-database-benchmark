from typing import NamedTuple

class Object(NamedTuple):
    position: tuple
    speed: tuple

def ask_position(obj: str) -> tuple:
    if False:
        i = 10
        return i + 15
    coorx = int(input(f'Coordenada x del objeto {obj}: '))
    coory = int(input(f'Coordenada y del objeto {obj}: '))
    return (coorx, coory)

def ask_speed(obj: str) -> tuple:
    if False:
        print('Hello World!')
    speedx = int(input(f'Velocidad en el eje x del objeto {obj}: '))
    speedy = int(input(f'Velocidad en el eje y del objeto {obj}: '))
    return (speedx, speedy)

def will_intersect(obj1: Object, obj2: Object) -> bool:
    if False:
        i = 10
        return i + 15
    return obj1.speed[1] / obj1.speed[0] != obj2.speed[1] / obj2.speed[0]

def same_position(obj1: Object, obj2: Object) -> bool:
    if False:
        while True:
            i = 10
    return obj1.position == obj2.position

def calculate_intersection(obj1: Object, obj2: Object) -> tuple:
    if False:
        print('Hello World!')
    m1 = obj1.speed[1] / obj1.speed[0]
    m2 = obj2.speed[1] / obj2.speed[0]
    '    \n    eq1 = f"y = {m1} * (x - {obj1.position[0]}) + {obj1.position[1]}"\n    eq2 = f"y = {m2} * (x - {obj2.position[0]}) + {obj2.position[1]}"\n    resol1 = f"{m1} * (x - {obj1.position[0]}) + {obj1.position[1]} = {m2} * (x - {obj2.position[0]}) + {obj2.position[1]}"\n    resol2 = f"{m1} * x - {m1} * {obj1.position[0]} + {obj1.position[1]} = {m2} * x - {m2} * {obj2.position[0]} + {obj2.position[1]}"\n    resol3 = f"{m1} * x - {m2} * x = - {m2} * {obj2.position[0]} + {obj2.position[1]} + {m1} * {obj1.position[0]} - {obj1.position[1]}"\n    resol4 = f"x * ({m1} - {m2}) = - {m2} * {obj2.position[0]} + {obj2.position[1]} + {m1} * {obj1.position[0]} - {obj1.position[1]}"\n    resol4 = f"x = (- {m2} * {obj2.position[0]} + {obj2.position[1]} + {m1} * {obj1.position[0]} - {obj1.position[1]}) / ({m1} - {m2})"\n    print("Estas rectas son las que describen el movimiento tanto para lo que hará como para lo que deberia hacer en tiempos negativos (avanzando en el tiempo tanto hacia delante como hacia detras)")\n    print(f"Ecuacion del objeto B: {eq1}\nEcuacion del objeto B: {eq2}\nProcedimiento:\nPaso 1: {resol1}\nPaso 2: {resol2}\nPaso 3: {resol3}\nPaso 4: {resol4}")\n    '
    coor_x = (-m2 * obj2.position[0] + obj2.position[1] + m1 * obj1.position[0] - obj1.position[1]) / (m1 - m2)
    coor_y = m1 * coor_x - obj1.position[0] + obj1.position[1]
    return (coor_x, coor_y)

def main() -> None:
    if False:
        while True:
            i = 10
    objectA = Object(position=ask_position('A'), speed=ask_speed('A'))
    objectB = Object(position=ask_position('B'), speed=ask_speed('B'))
    if will_intersect(objectA, objectB):
        intersection = calculate_intersection(objectA, objectB)
        if (intersection[0] - objectA.position[0]) % objectA.speed[0] == 0:
            t1 = (intersection[0] - objectA.position[0]) / objectA.speed[0]
            t2 = (intersection[0] - objectB.position[0]) / objectB.speed[0]
            if same_position(objectA, objectB):
                print(f'Aunque los objetos tengan distinta velocidad comienzan en el mismo punto de partida por lo que colisionarán en el punto {objectA.position} en el instante t = 0.')
            elif t1 == t2:
                print(f'Los objetos colisionarán en el punto {intersection} tras {t1} unidades de tiempo.')
            else:
                print(f'Aunque ambos objetos pasan por el mismo punto, necesitan distintos tiempos para alcanzar el punto {intersection}, El Objeto A necesita {t1} unidades de tiempo mientras que el Objeto B necesita {t2} unidades.')
        else:
            print('Los objetos A y B no colisionarán nunca dado que siguiendo sus proyecciones colisionaron en un momento anterior.')
    elif same_position(objectA, objectB):
        if objectA.speed == objectB.speed:
            print('Los objetos A y B tienen tanto la misma direccion como la misma velocidad por lo que colisionaran en todos los puntos por los que pasen.')
        else:
            print(f'Los objetos A y B parten desde el mismo punto y siguen la misma trayectoria, sin embargo, tienen velocidades distintas por lo que solo tendrán una colisión en {objectA.position} en el intsante t = 0.')
    else:
        print(f'Los objetos A y B nunca colisionarán al tener la misma pendiente en sus proyecciones y empezar en puntos distintos.')
if __name__ == '__main__':
    main()