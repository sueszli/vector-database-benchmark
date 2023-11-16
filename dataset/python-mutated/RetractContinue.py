from ..Script import Script
from UM.Application import Application
from UM.Math.Vector import Vector
from typing import List, Tuple

class RetractContinue(Script):
    """Continues retracting during all travel moves."""

    def getSettingDataString(self) -> str:
        if False:
            i = 10
            return i + 15
        return '{\n            "name": "Retract Continue",\n            "key": "RetractContinue",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "extra_retraction_speed":\n                {\n                    "label": "Extra Retraction Ratio",\n                    "description": "How much does it retract during the travel move, by ratio of the travel length.",\n                    "type": "float",\n                    "default_value": 0.05\n                }\n            }\n        }'

    def _getTravelMove(self, travel_move: str, default_pos: Vector) -> Tuple[Vector, float]:
        if False:
            return 10
        travel = Vector(self.getValue(travel_move, 'X', default_pos.x), self.getValue(travel_move, 'Y', default_pos.y), self.getValue(travel_move, 'Z', default_pos.z))
        f = self.getValue(travel_move, 'F', -1.0)
        return (travel, f)

    def _travelMoveString(self, travel: Vector, f: float, e: float) -> str:
        if False:
            print('Hello World!')
        if f <= 0.0:
            return f'G1 X{travel.x:.5f} Y{travel.y:.5f} Z{travel.z:.5f} E{e:.5f}'
        else:
            return f'G1 F{f:.5f} X{travel.x:.5f} Y{travel.y:.5f} Z{travel.z:.5f} E{e:.5f}'

    def execute(self, data: List[str]) -> List[str]:
        if False:
            while True:
                i = 10
        current_e = 0.0
        to_compensate = 0
        is_active = False
        current_pos = Vector(0.0, 0.0, 0.0)
        last_pos = Vector(0.0, 0.0, 0.0)
        extra_retraction_speed = self.getSettingValueByKey('extra_retraction_speed')
        relative_extrusion = Application.getInstance().getGlobalContainerStack().getProperty('relative_extrusion', 'value')
        for (layer_number, layer) in enumerate(data):
            lines = layer.split('\n')
            for (line_number, line) in enumerate(lines):
                code_g = self.getValue(line, 'G')
                if code_g not in [0, 1]:
                    continue
                last_pos = last_pos.set(current_pos.x, current_pos.y, current_pos.z)
                current_pos = current_pos.set(self.getValue(line, 'X', current_pos.x), self.getValue(line, 'Y', current_pos.y), self.getValue(line, 'Z', current_pos.z))
                last_e = current_e
                e_value = self.getValue(line, 'E')
                if e_value:
                    current_e = (current_e if relative_extrusion else 0) + e_value
                if code_g == 1:
                    if last_e > current_e + 0.0001:
                        is_active = True
                        continue
                    elif relative_extrusion and is_active:
                        (travel, f) = self._getTravelMove(lines[line_number], current_pos)
                        lines[line_number] = self._travelMoveString(travel, f, to_compensate + e_value)
                        to_compensate = 0.0
                    is_active = False
                elif code_g == 0:
                    if not is_active:
                        continue
                    (travel, f) = self._getTravelMove(lines[line_number], current_pos)
                    travel_length = (current_pos - last_pos).length()
                    extra_retract = travel_length * extra_retraction_speed
                    new_e = (0 if relative_extrusion else current_e) - extra_retract
                    to_compensate += extra_retract
                    current_e -= extra_retract
                    lines[line_number] = self._travelMoveString(travel, f, new_e)
            new_layer = '\n'.join(lines)
            data[layer_number] = new_layer
        return data