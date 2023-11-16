import re
import repath

class TemplateRoute:

    def __init__(self, route: str) -> None:
        if False:
            return 10
        self.__last_params = {}
        self.route = route

    def match(self, route_template: str) -> bool:
        if False:
            i = 10
            return i + 15
        for k in self.__last_params:
            setattr(self, k, None)
        pattern = repath.pattern(route_template)
        match = re.match(pattern, self.route)
        if match:
            self.__last_params = match.groupdict()
            for (k, v) in self.__last_params.items():
                setattr(self, k, v)
            return True
        return False