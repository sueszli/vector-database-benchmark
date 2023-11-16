"""Amazon Neptune GremlinParser Module (PRIVATE)."""
from typing import Any, Dict, List
import awswrangler.neptune._gremlin_init as gremlin

class GremlinParser:
    """Class representing a parser for returning Gremlin results as a dictionary."""

    @staticmethod
    def gremlin_results_to_dict(result: Any) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Take a Gremlin ResultSet and return a dictionary.\n\n        Parameters\n        ----------\n        result : Any\n            The Gremlin result set to convert\n\n        Returns\n        -------\n        List[Dict[str, Any]]\n            A list of dictionary results\n        '
        res = []
        if isinstance(result, (list, gremlin.Path)):
            for x in result:
                res.append(GremlinParser._parse_dict(x))
        elif isinstance(result, dict):
            res.append(result)
        else:
            res.append(GremlinParser._parse_dict(result))
        return res

    @staticmethod
    def _parse_dict(data: Any) -> Any:
        if False:
            i = 10
            return i + 15
        d: Dict[str, Any] = {}
        if isinstance(data, (list, gremlin.Path)):
            res = []
            for x in data:
                res.append(GremlinParser._parse_dict(x))
            return res
        if isinstance(data, (gremlin.Vertex, gremlin.Edge, gremlin.VertexProperty, gremlin.Property)):
            data = data.__dict__
        elif not hasattr(data, '__len__') or isinstance(data, str):
            data = {0: data}
        for (k, v) in data.items():
            if isinstance(k, (gremlin.Vertex, gremlin.Edge)):
                k = k.id
            if isinstance(v, list) and len(v) == 1:
                d[k] = v[0]
            else:
                d[k] = v
            if isinstance(data, (gremlin.Vertex, gremlin.Edge, gremlin.VertexProperty, gremlin.Property)):
                d[k] = d[k].__dict__
        return d