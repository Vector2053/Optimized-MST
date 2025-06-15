import json
from typing import Union, Optional, Tuple

import numpy as np

from obsolete.lct_mode import lct


def _serialize(obj: Union[float, str, Tuple, np.int64, np.float64]) -> str:
    if isinstance(obj, (np.integer, np.floating)):
        obj = float(obj)
    if isinstance(obj, (int, float)):
        return json.dumps({"type": "number", "value": obj})
    elif isinstance(obj, str):
        return json.dumps({"type": "string", "value": obj})
    elif isinstance(obj, tuple):
        elem_types = {type(e).__name__ for e in obj}
        if len(elem_types) != 1:
            raise ValueError("Tuple elements must be of the same type")
        elem_type = elem_types.pop()
        if elem_type not in ('str', 'int', 'float', 'float64', 'int64'):
            raise TypeError("Unsupported tuple element type")
        values = [o for o in obj] if elem_type == 'str' else [float(o) for o in obj]
        return json.dumps({
            "type": "tuple",
            "elem_type": 'str' if elem_type == 'str' else 'number',
            "value": values
        })
    else:
        raise TypeError("Unsupported type")


def _deserialize(s: str) -> Union[float, str, Tuple]:
    data = json.loads(s)
    if data['type'] == 'number':
        return float(data['value'])
    elif data['type'] == 'string':
        return data['value']
    elif data['type'] == 'tuple':
        elem_type = data['elem_type']
        values = data['value']
        if elem_type == 'str':
            return tuple(values)
        else:
            return tuple(map(float, values))
    else:
        raise ValueError("Unknown type")


class PyLinkCutTree:
    def __init__(self):
        self._lct = lct.LinkCutTree_Mode()

    def link(self, u: Union[float, str, Tuple], v: Union[float, str, Tuple], weight: float):
        u_str = _serialize(u)
        v_str = _serialize(v)
        self._lct.link(u_str, v_str, weight)

    def cut(self, u: Union[float, str, Tuple], v: Union[float, str, Tuple]):
        u_str = _serialize(u)
        v_str = _serialize(v)
        self._lct.cut(u_str, v_str)

    def is_connection(self, u: Union[float, str, Tuple], v: Union[float, str, Tuple]) -> bool:
        u_str = _serialize(u)
        v_str = _serialize(v)
        return self._lct.is_connection(u_str, v_str)

    def get_max_edge(self, u: Union[float, str, Tuple], v: Union[float, str, Tuple]
                     ) -> Tuple[Tuple[Optional[Union[float, str, Tuple]], Optional[Union[float, str, Tuple]]], float]:
        u_str = _serialize(u)
        v_str = _serialize(v)
        (u_opt, v_opt), weight = self._lct.get_max_edge(u_str, v_str)
        u_deserialized = _deserialize(u_opt) if u_opt else None
        v_deserialized = _deserialize(v_opt) if v_opt else None
        return (u_deserialized, v_deserialized), weight


if __name__ == '__main__':
    tree = PyLinkCutTree()

    tree.link((1, 1), (2, 2), 8)
    tree.link((1, 1), (np.float64(7), np.float64(7)), 8)
    tree.link((1, 1), (3, 3), 8)
    tree.link((1, 1), (5, 5), 8)
    tree.link((6, 6), (5, 5), 8)
    tree.cut((1, 1), (7, 7))
    tree.link((2, 2), (np.float64(7), np.float64(7)), 8)
    tree.cut((3, 3), (1, 1))
    tree.link((3, 3), (np.float64(7), np.float64(7)), 8)
    tree.cut((2, 2), (1, 1))
    print(tree.is_connection((3, 3), (6, 6)))
    print(tree.get_max_edge((3, 3), (6, 6)))
    # 获取最大边
