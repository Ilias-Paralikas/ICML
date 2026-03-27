import inspect, os
from pathlib import Path

def serialize_config(obj):
    """
    Recursively serialize config to a JSON/YAML–friendly format.
    Handles:
      - primitives
      - dict, list, tuple
      - custom objects (instance variables)
      - classes/types (e.g. torch.optim.Adam -> {"type": "Adam"})
      - paths (Path or os.path -> string)
    """
    # Base cases
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj

    # Paths
    if isinstance(obj, (Path, os.PathLike)):
        return str(obj)

    # Dict
    if isinstance(obj, dict):
        return {k: serialize_config(v) for k, v in obj.items()}

    # List/Tuple
    if isinstance(obj, (list, tuple)):
        return [serialize_config(x) for x in obj]

    # Class/type objects (like torch.optim.Adam)
    if isinstance(obj, type):
        return {"type": obj.__name__, "module": obj.__module__}

    # Instances (custom objects)
    cls_name = obj.__class__.__name__
    params = {}
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            params[k] = serialize_config(v)

    return {"type": cls_name, "params": params}
