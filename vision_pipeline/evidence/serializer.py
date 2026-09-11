import numpy as np

def serialize_pipeline_result(obj):
    """Recursively convert NumPy arrays and custom types into standard Python primitives for JSON encoding."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: serialize_pipeline_result(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_pipeline_result(i) for i in obj]
    return obj
