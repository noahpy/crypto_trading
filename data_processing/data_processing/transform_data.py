
from typing import List, Callable

def transform_data_to_time_snippets(data: dict, feature_schema: dict[str, Callable[dict]]) -> List[dict]:
    """
    Transformes given data accoring to the given feature_schema.
    feature_schema maps feature names to functions computing these, given the data.
    """
    pass
