"""Read centralised hyperparameters from ``config/hyperparams.json``."""
import json
import os

_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "hyperparams.json"
)


def load_hyperparams(path: str = _DEFAULT_PATH) -> dict:
    """Load the full hyperparams dict."""
    with open(path) as f:
        return json.load(f)


def get_method_params(
    dataset: str,
    model: str,
    method: str,
    path: str = _DEFAULT_PATH,
) -> dict:
    """Return merged ``shared`` + method-specific params for a combo.

    Method-specific values override ``shared`` if keys overlap.
    """
    hp = load_hyperparams(path)
    if dataset not in hp:
        raise KeyError(f"Dataset '{dataset}' not found in hyperparams.json")
    ds = hp[dataset]
    if model not in ds:
        raise KeyError(f"Model '{model}' not found in hyperparams.json[{dataset}]")
    model_params = ds[model]
    if method not in model_params:
        raise KeyError(
            f"Method '{method}' not found in hyperparams.json[{dataset}][{model}]"
        )

    result = dict(model_params.get("shared", {}))
    result.update(model_params[method])
    return result
