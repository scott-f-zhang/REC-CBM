import os
import random
import pandas as pd
import numpy as np
import torch
from typing import Dict, Optional, Set, Tuple, List, Any, Callable


def safe_model_name(name: str) -> str:
    """Filesystem-safe version of a HuggingFace model id.

    Replaces ``/`` with ``_`` so names like ``facebook/bart-base`` can be used
    as directory or file components.
    """
    return name.replace("/", "_")


def get_average_scores(score_list):
    """Average a list of (acc, macro_f1) tuples and scale to percentage.

    Returns a pair (acc_pct, macro_f1_pct) in [0, 100].
    """
    if not score_list:
        return (0.0, 0.0)
    s1 = s2 = 0.0
    n = 0
    for a, b in score_list:
        s1 += a
        s2 += b
        n += 1
    return ((s1 / n * 100), (s2 / n * 100))


def get_tuple_2f_fmt(tp):
    """Format a (x, y) tuple as a string with two decimals: 'xx.xx/yy.yy'."""
    f1, f2 = tp
    return f"{f1:.2f}/{f2:.2f}"


def get_existing_result_keys(dataset: str, write_mode: str, results_root_getter: Optional[Callable[[], str]] = None) -> Set[str]:
    """Get existing result keys for a dataset.
    
    Args:
        dataset: Dataset name
        write_mode: Write mode ('append' or 'replace')
        results_root_getter: Function to get results root (must be provided)
    
    Returns:
        Set of model names that already have results
    """
    if write_mode != "append":
        return set()
    
    if results_root_getter is None:
        raise ValueError("results_root_getter must be provided")
    
    out_dir = os.path.join(results_root_getter(), dataset)
    out_file = os.path.join(out_dir, f"result_{dataset}.csv")
    try:
        if os.path.exists(out_file):
            existing_df = pd.read_csv(out_file)
            if not existing_df.empty and "model" in existing_df.columns:
                return set(existing_df["model"].astype(str))
    except Exception as e:
        print(f"\t\tWarning: failed to read existing results for {dataset}: {e}")
    return set()



def normalize_result(result: Any) -> Tuple[List[float], List[float]]:
    """Normalize result from joint training to (task_scores, concept_scores).
    
    Args:
        result: Result dict with 'task' and 'concept' keys, or empty list
    
    Returns:
        Tuple of (task_scores, concept_scores) as lists
    """
    if isinstance(result, dict):
        return result.get('task', []), result.get('concept', [])
    else:
        return [], []


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all random number generators.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator (CPU and CUDA)
    
    Args:
        seed: Random seed value (default: 42)
    
    Note:
        For fully deterministic behavior, you may also want to set:
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False
        However, these may slow down training significantly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: for fully deterministic behavior (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def print_task_completion_time(start_time: float):
    """Print task completion time in a formatted way.
    
    Args:
        start_time: Start time from time.time()
    """
    import time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "=" * 80)
    print("Task completed!")
    if hours > 0:
        print(f"Total elapsed time: {hours} hours {minutes} minutes {seconds} seconds ({elapsed_time:.2f} seconds)")
    elif minutes > 0:
        print(f"Total elapsed time: {minutes} minutes {seconds} seconds ({elapsed_time:.2f} seconds)")
    else:
        print(f"Total elapsed time: {seconds} seconds ({elapsed_time:.2f} seconds)")
    print("=" * 80)
        