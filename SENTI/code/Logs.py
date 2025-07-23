# Logs.py

import os
import pandas as pd

def save_evaluation(
    file_name: str,
    start_index: int,
    end_index: int,
    nulls: int,
    exact_matches: int,
    avg_similarity: float | None,
    processing_time: float,
    faiss_time: float,
    imputation_time: float,
    total_processing_time: float,
    out_csv: str,
) -> None:
    df = pd.DataFrame(
        [{
            "file":                  file_name,
            "start_index":           start_index,
            "end_index":             end_index,
            "nulls":                 nulls,
            "exact_matches_SENTI":         exact_matches,
            "avg_semantic_sim_SENTI":        avg_similarity,
            "processing_time_SENTI":       processing_time,
            "faiss_time_SENTI":            faiss_time,
            "imputation_time_SENTI":       imputation_time,
            "total_time_SENTI": total_processing_time
        }]
    )
    df.to_csv(out_csv, index=False)
    print("    Saved evaluation CSV:", os.path.basename(out_csv))
