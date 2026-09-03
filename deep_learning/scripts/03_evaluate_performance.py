import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd
from pathlib import Path



def find_best_test_trials(root_dirs):
    all_summaries = []

    for root in root_dirs:
        root_path = Path(root)
        if not root_path.exists():
            continue

        print(f"🔍 Searching for trial_summary.json in {root}...")
        summary_files = list(root_path.glob("**/trial_summary.json"))
        print(f"📂 Found {len(summary_files)} summaries.")

        for sf in summary_files:
            try:
                with open(sf, "r") as f:
                    data = json.load(f)

                # Get the study directory (parent of trial_X)
                # e.g., SorbFormer_results_final/tscf/biot5/gated
                study_dir = sf.parent.parent

                # Identify Target and Mode from path
                target = "rcf" if "rcf" in str(study_dir).lower() else "tscf"
                mode = "rmse" if "rmse" in str(study_dir).lower() else "r2"

                # Identify Model and Fusion from path
                parts = list(study_dir.parts)
                model = parts[-2] if len(parts) >= 2 else "unknown"
                fusion = parts[-1] if len(parts) >= 1 else "unknown"

                # Extract metrics
                test_metrics = data.get("test_metrics", {})
                if not test_metrics:
                    # Try full_refit_test_metrics (older name logic)
                    test_metrics = data.get("full_refit_test_metrics", {})

                if not test_metrics:
                    continue

                row = {
                    "StudyName": study_dir.name,
                    "Target": target.upper(),
                    "Mode": mode.upper(),
                    "Model": model,
                    "Fusion": fusion,
                    "Trial": data.get("trial_number"),
                    "Test_R2": test_metrics.get("r2"),
                    "Test_RMSE": test_metrics.get("rmse"),
                    "Test_MAE": test_metrics.get("mae"),
                    "CV_Score": (
                        data.get("cv_performance")
                        if data.get("cv_performance") is not None
                        else data.get("mean_val_score")
                    ),
                    "Folder": str(study_dir),
                }
                all_summaries.append(row)
            except Exception as e:
                print(f"⚠️ Error reading {sf}: {e}")

    if not all_summaries:
        print("❌ No trial summaries found.")
        return

    df = pd.DataFrame(all_summaries)

    # Group by Folder and find best Trial based on Test_R2 (Maximize)
    # Or Test_RMSE (Minimize) depending on Mode?
    # Usually we want best R2 on test set.

    result_list = []

    for folder, group in df.groupby("Folder"):
        # For simplicity, let's just find the max R2 on Test Set
        # If R2 is missing, maybe fallback to RMSE
        target_mode = group["Mode"].iloc[0]

        if target_mode == "R2":
            best_idx = group["Test_R2"].idxmax()
        else:
            best_idx = group["Test_RMSE"].idxmin()

        best_row = group.loc[best_idx].to_dict()
        result_list.append(best_row)

    best_df = pd.DataFrame(result_list)

    # Display results
    for target in ["RCF", "TSCF"]:
        for mode in ["R2", "RMSE"]:
            mask = (best_df["Target"] == target) & (best_df["Mode"] == mode)
            sub_df = best_df[mask].copy()

            if not sub_df.empty:
                print("\n" + "=" * 110)
                print(
                    f"🏆 BEST TEST PERFORMANCE | TARGET: {target} | OPTIMIZATION: {mode}"
                )
                print("=" * 110)

                # Sort by Test_R2 for display
                sub_df = sub_df.sort_values(by="Test_R2", ascending=False)

                display_cols = [
                    "Model",
                    "Fusion",
                    "Trial",
                    "Test_R2",
                    "Test_RMSE",
                    "Test_MAE",
                    "CV_Score",
                ]
                print(sub_df[display_cols].to_string(index=False))

    best_df.to_csv("best_test_performance_summary.csv", index=False)
    print(
        f"\n✅ Best test performance summary saved to best_test_performance_summary.csv"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and summarize test performance across trials.")
    parser.add_argument(
        "--results_folder",
        type=str,
        default=None,
        help="Specific results directory to search for trial_summary.json files.",
    )
    args = parser.parse_args()

    if args.results_folder:
        search_paths = [args.results_folder]
    else:
        search_paths = [
            "SorbFormer_results",
            "SorbFormer_results_rmse",
            "SorbFormer_results_final",
            "SorbFormer_results_final_rmse",
            "final_models_rcf",
            "final_models_tscf",
        ]


    find_best_test_trials(search_paths)

