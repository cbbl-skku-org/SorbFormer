from sklearn.ensemble import AdaBoostRegressor
import argparse
import numpy as np
import os
import warnings
import joblib
import pandas as pd
from lightning.pytorch import seed_everything
from sklearn.model_selection import KFold
from preprocess_utils import fit_transform_fold, fit_transform_train_test, infer_embedding_cols, maybe_log10_target, restore_target_scale, evaluate_model_pair, write_results_txt_dual

warnings.filterwarnings("ignore")


def split_fold(n_folds, n_samples, seed=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(n_samples)
    return kf, indices


def load_xy(x_csv: str, y_csv: str):
    X_df = pd.read_csv(x_csv)
    y_df = pd.read_csv(y_csv)

    if y_df.shape[1] > 1:
        y = y_df.iloc[:, 0].to_numpy()
    else:
        y = y_df.squeeze("columns").to_numpy()

    return X_df, y, X_df.columns.tolist()


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    X_all, y_all, train_cols = load_xy(args.train_x_csv, args.train_y_csv)
    X_test, y_test, test_cols = load_xy(args.test_x_csv, args.test_y_csv)
    y_all = maybe_log10_target(y_all, args.train_x_csv, args.train_y_csv)
    y_test = maybe_log10_target(y_test, args.test_x_csv, args.test_y_csv)

    embedding_cols = infer_embedding_cols(X_all)

    if train_cols != test_cols:
        raise ValueError("Train/Test feature columns are not identical (order or names differ).")

    kf, indices = split_fold(n_folds=args.fold, n_samples=len(y_all), seed=args.seed)

    # Hyperparameter grid
    lr_list = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
    n_estimators_list = list(range(40, 310, 10))
    loss_list = ["exponential", "linear", "square"]

    best = {
        "mean_r2": -np.inf,
        "params": None,
        "cv_mean_current": None,
        "cv_std_current": None,
        "cv_mean_original": None,
        "cv_std_original": None,
    }

    # Cross-validation
    for lr in lr_list:
        for n_estimators in n_estimators_list:
            for loss in loss_list:
                fold_scores_current = []
                fold_scores_original = []
                for train_idx, val_idx in kf.split(indices):
                    X_train_df = X_all.iloc[train_idx].reset_index(drop=True)
                    X_val_df = X_all.iloc[val_idx].reset_index(drop=True)
                    y_train, y_val = y_all[train_idx], y_all[val_idx]

                    X_train_p, X_val_p, _ = fit_transform_fold(
                        X_train_df=X_train_df,
                        X_val_df=X_val_df,
                        embedding_cols=embedding_cols,
                        do_scale=True,
                    )

                    X_train = X_train_p.to_numpy()
                    X_val = X_val_p.to_numpy()
                    model = AdaBoostRegressor(
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        loss=loss,
                        random_state=123,
                    )
                    model.fit(X_train, y_train)

                    val_pred = model.predict(X_val)
                    score_bundle = evaluate_model_pair(y_val, val_pred, args.train_x_csv, args.train_y_csv)
                    fold_scores_current.append(score_bundle["current"])
                    if score_bundle["original"] is not None:
                        fold_scores_original.append(score_bundle["original"])

                fold_scores_current = np.array(fold_scores_current, dtype=float)
                cv_mean_current = np.nanmean(fold_scores_current, axis=0)
                cv_std_current = np.nanstd(fold_scores_current, axis=0)

                cv_mean_original = None
                cv_std_original = None
                if len(fold_scores_original) > 0:
                    fold_scores_original = np.array(fold_scores_original, dtype=float)
                    cv_mean_original = np.nanmean(fold_scores_original, axis=0)
                    cv_std_original = np.nanstd(fold_scores_original, axis=0)

                selection_r2 = cv_mean_original[3] if cv_mean_original is not None else cv_mean_current[3]
                if selection_r2 > best["mean_r2"]:
                    best["mean_r2"] = selection_r2
                    best["params"] = {
                        "n_estimators": n_estimators,
                        "learning_rate": lr,
                        "loss": loss,
                        "random_state": 123,
                    }
                    best["cv_mean_current"] = cv_mean_current
                    best["cv_std_current"] = cv_std_current
                    best["cv_mean_original"] = cv_mean_original
                    best["cv_std_original"] = cv_std_original

    if best["params"] is None:
        raise RuntimeError("No hyperparameter set was selected. Check your data and search loop.")

    # Refit best model

    X_all_p, X_test_p, _ = fit_transform_train_test(
        X_train_df=X_all,
        X_test_df=X_test,
        embedding_cols=embedding_cols,
        do_scale=True,
    )

    X_all_final = X_all_p.to_numpy()
    X_test_final = X_test_p.to_numpy()

    final_model = AdaBoostRegressor(**best["params"])

    final_model.fit(X_all_final, y_all)

    # Evaluate fitted model
    train_pred = final_model.predict(X_all_final)
    train_scores = evaluate_model_pair(y_all, train_pred, args.train_x_csv, args.train_y_csv)

    test_pred = final_model.predict(X_test_final)
    test_scores = evaluate_model_pair(y_test, test_pred, args.test_x_csv, args.test_y_csv)


    # Save outputs

    model_out = os.path.join(args.results_dir, "AdaBoost_best_model.pkl")
    joblib.dump(final_model, model_out)

    y_test_out = restore_target_scale(y_test, args.test_x_csv, args.test_y_csv)
    test_pred_out = restore_target_scale(test_pred, args.test_x_csv, args.test_y_csv)

    prob_path = os.path.join(args.results_dir, "prob_and_gt.csv")
    with open(prob_path, "w") as f:
        f.write("True,Predicted\n")
        for t, p in zip(y_test_out, test_pred_out):
            f.write(f"{t},{p}\n")

    results_path = os.path.join(args.results_dir, "results.txt")
    write_results_txt_dual(
        results_path,
        model_label="AdaBoost",
        best_params=best["params"],
        cv_mean_current=best["cv_mean_current"],
        cv_std_current=best["cv_std_current"],
        train_scores_current=train_scores["current"],
        test_scores_current=test_scores["current"],
        cv_mean_original=best["cv_mean_original"],
        cv_std_original=best["cv_std_original"],
        train_scores_original=train_scores["original"],
        test_scores_original=test_scores["original"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_x_csv", type=str, required=True, help="Training X csv")
    parser.add_argument("--train_y_csv", type=str, required=True, help="Training y csv")
    parser.add_argument("--test_x_csv", type=str, required=True, help="Testing X csv")
    parser.add_argument("--test_y_csv", type=str, required=True, help="Testing y csv")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--fold", type=int, default=5, help="Fold number for cross-validation")
    parser.add_argument("--results_dir", type=str, default="./results/AdaBoost", help="Directory to save results")

    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)
