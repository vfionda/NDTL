import NetworkGenerator as generator
import NDTL_discovery as ndtl
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import itertools
import pandas as pd
import os
import numpy as np
import time

def tenFoldCrossValidation(discoveryTh=1.0,tau=0.2,true_graphs = generator.generate_diffusion_graphs(500,False),false_graphs = generator.generate_diffusion_graphs(500,True)):


    all_graphs = true_graphs + false_graphs
    random.shuffle(all_graphs)

    # Split the dataset into 10 folds using KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    data = list(kf.split(all_graphs))

    # Metrics to store results for each fold
    metrics_per_fold = []

    all_y_true_true = []
    all_y_probs0_true = []
    all_y_probs1_true = []
    all_y_probs2_true = []

    all_y_true_false = []
    all_y_probs0_false = []
    all_y_probs1_false = []
    all_y_probs2_false = []

    for fold_idx, (train_indices, val_indices) in enumerate(data):
        # Split into training and validation sets
        print("split ",fold_idx+1, "train: ",len(train_indices), "test: ", len(val_indices))
        train_data = [all_graphs[i] for i in train_indices]
        val_data = [all_graphs[i] for i in val_indices]

        countTRUE = sum([1 for root, graph, label in train_data if label == "true"])
        countFALSE = sum([1 for root, graph, label in train_data if label == "false"])
        print(f"TRUE in train {countTRUE} FALSE IN train {countFALSE}")


        # Separate validation graphs and labels
        val_graphs = {root: graph for root, graph, label in val_data}
        val_labels = {root: label for root, graph, label in val_data}

        print("split ", fold_idx + 1, "separated train e validation")

        formulas = ndtl.ndtl_discovery(
            {root: graph for root, graph, label in train_data if label == "true"},
            {root: graph for root, graph, label in train_data if label == "false"},
            discoveryTh
        )

        print("split ", fold_idx + 1, "formulas discovered")


        predictions0 = {}
        predictions1 = {}
        predictions2 = {}
        for root, graph in val_graphs.items():
            result = ndtl.evaluate_formulas(graph, root, formulas["false"], formulas["true"], tau)
            print(val_labels[root], result)
            predictions0[root] = result["classification0"]
            predictions1[root] = result["classification1"]
            predictions2[root] = result["classification2"]


        # Map predictions to binary labels for metric computation
        y_true = [1 if val_labels[root] == "false" else 0 for root in val_labels]
        y_pred0 = [1 if predictions0[root] == "Fake" else 0 for root in predictions0]
        y_pred1 = [1 if predictions1[root] == "Fake" else 0 for root in predictions1]
        y_pred2 = [1 if predictions2[root] == "Fake" else 0 for root in predictions2]

        all_y_true_false.extend(y_true)
        all_y_probs0_false.extend(y_pred0)
        all_y_probs1_false.extend(y_pred1)
        all_y_probs2_false.extend(y_pred2)

        classified_false0 = sum(y_pred0)
        classified_false1 = sum(y_pred1)
        classified_false2 = sum(y_pred2)

        precision0, recall0, f10, _ = precision_recall_fscore_support(y_true, y_pred0, zero_division=0)
        precision1, recall1, f11, _ = precision_recall_fscore_support(y_true, y_pred1, zero_division=0)
        precision2, recall2, f12, _ = precision_recall_fscore_support(y_true, y_pred2, zero_division=0)
        print("VAL FAKE 0:", precision0, recall0, f10)
        print("VAL FAKE 1:", precision1, recall1, f11)
        print("VAL FAKE 2:", precision2, recall2, f12)


        metrics = {
            "Fake": {
                "precision0": precision0[1],
                "recall0": recall0[1],
                "f10": f10[1],
                "precision1": precision1[1],
                "recall1": recall1[1],
                "f11": f11[1],
                "precision2": precision2[1],
                "recall2": recall2[1],
                "f12": f12[1]
            },
            "total_fake_news": sum(1 for label in val_labels.values() if label == "false"),
            "classified_fake0": classified_false0,
            "classified_fake1": classified_false1,
            "classified_fake2": classified_false2,
            "total_news": len(y_true)
        }

        # Print class-wise metrics
        print(
            f"Fake News - Precision0: {metrics['Fake']['precision0']}, Recall0: {metrics['Fake']['recall0']}, F1-score0: {metrics['Fake']['f10']}")
        print(
            f"Fake News - Precision1: {metrics['Fake']['precision1']}, Recall1: {metrics['Fake']['recall1']}, F1-score1: {metrics['Fake']['f11']}")
        print(
            f"Fake News - Precision2: {metrics['Fake']['precision2']}, Recall2: {metrics['Fake']['recall2']}, F1-score2: {metrics['Fake']['f12']}")
        print(f"Total Predictions0 Fake: {metrics['classified_fake0']}, Total Predictions1 Fake: {metrics['classified_fake1']}, Total Predictions2 Fake: {metrics['classified_fake2']}, Total Fake News: {metrics['total_fake_news']}, Total News: {metrics['total_news']}")

        # Map predictions to binary labels for metric computation
        y_true = [1 if val_labels[root] == "true" else 0 for root in val_labels]
        y_pred0 = [1 if predictions0[root] == "True" else 0 for root in predictions0]
        y_pred1 = [1 if predictions1[root] == "True" else 0 for root in predictions1]
        y_pred2 = [1 if predictions2[root] == "True" else 0 for root in predictions2]

        all_y_true_true.extend(y_true)
        all_y_probs0_true.extend(y_pred0)
        all_y_probs1_true.extend(y_pred1)
        all_y_probs2_true.extend(y_pred2)

        classified_true0 = sum(y_pred0)
        classified_true1 = sum(y_pred1)
        classified_true2 = sum(y_pred2)

        precision0, recall0, f10, _ = precision_recall_fscore_support(y_true, y_pred0, zero_division=0)
        precision1, recall1, f11, _ = precision_recall_fscore_support(y_true, y_pred1, zero_division=0)
        precision2, recall2, f12, _ = precision_recall_fscore_support(y_true, y_pred2, zero_division=0)
        print("VAL TRUE 0:", precision0, recall0, f10)
        print("VAL TRUE 1:", precision1, recall1, f11)
        print("VAL TRUE 2:", precision2, recall2, f12)


        metrics["True"] = {
            "precision0": precision0[1],
                "recall0": recall0[1],
                "f10": f10[1],
                "precision1": precision1[1],
                "recall1": recall1[1],
                "f11": f11[1],
                "precision2": precision2[1],
                "recall2": recall2[1],
                "f12": f12[1]
        }

        metrics["total_true_news"]= sum(1 for label in val_labels.values() if label == "true")
        metrics["classified_true0"] = classified_true0
        metrics["classified_true1"] = classified_true1
        metrics["classified_true2"] = classified_true2

        print(
            f"True News - Precision0: {metrics['True']['precision0']}, Recall0: {metrics['True']['recall0']}, F1-score0: {metrics['True']['f10']}")
        print(
            f"True News - Precision1: {metrics['True']['precision1']}, Recall1: {metrics['True']['recall1']}, F1-score1: {metrics['True']['f11']}")
        print(
            f"True News - Precision2: {metrics['True']['precision2']}, Recall2: {metrics['True']['recall2']}, F1-score2: {metrics['True']['f12']}")
        print(
            f"Total Predictions0 True: {metrics['classified_true0']}, Total Predictions1 True: {metrics['classified_true1']}, Total Predictions2 True: {metrics['classified_true2']}, Total True News: {metrics['total_true_news']}, Total News: {metrics['total_news']}")

        # Store metrics for this fold
        metrics_per_fold.append(metrics)

    avg_metrics = {
        "Fake": {
            "precision0": sum(m["Fake"]["precision0"] for m in metrics_per_fold) / len(metrics_per_fold),
            "recall0": sum(m["Fake"]["recall0"] for m in metrics_per_fold) / len(metrics_per_fold),
            "f10": sum(m["Fake"]["f10"] for m in metrics_per_fold) / len(metrics_per_fold),
            "precision1": sum(m["Fake"]["precision1"] for m in metrics_per_fold) / len(metrics_per_fold),
            "recall1": sum(m["Fake"]["recall1"] for m in metrics_per_fold) / len(metrics_per_fold),
            "f11": sum(m["Fake"]["f11"] for m in metrics_per_fold) / len(metrics_per_fold),
            "precision2": sum(m["Fake"]["precision2"] for m in metrics_per_fold) / len(metrics_per_fold),
            "recall2": sum(m["Fake"]["recall2"] for m in metrics_per_fold) / len(metrics_per_fold),
            "f12": sum(m["Fake"]["f12"] for m in metrics_per_fold) / len(metrics_per_fold)
        },
        "True": {
            "precision0": sum(m["True"]["precision0"] for m in metrics_per_fold) / len(metrics_per_fold),
            "recall0": sum(m["True"]["recall0"] for m in metrics_per_fold) / len(metrics_per_fold),
            "f10": sum(m["True"]["f10"] for m in metrics_per_fold) / len(metrics_per_fold),
            "precision1": sum(m["True"]["precision1"] for m in metrics_per_fold) / len(metrics_per_fold),
            "recall1": sum(m["True"]["recall1"] for m in metrics_per_fold) / len(metrics_per_fold),
            "f11": sum(m["True"]["f11"] for m in metrics_per_fold) / len(metrics_per_fold),
            "precision2": sum(m["True"]["precision2"] for m in metrics_per_fold) / len(metrics_per_fold),
            "recall2": sum(m["True"]["recall2"] for m in metrics_per_fold) / len(metrics_per_fold),
            "f12": sum(m["True"]["f12"] for m in metrics_per_fold) / len(metrics_per_fold)
        },
        "total_classified_fake0": sum(m["classified_fake0"] for m in metrics_per_fold),
        "total_classified_fake1": sum(m["classified_fake1"] for m in metrics_per_fold),
        "total_classified_fake2": sum(m["classified_fake2"] for m in metrics_per_fold),
        "total_classified_true0": sum(m["classified_true0"] for m in metrics_per_fold),
        "total_classified_true1": sum(m["classified_true1"] for m in metrics_per_fold),
        "total_classified_true2": sum(m["classified_true2"] for m in metrics_per_fold),
        "total_fake_news": sum(m["total_fake_news"] for m in metrics_per_fold),
        "total_true_news": sum(m["total_true_news"] for m in metrics_per_fold),
        "total_news": sum(m["total_news"] for m in metrics_per_fold)
    }


    print("Average Metrics Across Folds:", avg_metrics)


    return avg_metrics

if __name__ == "__main__":
    #tenFoldCrossValidationROC(discoveryTh=1,tau=0.347)
    # Separate graphs into true and false categories
  #  true_graphs = generator.generate_diffusion_graphs(500, False)
  #  false_graphs = generator.generate_diffusion_graphs(500, True)
  #  tenFoldCrossValidation(discoveryTh=0.7, tau=0.45, true_graphs=true_graphs, false_graphs=false_graphs)

  #  min_max_depthF_range = [6, 7, 8]
  #  max_max_depthF_range = [9, 10, 12]
  #  min_branching_factorF_range = [3, 4, 5]
  # max_branching_factorF_range = [6, 7, 8]
  #  time_scaleF_range = [0.3, 0.4, 0.5]
  #  cascade_probF_range = [0.7, 0.8, 0.9]

  #  min_max_depthT_range = [3, 4, 5]
  #  max_max_depthT_range = [5, 6, 7]
  #  min_branching_factorT_range = [2, 3, 4]
  #  max_branching_factorT_range = [3, 4, 5]
  #  time_scaleT_range = [0.8, 1.0, 1.2]
  #  cascade_probT_range = [0.4, 0.5, 0.6]

    min_max_depthF_range = [6]
    max_max_depthF_range = [9]
    min_branching_factorF_range = [3]
    max_branching_factorF_range = [6]
    time_scaleF_range = [0.5]
    cascade_probF_range = [0.7, 0.8]

    min_max_depthT_range = [3]
    max_max_depthT_range = [5]
    min_branching_factorT_range = [2]
    max_branching_factorT_range = [4]
    time_scaleT_range = [0.8, 1, 1.2]
    cascade_probT_range = [0.4, 0.5, 0.6]

    discoveryTh_range = [0.7, 0.8, 0.9, 1.0]
    #tau_range = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    tau_range = [0.2, 0.3,  0.4]

    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Zip the pairs that need to be iterated index-wise:
    paired_depthF = zip(min_max_depthF_range, max_max_depthF_range)
    paired_branchF = zip(min_branching_factorF_range, max_branching_factorF_range)
    paired_depthT = zip(min_max_depthT_range, max_max_depthT_range)
    paired_branchT = zip(min_branching_factorT_range, max_branching_factorT_range)

    for ((min_max_depthF, max_max_depthF),
         (min_branching_factorF, max_branching_factorF),
         (min_max_depthT, max_max_depthT),
         (min_branching_factorT, max_branching_factorT),
         time_scaleF, cascade_probF,
         time_scaleT, cascade_probT) in itertools.product(
        paired_depthF, paired_branchF, paired_depthT, paired_branchT,
        time_scaleF_range, cascade_probF_range,
        time_scaleT_range, cascade_probT_range
    ):
        print(min_max_depthF, max_max_depthF,
              min_branching_factorF, max_branching_factorF,
              time_scaleF, cascade_probF,
              min_max_depthT, max_max_depthT,
              min_branching_factorT, max_branching_factorT,
              time_scaleT, cascade_probT)

        true_graphs = generator.generate_diffusion_graphs(500, fake=False, min_max_depth=min_max_depthT, max_max_depth=max_max_depthT, min_branching_factor=min_branching_factorT, max_branching_factor=max_branching_factorT, time_scale=time_scaleT, cascade_prob=cascade_probT)
        false_graphs = generator.generate_diffusion_graphs(500, True, min_max_depth=min_max_depthF, max_max_depth=max_max_depthF, min_branching_factor=min_branching_factorF, max_branching_factor=max_branching_factorF, time_scale=time_scaleF, cascade_prob=cascade_probF)

        results = []

        for discoveryTh, tau in itertools.product(discoveryTh_range, tau_range):
            avg_metrics = tenFoldCrossValidation(discoveryTh=discoveryTh, tau=tau, true_graphs=true_graphs, false_graphs=false_graphs)

            filename = f"results_th{discoveryTh}_tau{tau}_mdF{min_max_depthF}-{max_max_depthF}_mdT{min_max_depthT}-{max_max_depthT}_bfF{min_branching_factorF}-{max_branching_factorF}_bfT{min_branching_factorT}-{max_branching_factorT}_tsF{time_scaleF}_tsT{time_scaleT}_cpF{cascade_probF}_cpT{cascade_probT}.csv"
            file_path = os.path.join(output_dir, filename)

            results.append({
                "min_max_depthF": min_max_depthF,
                "max_max_depthF": max_max_depthF,
                "min_branching_factorF": min_branching_factorF,
                "max_branching_factorF": max_branching_factorF,
                "time_scaleF": time_scaleF,
                "cascade_probF": cascade_probF,
                "min_max_depthT": min_max_depthT,
                "max_max_depthT": max_max_depthT,
                "min_branching_factorT": min_branching_factorT,
                "max_branching_factorT": max_branching_factorT,
                "time_scaleT": time_scaleT,
                "cascade_probT": cascade_probT,
                "discoveryTh": discoveryTh,
                "tau": tau,
                "precision0_Fake": avg_metrics["Fake"]["precision0"],
                "recall0_Fake": avg_metrics["Fake"]["recall0"],
                "f10_Fake": avg_metrics["Fake"]["f10"],
                "precision1_Fake": avg_metrics["Fake"]["precision1"],
                "recall1_Fake": avg_metrics["Fake"]["recall1"],
                "f11_Fake": avg_metrics["Fake"]["f11"],
                "precision0_True": avg_metrics["True"]["precision0"],
                "recall0_True": avg_metrics["True"]["recall0"],
                "f10_True": avg_metrics["True"]["f10"],
                "precision1_True": avg_metrics["True"]["precision1"],
                "recall1_True": avg_metrics["True"]["recall1"],
                "f11_True": avg_metrics["True"]["f11"]
            })

            print(
                f"Completed: DiscoveryTh={discoveryTh}, Tau={tau}, Params: {min_max_depthF, max_max_depthF, min_branching_factorF, max_branching_factorF,min_max_depthT, max_max_depthT, min_branching_factorT, max_branching_factorT}")

            # Convert results to a DataFrame and save to CSV
        df_all = pd.DataFrame(results)
        df_all.to_csv(file_path, index=False)
    print(f"\n All simulations completed! Results are stored in {output_dir}")

    # if not df_all.empty:
    #     # For example, select results matching the first configuration:
    #     config = df_all.iloc[0]
    #     filter_dict = {
    #         "min_max_depthF": config["min_max_depthF"],
    #         "max_max_depthF": config["max_max_depthF"],
    #         "min_branching_factorF": config["min_branching_factorF"],
    #         "max_branching_factorF": config["max_max_depthF"],  # (Note: Adjust if needed)
    #         "time_scaleF": config["time_scaleF"],
    #         "cascade_probF": config["cascade_probF"],
    #         "min_max_depthT": config["min_max_depthT"],
    #         "max_max_depthT": config["max_max_depthT"],
    #         "min_branching_factorT": config["min_branching_factorT"],
    #         "max_branching_factorT": config["max_branching_factorT"],
    #         "time_scaleT": config["time_scaleT"],
    #         "cascade_probT": config["cascade_probT"]
    #     }
    #     filtered_df = df_all.copy()
    #     for key, value in filter_dict.items():
    #         filtered_df = filtered_df[filtered_df[key] == value]
    # else:
    #     filtered_df = df_all
    #
    #
    # # We will use discoveryTh as the row index (th_disc/tau) and tau as the column header.
    # # For each metric, we pivot the data accordingly.
    # def pivot_metric(df, metric_fake, metric_true):
    #     pivot_fake = df.pivot(index="discoveryTh", columns="tau", values=metric_fake).reindex(discoveryTh_range)
    #     pivot_true = df.pivot(index="discoveryTh", columns="tau", values=metric_true).reindex(discoveryTh_range)
    #     pivot_fake.reset_index(inplace=True)
    #     pivot_true.reset_index(inplace=True)
    #     return pivot_fake, pivot_true
    #
    #
    # precision_fake_df, precision_true_df = pivot_metric(filtered_df, "precision0_Fake", "precision0_True")
    # recall_fake_df, recall_true_df = pivot_metric(filtered_df, "recall0_Fake", "recall0_True")
    # f1_fake_df, f1_true_df = pivot_metric(filtered_df, "f10_Fake", "f10_True")
    # # Here, we use the precision1 columns as a placeholder for "num classified"
    # num_classified_fake_df, num_classified_true_df = pivot_metric(filtered_df, "precision1_Fake", "precision1_True")
    #
    #
    # # Function to combine FAKE and TRUE data side by side with a header row
    # def combine_fake_true(df_fake, df_true, metric_name):
    #     # Insert an empty column between fake and true dataframes
    #     df_combined = pd.concat([df_fake, pd.DataFrame([[""]] * df_fake.shape[0], columns=[""]), df_true], axis=1)
    #     header = [metric_name] + [""] * (df_fake.shape[1] - 1) + [""] + [metric_name] + [""] * (df_true.shape[1] - 1)
    #     header_df = pd.DataFrame([header], columns=df_combined.columns)
    #     df_combined = pd.concat([header_df, df_combined], ignore_index=True)
    #     return df_combined
    #
    #
    # df_precision_structured = combine_fake_true(precision_fake_df, precision_true_df, "Precision")
    # df_recall_structured = combine_fake_true(recall_fake_df, recall_true_df, "Recall")
    # df_f1_structured = combine_fake_true(f1_fake_df, f1_true_df, "F1")
    # df_num_classified_structured = combine_fake_true(num_classified_fake_df, num_classified_true_df, "num classified")
    #
    # # Save structured results to an Excel file
    # structured_output_file = os.path.join(output_dir, "structured_simulation_results.xlsx")
    # with pd.ExcelWriter(structured_output_file) as writer:
    #     df_precision_structured.to_excel(writer, sheet_name="Precision", index=False)
    #     df_recall_structured.to_excel(writer, sheet_name="Recall", index=False)
    #     df_f1_structured.to_excel(writer, sheet_name="F1", index=False)
    #     df_num_classified_structured.to_excel(writer, sheet_name="Num_Classified", index=False)
    #
    # print(f"\n✅ Structured simulation results saved to {structured_output_file}")




