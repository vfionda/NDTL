import Dataset as dataset
import NDTL_discovery as ndtl
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
import itertools

def evaluate_model(y_true, y_probs):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Fake News Detection')
    plt.legend(loc='lower right')
    plt.show()

    # Find the optimal threshold
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f'Optimal Decision Threshold: {optimal_threshold:.3f}')

    return optimal_threshold


def main():
    # File paths
    nodes_file = "../../dataset/completi/rumor_detection_acl2017/twitter16/final/nodes.txt"
    edges_file = "../../dataset/completi/rumor_detection_acl2017/twitter16/final/edges.txt"
    labels_file = "../../dataset/completi/rumor_detection_acl2017/twitter16/filtered_label.txt"

    # Read nodes, edges, and labels
    nodes = dataset.read_nodes(nodes_file)
    edges_by_root = dataset.read_edges_by_root(edges_file)
    labels = dataset.read_labels(labels_file)

    # Construct subgraphs
    subgraphs = dataset.construct_subgraphs(nodes, edges_by_root)

    # Separate graphs into true and false categories
    true_graphs, false_graphs = dataset.separate_graphs_by_label(subgraphs, labels)

    # Print some information about the separated graphs
    print(f"Total number of true propagation graphs: {len(true_graphs)}")
    print(f"Total number of false propagation graphs: {len(false_graphs)}")

     # Perform NDTL Discovery
    discovered_formulas = ndtl.ndtl_discovery(true_graphs, false_graphs, 0.8)

    # Print discovered formulas
    print("True Graph Formulas:")
    for formula in discovered_formulas["non-rumor"]:
        print(formula)

    print("\nFalse Graph Formulas:")
    for formula in discovered_formulas["false"]:
        print(formula)



def tenFoldCrossValidation(discoveryTh=0.7,tau=0.2):
    # File paths
    nodes_file = "../../dataset/completi/rumor_detection_acl2017/twitter16/final/nodes.txt"
    edges_file = "../../dataset/completi/rumor_detection_acl2017/twitter16/final/edges.txt"
    labels_file = "../../dataset/completi/rumor_detection_acl2017/twitter16/filtered_label.txt"

    # Read nodes, edges, and labels
    nodes = dataset.read_nodes(nodes_file)
    edges_by_root = dataset.read_edges_by_root(edges_file)
    labels = dataset.read_labels(labels_file)

    # Construct subgraphs
    subgraphs = dataset.construct_subgraphs(nodes, edges_by_root)

    # Separate graphs into true and false categories
    true_graphs, false_graphs = dataset.separate_graphs_by_label(subgraphs, labels)

    # Print some information about the separated graphs
    total_news = len(true_graphs) + len(false_graphs)
    print(f"Total number of news: {total_news}")
    print(f"Total number of true propagation graphs: {len(true_graphs)}")
    print(f"Total number of false propagation graphs: {len(false_graphs)}")

    # Combine and shuffle graphs, adding labels to each entry
    true_graphs_labeled = [(root, graph, "non-rumor") for root, graph in true_graphs.items()]
    false_graphs_labeled = [(root, graph, "false") for root, graph in false_graphs.items()]

    all_graphs = true_graphs_labeled + false_graphs_labeled
    random.shuffle(all_graphs)

    # Split the dataset into 10 folds using KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    folds = list(kf.split(all_graphs))

    # Metrics to store results for each fold
    metrics_per_fold = []

    metrics_per_fold = []

    all_y_true_true = []
    all_y_probs0_true = []
    all_y_probs1_true = []
    all_y_probs2_true = []

    all_y_true_false = []
    all_y_probs0_false = []
    all_y_probs1_false = []
    all_y_probs2_false = []

    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        # Split into training and validation sets
        print("split ",fold_idx+1)
        train_data = [all_graphs[i] for i in train_indices]
        val_data = [all_graphs[i] for i in val_indices]

        # Separate training graphs and labels
        train_graphs = {root: graph for root, graph, label in train_data}
        train_labels = {root: label for root, graph, label in train_data}


        # Separate validation graphs and labels
        val_graphs = {root: graph for root, graph, label in val_data}
        val_labels = {root: label for root, graph, label in val_data}

        print("split ", fold_idx + 1, "separated train e validation")


        formulas = ndtl.ndtl_discovery(
            {k: v for k, v in train_graphs.items() if train_labels[k] == "non-rumor"},
            {k: v for k, v in train_graphs.items() if train_labels[k] == "false"},
            discoveryTh
        )


        print("split ", fold_idx + 1, "formulas discovered")

        # Evaluate on validation set
        predictions0 = {}
        predictions1 = {}
        predictions2 = {}
        for root, graph in val_graphs.items():
            #print("split ", fold_idx + 1, "evaluating graph with root", root, graph.number_of_nodes(), graph.number_of_edges())
            result = ndtl.evaluate_formulas(graph, root, formulas["false"], formulas["true"], tau)
            print(val_labels[root], result)
            predictions0[root] = result["classification0"]
            predictions1[root] = result["classification1"]
            predictions2[root] = result["classification2"]

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
        print(
            f"Total Predictions0 Fake: {metrics['classified_fake0']}, Total Predictions1 Fake: {metrics['classified_fake1']}, Total Predictions2 Fake: {metrics['classified_fake2']}, Total Fake News: {metrics['total_fake_news']}, Total News: {metrics['total_news']}")

        # Map predictions to binary labels for metric computation
        y_true = [1 if val_labels[root] == "non-rumor" else 0 for root in val_labels]
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

        metrics["total_true_news"] = sum(1 for label in val_labels.values() if label == "true")
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
    discoveryTh_range = [0.6, 0.7, 0.8, 0.9, 1.0]
    tau_range = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    output_dir = "simulation_results_real"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    filename = f"results_twitter16.csv"
    file_path = os.path.join(output_dir, filename)

    for discoveryTh, tau in itertools.product(discoveryTh_range, tau_range):
        avg_metrics = tenFoldCrossValidation(discoveryTh=discoveryTh, tau=tau)



        results.append({
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
            f"Completed: DiscoveryTh={discoveryTh}, Tau={tau}")

    df_all = pd.DataFrame(results)
    df_all.to_csv(file_path, index=False)
    print(f"\n All simulations completed! Results are stored in {output_dir}")
