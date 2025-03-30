import numpy as np
import networkx as nx
from lark import Transformer, Lark
from ndtl_parser import NDTLTransformer
from pathlib import Path
from scipy.stats import gaussian_kde

def calculate_metrics(graph):

    #Calculates temporal metrics for a single propagation graph using its structure.
    #Returns a dictionary of metrics.

    metrics = {}

    for n, d in graph.nodes(data=True):
        if "type" not in d:
            print(f"Node {n} is missing 'type' attribute. Data: {d}")

    # Collect timestamps by type
    retweet_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "Retweet"]
    tweet_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "Tweet"]

    retweet_times = [graph.nodes[n]["t"] for n in retweet_nodes]
    tweet_times = [graph.nodes[n]["t"] for n in tweet_nodes]

    # 1. Average time interval between consecutive retweets (t1)
    retweet_intervals = []
    for node in retweet_nodes:
        neighbors = list(graph.successors(node))
        for neighbor in neighbors:
            if graph.nodes[neighbor]["type"] == "Retweet":
                retweet_intervals.append(graph.nodes[neighbor]["delta"])
    metrics["t1"] = np.mean(retweet_intervals) if retweet_intervals else None


    # 2. Find the Tweet node with the minimum timestamp and retweet with the largest timestamp (t2)
    if not tweet_times:
        metrics["t2"] = None  # No Tweet nodes in the graph
    else:
        earliest_tweet_time = min(tweet_times)

        if not retweet_times:
            metrics["t2"] = None  # No Retweet nodes in the graph
        else:
            latest_retweet_time = max(retweet_times)
            metrics["t2"] = latest_retweet_time - earliest_tweet_time

    # 3. Time span between earliest and latest tweets (t3)
    if tweet_times:
        metrics["t3"] = max(tweet_times) - min(tweet_times)
    else:
        metrics["t3"] = None

    # 4. Time difference between a tweet and its first retweet (t4)
    tweet_to_first_retweet_times = []
    for node in tweet_nodes:
        neighbors = list(graph.successors(node))
        retweet_times_inner = [
            graph.nodes[neighbor]["delta"]
            for neighbor in neighbors
            if graph.nodes[neighbor]["type"] == "Retweet"
        ]
        if retweet_times_inner:
            tweet_to_first_retweet_times.append(min(retweet_times_inner))
    metrics["t4"] = min(tweet_to_first_retweet_times) if tweet_to_first_retweet_times else None

    # 5. Time difference between a tweet and its last retweet (t5)
    tweet_to_last_retweet_times = []
    for node in tweet_nodes:
        # Get all descendants (subtree) of the current Tweet node
        subtree_nodes = nx.descendants(graph, node)

        # Filter only Retweet nodes in the subtree and collect their timestamps
        retweet_times_in = [
            graph.nodes[descendant]["delta"]
            for descendant in subtree_nodes
            if graph.nodes[descendant]["type"] == "Retweet"
        ]

        # If there are any Retweet nodes in the subtree, compute the time difference
        if retweet_times_in:
            tweet_to_last_retweet_times.append(max(retweet_times_in))

    # Compute the minimum time difference across all Tweet nodes
    metrics["t5"] = min(tweet_to_last_retweet_times) if tweet_to_last_retweet_times else None

    # Compute the maximum time of all Tweet nodes
    metrics["t6"] = max(tweet_times) if tweet_times else None

    # Compute the maximum time of all Retweet nodes
    metrics["t7"] = max(retweet_times) if retweet_times else None

    # Compute the difference between the maximum time of all Retweet nodes and the minimum time of all Retweet nodes
    metrics["t8"] = max(retweet_times)-min(retweet_times) if retweet_times else None

    return metrics

# def calculate_graph_metrics_distribution(graph):
#
#     #Calculates temporal metrics distributionsa for a single propagation graph using its structure.
#     #Returns a dictionary of metrics.
#
#     metrics = {}
#
#     for n, d in graph.nodes(data=True):
#         if "type" not in d:
#             print(f"Node {n} is missing 'type' attribute. Data: {d}")
#
#     # Collect timestamps by type
#     retweet_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "Retweet"]
#     tweet_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "Tweet"]
#
#     retweet_times = [graph.nodes[n]["t"] for n in retweet_nodes]
#     tweet_times = [graph.nodes[n]["t"] for n in tweet_nodes]
#
#     # 1. Average time interval between consecutive retweets (t1)
#     retweet_intervals = []
#     for node in retweet_nodes:
#         neighbors = list(graph.successors(node))
#         for neighbor in neighbors:
#             if graph.nodes[neighbor]["type"] == "Retweet":
#                 retweet_intervals.append(graph.nodes[neighbor]["delta"])
#     metrics["t1"] = retweet_intervals
#
#
#     # 2. Find the Tweet node with the minimum timestamp and retweet with the largest timestamp (t2)
#     if not tweet_times:
#         metrics["t2"] = None  # No Tweet nodes in the graph
#     else:
#         earliest_tweet_time = min(tweet_times)
#
#         if not retweet_times:
#             metrics["t2"] = None  # No Retweet nodes in the graph
#         else:
#             latest_retweet_time = max(retweet_times)
#             metrics["t2"] = latest_retweet_time - earliest_tweet_time
#
#     # 3. Time span between earliest and latest tweets (t3)
#     if tweet_times:
#         metrics["t3"] = max(tweet_times) - min(tweet_times)
#     else:
#         metrics["t3"] = None
#
#     # 4. Time difference between a tweet and its first retweet (t4)
#     tweet_to_first_retweet_times = []
#     for node in tweet_nodes:
#         neighbors = list(graph.successors(node))
#         retweet_times_inner = [
#             graph.nodes[neighbor]["delta"]
#             for neighbor in neighbors
#             if graph.nodes[neighbor]["type"] == "Retweet"
#         ]
#         if retweet_times_inner:
#             tweet_to_first_retweet_times.append(min(retweet_times_inner))
#     metrics["t4"] = tweet_to_first_retweet_times
#
#     # 5. Time difference between a tweet and its last retweet (t5)
#     tweet_to_last_retweet_times = []
#     for node in tweet_nodes:
#         # Get all descendants (subtree) of the current Tweet node
#         subtree_nodes = nx.descendants(graph, node)
#
#         # Filter only Retweet nodes in the subtree and collect their timestamps
#         retweet_times_in = [
#             graph.nodes[descendant]["delta"]
#             for descendant in subtree_nodes
#             if graph.nodes[descendant]["type"] == "Retweet"
#         ]
#
#         # If there are any Retweet nodes in the subtree, compute the time difference
#         if retweet_times_in:
#             tweet_to_last_retweet_times.append(max(retweet_times_in))
#
#     # Compute the minimum time difference across all Tweet nodes
#     metrics["t5"] = tweet_to_last_retweet_times
#
#     # Compute the maximum time of all Tweet nodes
#     #metrics["t6"] = max(tweet_times) if tweet_times else None
#
#     # Compute the maximum time of all Retweet nodes
#     #metrics["t7"] = max(retweet_times) if retweet_times else None
#
#     # Compute the difference between the maximum time of all Retweet nodes and the minimum time of all Retweet nodes
#     #metrics["t8"] = max(retweet_times)-min(retweet_times) if retweet_times else None
#
#     return metrics

# def compute_distributions(true_graphs, false_graphs, percentage=0.95):
#
#     metrics_true = {f"t{i}": [] for i in range(1, 6)}
#     metrics_false = {f"t{i}": [] for i in range(1, 6)}
#
#     # Compute metrics for true graphs
#     for graph in true_graphs.values():
#         metrics = calculate_graph_metrics_distribution(graph)
#         for key, value in metrics.items():
#             if value is not None:
#                 if isinstance(value, list):
#                     metrics_true[key].extend(value)
#                 else:
#                     metrics_true[key].append(value)
#                 #metrics_true["t6"].append(value)
#                 #metrics_true["t7"].append(value)
#                 #metrics_true["t8"].append(value)
#
#     # Compute metrics for false graphs
#     for graph in false_graphs.values():
#         metrics = calculate_graph_metrics_distribution(graph)
#         for key, value in metrics.items():
#             if value is not None:
#                 if value is not None:
#                     if isinstance(value, list):
#                         metrics_false[key].extend(value)
#                     else:
#                         metrics_false[key].append(value)
#                 #metrics_false["t6"].append(value)
#                 #metrics_false["t7"].append(value)
#                 #metrics_false["t8"].append(value)
#
#     distribution_true = {}
#     distribution_false = {}
#
#     for key in metrics_true.keys():
#         # Sort metrics for true and false graphs
#         sorted_true = sorted(metrics_true[key])
#         sorted_false = sorted(metrics_false[key])
#
#         # Compute lower and upper bounds for true graphs
#         lower_bound_true = int(len(sorted_true) * (1 - percentage))
#         upper_bound_true = int(len(sorted_true) * percentage) - 1
#
#         # Compute lower and upper bounds for false graphs
#         lower_bound_false = int(len(sorted_false) * (1 - percentage))
#         upper_bound_false = int(len(sorted_false) * percentage) - 1
#
#         distribution_true[key] = np.mean(sorted_true[lower_bound_true:upper_bound_true]), np.std(sorted_true[lower_bound_true:upper_bound_true])
#         distribution_false[key] = np.mean(sorted_false[lower_bound_false:upper_bound_false]), np.std(sorted_false[lower_bound_false:upper_bound_false])
#
#     # Calculate distributions
#
#     for key in metrics_true.keys():
#         #arr = np.array(metrics_true[key])
#         #distribution_true[key] = np.mean(np.log(arr+1)), np.std(np.log(arr+1))
#         distribution_true[key] = np.mean(metrics_true[key]), np.std(metrics_true[key])
#         #arr = np.array(metrics_false[key])
#         #distribution_false[key] = np.mean(np.log(arr+1)), np.std(np.log(arr+1))
#         distribution_false[key] = np.mean(metrics_false[key]), np.std(metrics_false[key])
#
#     return distribution_true, distribution_false

def compute_thresholds(true_graphs, false_graphs, percentage=1):
    """
    Computes thresholds (t1, t2, ..., t5) for NDTL formulas.
    The thresholds ensure the formulas hold for a fixed percentage of true and false graphs.

    :param true_graphs: Dictionary of true propagation graphs
    :param false_graphs: Dictionary of false propagation graphs
    :param percentage: The fraction of graphs (e.g., 0.8 for 80%) the thresholds should hold for
    :return: A dictionary of thresholds for true and false graphs
    """
    metrics_true = {f"t{i}": [] for i in range(1, 6)}
    metrics_false = {f"t{i}": [] for i in range(1, 6)}

    # Compute metrics for true graphs
    for graph in true_graphs.values():
        metrics = calculate_metrics(graph)
        for key, value in metrics.items():
            if value is not None and key in metrics_true:
                metrics_true[key].append(value)

    # Compute metrics for false graphs
    for graph in false_graphs.values():
        metrics = calculate_metrics(graph)
        for key, value in metrics.items():
            if value is not None and key in metrics_false:
                metrics_false[key].append(value)


    # Calculate thresholds
    thresholds = {}
    for key in metrics_true.keys():
        # Sort metrics for true and false graphs
        sorted_true = sorted(metrics_true[key])
        sorted_false = sorted(metrics_false[key])

        # Compute lower and upper bounds for true graphs
        lower_bound_true = int(len(sorted_true) * (1 - percentage))
        upper_bound_true = int(len(sorted_true) * percentage) - 1

        # Compute lower and upper bounds for false graphs
        lower_bound_false = int(len(sorted_false) * (1 - percentage))
        upper_bound_false = int(len(sorted_false) * percentage) - 1

        max_true = sorted_true[upper_bound_true] if len(sorted_true) > 0 else None
        min_true = sorted_true[lower_bound_true] if len(sorted_true) > 0 else None

        max_false = sorted_false[upper_bound_false] if len(sorted_false) > 0 else None
        min_false = sorted_false[lower_bound_false] if len(sorted_false) > 0 else None

        mean_true = (
            np.mean(sorted_true[lower_bound_true:upper_bound_true])
            if len(sorted_true) > 0
            else None
        )
        mean_false = (
            np.mean(sorted_false[lower_bound_false:upper_bound_false])
            if len(sorted_false) > 0
            else None
        )

        print(key, 'true', min_true, max_true, mean_true)
        print(key, 'false', min_false, max_false, mean_false)

        thresholds[key]=[]

        # Check compatibility of true and false thresholds
        #if max_true is not None and min_false is not None and max_true <= min_false:
        #    thresholds[key].append((max_true, min_false))
        #elif max_false is not None and min_true is not None and max_false <= min_true:
        #    thresholds[key].append((max_false, min_true))
        #else:
        if min_true is not None and min_false is not None and min_true<min_false:
            thresholds[key].append (("true","min",min_false))
        if max_true is not None and max_false is not None and max_true > max_false:
            thresholds[key].append(("true", "max", max_false))
        if min_true is not None and min_false is not None and min_false < min_true:
            thresholds[key].append(("false", "min", min_true))
        if max_true is not None and max_false is not None and max_false > max_true:
            thresholds[key].append(("false", "max", max_true))

    return thresholds

# def ndtl_discovery_distribution(true_graphs, false_graphs):
#     """
#     Discovers and instantiates NDTL formulas using thresholds.
#     Returns instantiated formulas for true and false graphs.
#     """
#     # Compute thresholds for the formulas
#     distributions = compute_distributions (true_graphs, false_graphs)
#
#     print(distributions[0])
#     print(distributions[1])
#
#     # Instantiate formulas for true and false graphs
#     formulas = {"true": [], "false": []}
#
#     for key, value in distributions[0].items():
#         if key == "t1":
#             formulas["true"].append(f"G(Retweet -> avgdelta(Gamma(Retweet, 1)) = {tuple(map(float, value))})")
#         elif key == "t2":
#             formulas["true"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) = {tuple(map(float, value))}))")
#         elif key == "t3":
#             formulas["true"].append(f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) = {tuple(map(float, value))}))")
#         elif key == "t4":
#             formulas["true"].append(f"G (Tweet -> mindelta(Gamma(Retweet, 1)) = {tuple(map(float, value))})")
#         elif key == "t5":
#             formulas["true"].append(f"G (Tweet -> maxdelta(Gamma(Retweet, inf)) = {tuple(map(float, value))})")
#         elif key == "t6":
#             formulas["true"].append(f"G (Root -> maxt(Gamma(Tweet, inf)) = {tuple(map(float, value))})")
#         elif key == "t7":
#             formulas["true"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) = {tuple(map(float, value))})")
#         elif key == "t8":
#             formulas["true"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) = {tuple(map(float, value))}))")
#
#     for key, value in distributions[1].items():
#         if key == "t1":
#             formulas["false"].append(f"G(Retweet -> avgdelta(Gamma(Retweet, 1)) = {tuple(map(float, value))})")
#         elif key == "t2":
#             formulas["false"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) = {tuple(map(float, value))}))")
#         elif key == "t3":
#             formulas["false"].append(f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) = {tuple(map(float, value))}))")
#         elif key == "t4":
#             formulas["false"].append(f"G (Tweet -> mindelta(Gamma(Retweet, 1)) = {tuple(map(float, value))})")
#         elif key == "t5":
#             formulas["false"].append(f"G (Tweet -> maxdelta(Gamma(Retweet, inf)) = {tuple(map(float, value))})")
#         elif key == "t6":
#             formulas["false"].append(f"G (Root -> maxt(Gamma(Tweet, inf)) = {tuple(map(float, value))})")
#         elif key == "t7":
#             formulas["false"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) = {tuple(map(float, value))})")
#         elif key == "t8":
#             formulas["false"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) = {tuple(map(float, value))}))")
#
#
#     print(formulas["true"])
#     print(formulas["false"])
#     return formulas

def ndtl_discovery(true_graphs, false_graphs, threshold=0.8):
    """
    Discovers and instantiates NDTL formulas using thresholds.
    Returns instantiated formulas for true and false graphs.
    """
    # Compute thresholds for the formulas
    thresholds = compute_thresholds(true_graphs, false_graphs, threshold)

    print(thresholds)

    # Instantiate formulas for true and false graphs
    formulas = {"true": [], "false": []}

    for key, value in thresholds.items():
        if len(value)>1 or (len(value)==1 and len(value[0])>2):
            for val in value:
                category, type, th = val
                if key == "t1":
                    if category=="true":
                        if type=="min":
                            formulas["true"].append(f"G0.8 (Retweet -> avgdelta(Gamma(Retweet, 1)) < {th})")
                        else:
                            formulas["true"].append(f"G0.8 (Retweet -> avgdelta(Gamma(Retweet, 1)) > {th})")
                    else:
                        if type == "min":
                            formulas["false"].append(f"G0.8 (Retweet -> avgdelta(Gamma(Retweet, 1)) < {th})")
                        else:
                            formulas["false"].append(f"G0.8 (Retweet -> avgdelta(Gamma(Retweet, 1)) > {th})")
                elif key == "t2":
                    if category=="true":
                        if type=="min":
                            formulas["true"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) < {th}))")
                        else:
                            formulas["true"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) > {th}))")
                    else:
                        if type == "min":
                            formulas["false"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) < {th}))")
                        else:
                            formulas["false"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) > {th}))")
                elif key == "t3":
                    if category=="true":
                        if type=="min":
                            formulas["true"].append(f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) < {th}))")
                        else:
                            formulas["true"].append(f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) > {th}))")
                    else:
                        if type == "min":
                            formulas["false"].append(f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) < {th}))")
                        else:
                            formulas["false"].append(f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) > {th}))")
                elif key == "t4":
                    if category=="true":
                        if type=="min":
                            formulas["true"].append(f"G0.8 (Tweet -> mindelta(Gamma(Retweet, 1)) < {th})")
                        else:
                            formulas["true"].append(f"G0.8 (Tweet -> mindelta(Gamma(Retweet, 1)) > {th})")
                    else:
                        if type == "min":
                            formulas["false"].append(f"G0.8 (Tweet -> mindelta(Gamma(Retweet, 1)) < {th})")
                        else:
                            formulas["false"].append(f"G0.8 (Tweet -> mindelta(Gamma(Retweet, 1)) > {th})")
                elif key == "t5":
                    if category=="true":
                        if type=="min":
                            formulas["true"].append(f"G0.8 (Tweet -> maxdelta(Gamma(Retweet, inf)) < {th})")
                        else:
                            formulas["true"].append(f"G0.8 (Tweet -> maxdelta(Gamma(Retweet, inf)) > {th})")
                    else:
                        if type == "min":
                            formulas["false"].append(f"G0.8 (Tweet -> maxdelta(Gamma(Retweet, inf)) < {th})")
                        else:
                            formulas["false"].append(f"G0.8 (Tweet -> maxdelta(Gamma(Retweet, inf)) > {th})")
                # elif key == "t6":
                #     if category=="true":
                #         if type=="min":
                #             formulas["true"].append(f"G (Root -> maxt(Gamma(Tweet, inf)) < {th})")
                #         else:
                #             formulas["true"].append(f"G (Root -> maxt(Gamma(Tweet, inf)) > {th})")
                #     else:
                #         if type == "min":
                #             formulas["false"].append(f"G (Root -> maxt(Gamma(Tweet, inf)) < {th})")
                #         else:
                #             formulas["false"].append(f"G (Root -> maxt(Gamma(Tweet, inf)) > {th})")
                # elif key == "t7":
                #     if category=="true":
                #         if type=="min":
                #             formulas["true"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) < {th})")
                #         else:
                #             formulas["true"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) > {th})")
                #     else:
                #         if type == "min":
                #             formulas["false"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) < {th})")
                #         else:
                #             formulas["false"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) > {th})")
                # elif key == "t8":
                #     if category=="true":
                #         if type=="min":
                #             formulas["true"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) < {th}))")
                #         else:
                #             formulas["true"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) > {th}))")
                #     else:
                #         if type == "min":
                #             formulas["false"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) < {th}))")
                #         else:
                #             formulas["false"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) > {th}))")
        # elif len(value)==1 and isinstance(value[0], tuple):
        #     # Compatible thresholds: Separate true and false
        #     true_threshold, false_threshold = value[0]
        #
        #     if key == "t1":
        #         if true_threshold is not None:
        #             formulas["true"].append(f"G0.8 (Retweet -> avgdelta(Gamma(Retweet, 1)) <= {true_threshold})")
        #         if false_threshold is not None:
        #             formulas["false"].append(f"G0.8 (Retweet -> avgdelta(Gamma(Retweet, 1)) > {false_threshold})")
        #
        #     elif key == "t2":
        #         if true_threshold is not None:
        #             formulas["true"].append(
        #                 f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) <= {true_threshold}))")
        #         if false_threshold is not None:
        #             formulas["false"].append(
        #                 f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) > {false_threshold}))")
        #
        #     elif key == "t3":
        #         if true_threshold is not None:
        #             formulas["true"].append(
        #                 f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) <= {true_threshold}))")
        #         if false_threshold is not None:
        #             formulas["false"].append(
        #                 f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) > {false_threshold}))")
        #
        #     elif key == "t4":
        #         if true_threshold is not None:
        #             formulas["true"].append(f"G0.8 (Tweet -> mindelta(Gamma(Retweet, 1)) <= {true_threshold})")
        #         if false_threshold is not None:
        #             formulas["false"].append(f"G0.8 (Tweet -> mindelta(Gamma(Retweet, 1)) > {false_threshold})")
        #
        #     elif key == "t5":
        #         if true_threshold is not None:
        #             formulas["true"].append(f"G0.8 (Tweet -> maxdelta(Gamma(Retweet, inf)) <= {true_threshold})")
        #         if false_threshold is not None:
        #             formulas["false"].append(f"G0.8 (Tweet -> maxdelta(Gamma(Retweet, inf)) > {false_threshold})")

            # elif key == "t6":
            #     if true_threshold is not None:
            #         formulas["true"].append(f"G (Root -> maxt(Gamma(Tweet, inf)) <= {true_threshold})")
            #     if false_threshold is not None:
            #         formulas["false"].append(f"G (Root -> maxt(Gamma(Tweet, inf)) > {false_threshold})")
            #
            # elif key == "t7":
            #     if true_threshold is not None:
            #         formulas["true"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) <= {true_threshold})")
            #     if false_threshold is not None:
            #         formulas["false"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) > {false_threshold})")
            # elif key == "t8":
            #     if true_threshold is not None:
            #         formulas["true"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) <= {true_threshold}))")
            #     if false_threshold is not None:
            #         formulas["false"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) > {false_threshold}))")

        # elif len(value)==1 and not isinstance(value[0], tuple):
        #     # Fallback: Single threshold
        #     if key == "t1" and value is not None:
        #         formulas["true"].append(f"G0.8 (Retweet -> avgdelta(Gamma(Retweet, 1)) <= {value[0]})")
        #         formulas["false"].append(f"G0.8 (Retweet -> avgdelta(Gamma(Retweet, 1)) > {value[0]})")
        #
        #     elif key == "t2" and value is not None:
        #         formulas["true"].append(
        #             f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) <= {value[0]}))")
        #         formulas["false"].append(
        #             f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Tweet, 1)) > {value[0]}))")
        #
        #     elif key == "t3" and value is not None:
        #         formulas["true"].append(
        #             f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) <= {value[0]}))")
        #         formulas["false"].append(
        #             f"G (Root -> (maxt(Gamma(Tweet, 1)) - mint(Gamma(Tweet, 1)) > {value[0]}))")
        #
        #     elif key == "t4" and value is not None:
        #         formulas["true"].append(f"G0.8 (Tweet -> mindelta(Gamma(Retweet, 1)) <= {value[0]})")
        #         formulas["false"].append(f"G0.8 (Tweet -> mindelta(Gamma(Retweet, 1)) > {value[0]})")
        #
        #     elif key == "t5" and value is not None:
        #         formulas["true"].append(f"G0.8 (Tweet -> maxdelta(Gamma(Retweet, inf)) <= {value[0]})")
        #         formulas["false"].append(f"G0.8 (Tweet -> maxdelta(Gamma(Retweet, inf)) > {value[0]})")
            # elif key == "t6":
            #     formulas["true"].append(f"G (Root -> maxt(Gamma(Tweet, 1)) <= {value[0]})")
            #     formulas["false"].append(f"G (Root -> maxt(Gamma(Tweet, 1)) > {value[0]})")
            # elif key == "t7":
            #     formulas["true"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) <= {value[0]})")
            #     formulas["false"].append(f"G (Root -> maxt(Gamma(Retweet, inf)) > {value[0]})")
            # elif key == "t8" and value is not None:
            #     formulas["true"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) <= {value[0]}))")
            #     formulas["false"].append(f"G (Root -> (maxt(Gamma(Retweet, inf)) - mint(Gamma(Retweet, inf)) > {value[0]}))")

    print(formulas["true"])
    print(formulas["false"])
    return formulas

# def evaluate_formulas_distribution(graph, root, fake_formulas, true_formulas, tau=0.4):
#
#     ndtl_parser = Lark(
#         Path("ndtl_grammar.lark").open("r").read(), parser="lalr", start="start"
#     )
#
#     transformer = NDTLTransformer()
#
#
#     false_values = [transformer.transform(ndtl_parser.parse(formula)).evaluate(graph, root) for formula in fake_formulas]
#     array_fake = [1 if transformer.transform(ndtl_parser.parse(formula)).evaluate(graph,root) else 0 for formula in fake_formulas]
#
#     satisfied_fake = sum(value for value in array_fake)
#     P_f = satisfied_fake / len(fake_formulas) if fake_formulas else 0.0
#
#
#     # Evaluate true formulas (Φt)
#
#
#     true_values=[transformer.transform(ndtl_parser.parse(formula)).evaluate(graph, root) for formula in true_formulas]
#     array_true = [
#         1 if result else 0
#         for formula in true_formulas
#         if (result := transformer.transform(ndtl_parser.parse(formula)).evaluate(graph, root)) != "nan"
#     ]
#
#
#     satisfied_true = sum(value for value in array_true)
#     P_t = satisfied_true / len(true_formulas) if true_formulas else 0.0
#
#     if (P_f-P_t>=tau):
#         classification = "Fake"
#     elif (P_t-P_f>=tau):
#         classification = "True"
#     else:
#         classification = "Uncertain"


    # Classify based on thresholds
    #if P_f >= tau_f_f and P_t <= tau_f_t:
    #    classification = "Fake"
    #elif P_t >= tau_t_t and P_f <= tau_t_f:
    #    classification = "True"
    #else:
    #    classification = "Uncertain"

    # return {
    #     "P_f": P_f,
    #     "satisfied_fake": false_values,
    #     "P_t": P_t,
    #     "satisfied_true": true_values,
    #     "classification": classification
    # }


def evaluate_formulas(graph, root, fake_formulas, true_formulas, tau=0.4, tau_f_f=0.6, tau_f_t=0.3, tau_t_t=0.6, tau_t_f=0.3):
    """
    Evaluates the likelihood that a given propagation graph corresponds to fake or true news.

    Parameters:
        graph (nx.DiGraph): The propagation graph of the news story.
        root (str): The root node (news post node) of the propagation graph.
        fake_formulas (list): List of NDTL formulas for fake news (Φf).
        true_formulas (list): List of NDTL formulas for true news (Φt).
        tau_h (float): Upper threshold for likelihood classification.
        tau_l (float): Lower threshold for likelihood classification.

    Returns:
        dict: A dictionary containing the likelihood scores (P^f, P^t) and the classification result.
    """

    ndtl_parser = Lark(
        Path("ndtl_grammar.lark").open("r").read(), parser="lalr", start="start"
    )

    transformer = NDTLTransformer()

    false_values = [transformer.transform(ndtl_parser.parse(formula)).evaluate(graph, root) for formula in fake_formulas]
    array_fake = [1 for n in false_values if n]

    #epsilon = 1e-6  # Small constant to avoid log(0)
    #log_scaled_fake = [np.log(1 + value + epsilon) for value in array_fake]
    #P_f = sum(log_scaled_fake) / len(log_scaled_fake) if log_scaled_fake else 0

    satisfied_fake = sum(value for value in array_fake)
    P_f = satisfied_fake / len(fake_formulas) if fake_formulas else 0.0


    true_values=[transformer.transform(ndtl_parser.parse(formula)).evaluate(graph, root) for formula in true_formulas]
    array_true = [1 for n in true_values if n]

    #log_scaled_true = [np.log(1 + value + epsilon) for value in array_true]
    #P_t = sum(log_scaled_true) / len(log_scaled_true) if log_scaled_true else 0


    satisfied_true = sum(value for value in array_true)
    P_t = satisfied_true / len(true_formulas) if true_formulas else 0.0


    if (P_f-P_t>=tau):
        classification1 = "Fake"
    elif (P_t-P_f>=tau):
        classification1 = "True"
    else:
        classification1 = "Uncertain"

    #if (P_f>=tau_f):
    #    classification = "Fake"
    #elif (P_t>=tau_t):
    #    classification = "True"
    #else:
    #    classification = "Uncertain"


    if P_f >= tau_f_f and P_t <= tau_f_t:
        classification2 = "Fake"
    elif P_t >= tau_t_t and P_f <= tau_t_f:
        classification2 = "True"
    else:
        classification2 = "Uncertain"

    if (P_f > P_t):
        classification0 = "Fake"
    elif (P_t > P_f):
        classification0 = "True"
    else:
        classification0 = "Uncertain"

    return {
        "P_f": P_f,
        "satisfied_fake": false_values,
        "P_t": P_t,
        "satisfied_true": true_values,
        "classification0": classification0,
        "classification1": classification1,
        "classification2": classification2

    }
