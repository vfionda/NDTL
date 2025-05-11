import networkx as nx
import ast

def read_nodes(file_path):
    """
    Reads the nodes file and returns a dictionary of all nodes with their attributes.
    """
    nodes = {}
    with open(file_path, 'r') as f:
        for line in f:
            #print (line)
            parts = line.strip().split(maxsplit=1)
            node_id = parts[0]
            attributes = ast.literal_eval(parts[1])  # Safely parse the attributes list
            nodes[node_id] = {
                "user_id": attributes[0],
                "tweet_id": attributes[1],
                "t": attributes[2],
                "delta": attributes[3],
                "type": attributes[4]
            }

    return nodes

def read_edges_by_root(file_path):
    """
    Reads the edges file and groups edges by Root node.
    Each Root corresponds to a separate propagation graph.
    """
    subgraphs_edges = {}
    current_root = None

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(", ")
            source = parts[0]
            target = parts[1]

            # Detect new Root node
            if "ROOT" in source:
                current_root = target
                if current_root not in subgraphs_edges:
                    subgraphs_edges[current_root] = []
            # Add edges to the current Root's subgraph
            elif current_root is not None:
                subgraphs_edges[current_root].append((source, target))

    return subgraphs_edges

def read_labels(file_path):
    """
    Reads the filtered_label.txt file and returns a dictionary mapping tweet IDs to labels.
    """
    labels = {}
    with open(file_path, 'r') as f:
        for line in f:
            label, tweet_id = line.strip().split(":")
            labels[tweet_id] = label  # "true" or "false"
    return labels


def construct_subgraphs(nodes, edges_by_root):
    """
    Constructs subgraphs for each Root node using edges and node attributes.
    """
    subgraphs = {}

    for root, edges in edges_by_root.items():
        # Create a new directed graph for this Root
        G = nx.DiGraph()

        # Add edges
        G.add_edges_from(edges)

        # Add node attributes
        for node in G.nodes:
            if node in nodes:
                G.nodes[node].update(nodes[node])

        subgraphs[root] = G

    return subgraphs

def separate_graphs_by_label(subgraphs, labels):
    """
    Separates subgraphs into 'true' and 'false' categories based on labels.
    """
    true_graphs = {}
    false_graphs = {}

    for root, graph in subgraphs.items():
        # Get the tweet ID of the Root node
        root_node = root
        tweet_id = graph.nodes[root_node]["tweet_id"]

        # Check the label of the tweet ID
        if str(tweet_id) in labels:
            label = labels[str(tweet_id)]
            if label == "non-rumor":
                true_graphs[root] = graph
            elif label == "false":
                false_graphs[root] = graph

    return true_graphs, false_graphs

