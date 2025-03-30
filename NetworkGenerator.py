import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt


class NewsDiffusionNetworkGenerator:
    def __init__(self, is_fake=True, seed=None, min_max_depth=5, max_max_depth=8, min_branching_factor=3, max_branching_factor=6,  time_scale=0.5, cascade_prob=0.8):
        """
        Parameters:
            is_fake (bool): If True, generate a diffusion network with characteristics of fake news.
                            If False, generate one with characteristics of true news.
            seed (int): Random seed for reproducibility.
        """
        self.is_fake = is_fake

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Fake news: deeper cascades (higher max_depth, higher cascade_prob), wider (higher branching factor), faster bursts (lower time_scale


        self.max_depth = random.randint(min_max_depth, max_max_depth)
        self.branching_factor = (min_branching_factor, max_branching_factor)
        self.time_scale = time_scale
        self.cascade_prob = cascade_prob

        # Graph will be a directed graph with node attributes:
        #   'type': "root", "tweet", or "retweet"
        #   't': absolute time of event
        #   'delta': time elapsed since parent event (for source, delta=0)
        #   'sigma': time elapsed since source (cumulative time)
        self.graph = nx.DiGraph()
        self.node_id = 0

    def _add_node(self, node_id, event_type, t, delta, sigma):
        """Helper to add a node with the required attributes."""
        if node_id is None:
            self.graph.add_node(self.node_id,type=event_type,t=t,delta=delta,sigma=sigma)
            current_id = self.node_id
            self.node_id += 1
            return current_id
        else:
            self.graph.add_node(node_id, type=event_type, t=t, delta=delta, sigma=sigma)
            current_id = node_id
            self.node_id += node_id+1
            return current_id

    def _simulate_branch(self, parent_id, current_depth, parent_sigma, parent_time):
        """Recursively simulate a branch (cascade) from a given parent node."""
        if current_depth >= self.max_depth:
            return

        # Decide how many children (retweets) to generate
        min_br, max_br = self.branching_factor
        num_children = random.randint(min_br, max_br)

        for _ in range(num_children):
            # With some probability, a branch might not continue (simulate burstiness)
            if random.random() > self.cascade_prob:
                continue

            # Generate time difference delta: using an exponential distribution scaled by time_scale.
            delta = np.random.exponential(scale=self.time_scale)
            child_time = parent_time + delta
            child_sigma = parent_sigma + delta

            # Decide the type of event: if parent is root, then child is tweet; if parent is tweet, then child is retweet.
            parent_type = self.graph.nodes[parent_id]['type']
            if parent_type in ["Root"]:
                child_type = "Tweet"
            else:
                child_type = "Retweet"

            child_id = self._add_node(None, child_type, child_time, delta, child_sigma)
            self.graph.add_edge(parent_id, child_id)

            # Recursively simulate further retweets from this child node.
            self._simulate_branch(child_id, current_depth + 1, child_sigma, child_time)

    def generate_network(self, root):
        """Generates and returns a synthetic news diffusion network as a NetworkX DiGraph."""
        # Create the source node (news post) at time 0.
        source_id = self._add_node(root,"Root", t=0.0, delta=0.0, sigma=0.0)

        # Start a number of initial cascades (tweets) from the news post.
        num_cascades = random.randint(3, 5) if self.is_fake else random.randint(2, 4)
        for _ in range(num_cascades):
            # For each cascade, simulate a tweet from the source
            delta = np.random.exponential(scale=self.time_scale)
            tweet_time = delta  # time since source
            tweet_sigma = delta  # cumulative time from source
            tweet_id = self._add_node(None,"Tweet", tweet_time, delta, tweet_sigma)
            self.graph.add_edge(source_id, tweet_id)
            # Simulate the cascade from this tweet
            self._simulate_branch(tweet_id, current_depth=1, parent_sigma=tweet_sigma, parent_time=tweet_time)

        return (source_id, self.graph)




def draw_diffusion_graph(G, title="Diffusion Network"):
    pos = nx.spring_layout(G, seed=42)
    node_types = nx.get_node_attributes(G, 'type')
    color_map = {"Root": "red", "Tweet": "blue", "Retweet": "green"}
    colors = [color_map.get(node_types[n], "black") for n in G.nodes]
    labels = {n: f"{node_types[n]} \n t={G.nodes[n]['t']:.2f}" for n in G.nodes}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, node_size=800, font_size=8)
    plt.title(title)
    plt.show()

def generate_diffusion_graphs (n,fake=True,min_max_depth=5, max_max_depth=8, min_branching_factor=3, max_branching_factor=6, time_scale=0.5, cascade_prob=0.8):
    graphs = []
    if fake:
        for i in range(n):
            generator=NewsDiffusionNetworkGenerator(fake, seed=i, min_max_depth=min_max_depth, max_max_depth=max_max_depth, min_branching_factor=min_branching_factor, max_branching_factor=max_branching_factor, time_scale=time_scale, cascade_prob=cascade_prob)
            (i,graph)=generator.generate_network(i)
            graphs.append((i,graph,"false"))
    else:
        for i in range(n,2*n):
            generator=NewsDiffusionNetworkGenerator(fake, seed=i, min_max_depth=min_max_depth, max_max_depth=max_max_depth, min_branching_factor=min_branching_factor, max_branching_factor=max_branching_factor, time_scale=time_scale, cascade_prob=cascade_prob)
            (i,graph)=generator.generate_network(i)
            graphs.append((i,graph,"true"))

    return graphs
