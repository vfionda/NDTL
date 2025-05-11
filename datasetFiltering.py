import os
import shutil

def filter():
    # Define file paths
    label_file_path = '../../dataset/completi/rumor_detection_acl2017/twitter16/label.txt'
    source_tweets_file_path = '../../dataset/completi/rumor_detection_acl2017/twitter16/source_tweets.txt'
    tree_folder = '../../dataset/completi/rumor_detection_acl2017/twitter16/tree'
    filtered_tree_folder = '../../dataset/completi/rumor_detection_acl2017/twitter16/filtered_tree'

    # Create output folder if it doesn't exist
    os.makedirs(filtered_tree_folder, exist_ok=True)

    # Step 1: Read label.txt and filter for true/false rumors
    true_false_ids = set()
    filtered_labels = []

    with open(label_file_path, 'r') as label_file:
        for line in label_file:
            label, tweet_id = line.strip().split(':')
            if label in ('non-rumor', 'false'):
                true_false_ids.add(tweet_id)
                filtered_labels.append(line)

    # Step 2: Copy relevant id.txt files from tree folder to filtered_tree_folder
    for tweet_id in true_false_ids:
        src_file = os.path.join(tree_folder, f'{tweet_id}.txt')
        dest_file = os.path.join(filtered_tree_folder, f'{tweet_id}.txt')
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)

    # Step 3: Write filtered label.txt file with only true and false rumors
    filtered_label_file_path = '../../dataset/completi/rumor_detection_acl2017/twitter16/filtered_label.txt'
    with open(filtered_label_file_path, 'w') as filtered_label_file:
        filtered_label_file.writelines(filtered_labels)

    # Step 4: Filter source_tweets.txt to include only true and false rumor tweets
    filtered_source_tweets = []

    with open(source_tweets_file_path, 'r') as source_tweets_file:
        for line in source_tweets_file:
            tweet_id = line.split('\t')[0]
            if tweet_id in true_false_ids:
                filtered_source_tweets.append(line)

    # Write filtered source_tweets.txt file
    filtered_source_tweets_file_path = '../../dataset/completi/rumor_detection_acl2017/twitter16/filtered_source_tweets.txt'
    with open(filtered_source_tweets_file_path, 'w') as filtered_source_tweets_file:
        filtered_source_tweets_file.writelines(filtered_source_tweets)

    print("Filtering complete. Files saved:")
    print(f"- Filtered labels: {filtered_label_file_path}")
    print(f"- Filtered source tweets: {filtered_source_tweets_file_path}")
    print(f"- Filtered tree folder: {filtered_tree_folder}")


def classifyingAndStats(tree_folder='../../dataset/completi/rumor_detection_acl2017/twitter15/filtered_tree', label_file='../../dataset/completi/rumor_detection_acl2017/twitter16/label.txt'):
    # Define output files for nodes and edges
    nodes_file_path = '../../dataset/completi/rumor_detection_acl2017/twitter15/final/nodes.txt'
    edges_file_path = '../../dataset/completi/rumor_detection_acl2017/twitter15/final/edges.txt'

    # Load labels from label.txt
    labels = {}
    with open(label_file, 'r') as label_f:
        for line in label_f:
            label, source_id = line.strip().split(':')
            labels[source_id] = label

    # Initialize cumulative statistics and lists to store nodes and edges
    total_users = set()
    total_threads = 0
    total_time_length = 0
    true_count = 0
    false_count = 0
    total_tweet_count = 0
    total_retweet_count = 0
    trees_with_retweets = 0  # Counter for trees that contain retweet nodes

    nodes = []  # List to store nodes
    node_ids = []
    edges = []  # List to store edges

    def classify_nodes(tree_file_path):
        nonlocal total_threads, total_time_length, true_count, false_count
        nonlocal total_tweet_count, total_retweet_count, trees_with_retweets

        # Get the source tweet ID from filename and check its label
        source_id = os.path.splitext(os.path.basename(tree_file_path))[0]
        label = labels.get(source_id)

        # Track unique users and posts in each tree
        users_in_tree = set()
        post_times = []
        retweet_count = 0
        tweet_count = 0
        actual_root = None  # To store the actual root tweet ID

        with open(tree_file_path, 'r') as file:
            for line in file:

                parent, child = line.strip().split('->')
                parent_info = eval(parent)  # Convert string to list
                child_info = eval(child)    # Convert string to list

                parent_uid, parent_tweet_id, parent_time_from_root = parent_info
                child_uid, child_tweet_id, child_time_from_root = child_info
                time_from_parent = float(child_time_from_root) - float(parent_time_from_root)

               # if (parent_uid=='239672340' and parent_tweet_id=='514513238613303296' or child_uid=='239672340' and child_tweet_id=='514513238613303296'):
                #    print(tree_file_path, line)

                # Identify the actual root tweet
                if parent_info[0] == 'ROOT' and actual_root is None:
                    actual_root = child_info  # First child of ROOT is the actual root
                    node_type = "Root"
                elif parent_info == actual_root and parent_tweet_id == child_tweet_id:
                    # Nodes directly connected to the actual root are labeled as "Tweet"
                    node_type = "Tweet"
                    tweet_count += 1
                elif parent_tweet_id == child_tweet_id:
                    # Nodes further connected by retweets are labeled as "Retweet"
                    node_type = "Retweet"
                    retweet_count += 1
                elif (not actual_root is None) and child_tweet_id == actual_root[1]:
                    node_type = "Retweet"
                    retweet_count += 1
                else:
                    node_type = "Reply"
                    #continue


                if (time_from_parent>=0 and (f"{parent_uid}_{parent_tweet_id}" in node_ids or node_type == "Root")):
                    # Add node and edge to lists
                    node_id = f"{child_uid}_{child_tweet_id}"  # Unique node identifier
                    node_ids.append(node_id)
                    nodes.append(f"{node_id} [{child_uid}, {child_tweet_id}, {child_time_from_root}, {time_from_parent}, '{node_type}']\n")
                    if ((parent_uid != child_uid) or (parent_tweet_id != child_tweet_id)) and not(f"{child_uid}_{child_tweet_id}, {parent_uid}_{parent_tweet_id}\n" in edges):
                        edge = f"{parent_uid}_{parent_tweet_id}, {child_uid}_{child_tweet_id}\n"
                        edges.append(edge)

                    # Update statistics for root and retweet nodes only
                    users_in_tree.add(child_uid)
                    post_times.append(float(child_time_from_root))

        # If the tree has at least one retweet, increment the trees_with_retweets counter
        if retweet_count > 0:
            trees_with_retweets += 1

        # Update cumulative statistics if there are tweets or retweets in the current tree
        if retweet_count > 0 or tweet_count > 0:
            total_users.update(users_in_tree)
            total_threads += 1
            total_time_length += max(post_times) if post_times else 0
            total_retweet_count += retweet_count
            total_tweet_count += tweet_count

            # Update true/false count based on the label
            if label == "non-rumor":
                true_count += 1
            elif label == "false":
                false_count += 1

    # Process all files in the tree folder
    for filename in os.listdir(tree_folder):
        if filename.endswith('.txt'):
            tree_file_path = os.path.join(tree_folder, filename)
            classify_nodes(tree_file_path)

    # Write nodes and edges to their respective files
    with open(nodes_file_path, 'w') as nodes_file:
        nodes_file.writelines(nodes)
    with open(edges_file_path, 'w') as edges_file:
        edges_file.writelines(edges)

    # Print cumulative statistics
    print("Cumulative Statistics Across All Trees:")
    print(f"- # of different users involved: {len(total_users)}")
    print(f"- # of threads with root and retweets: {total_threads}")
    print(f"- Avg. time length over trees: {total_time_length / total_threads if total_threads else 0:.2f} minutes")
    print(f"- # of true threads: {true_count}")
    print(f"- # of false threads: {false_count}")
    print(f"- # of trees containing retweet nodes: {trees_with_retweets}")
    print("\nTweet and Retweet Statistics:")
    print(f"- Avg. # of tweet nodes per tree: {total_tweet_count / total_threads if total_threads else 0:.2f}")
    print(f"- Avg. # of retweet nodes per tree: {total_retweet_count / total_threads if total_threads else 0:.2f}")

# Call the function to execute
#filter()
classifyingAndStats()

