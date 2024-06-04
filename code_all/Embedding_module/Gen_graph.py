import dgl
import torch
import torch.nn as nn
import pandas as pd
import os

from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_map(path, node1_name, node2_name):
    df = pd.read_csv(path)
    # ques2skill = torch.tensor(ques2skill_df['skill']).view(-1, 1)
    # Create an empty dictionary to store the index list of ques->skill
    index_dict = {}
    # 遍历数据框的每一行
    for index, row in df.iterrows():
        node1 = row[node1_name]
        node2 = row[node2_name]
        # Check if there is already a list of indices for ques in the dictionary, if not, create an empty list
        if node1 not in index_dict:
            index_dict[node1] = []
        # Add the current skill to the index list of ques
        index_dict[node1].append(node2)
    # Find the minimum and maximum ques values, then generate a continuous range of ques
    min_node1 = min(index_dict.keys())
    max_node1 = max(index_dict.keys())
    max_node2 = max(index_dict.values())
    all_ques_range = range(min_node1, max_node1 + 1)
    # Create a two-dimensional list to store the index list
    index_list = [torch.tensor(index_dict.get(ques, [])) for ques in all_ques_range]
    index_list = pad_sequence(index_list, batch_first=True, padding_value=max_node2[0])
    return index_list


def load_Graph(dataset, embed_dim):
    # Get the mapping of ques -> skills (1 to n)
    # ques2skill_df = pd.read_csv('../data/{0}/graph/ques_skill.csv'.format(dataset))
    # ques2skill = torch.tensor(ques2skill_df['skill']).view(-1, 1)
    ques2skill = load_map('../data_all/{0}/graph/ques_skill.csv'.format(dataset), 'ques', 'skill')

    ques2diff_df = pd.read_csv('../data_all/{0}/graph/ques_diff.csv'.format(dataset))
    ques2diff = torch.tensor(ques2diff_df['diff']).view(-1, 1)

    # Get the mapping of ques -> area (1 to 1)
    ques2area_df = pd.read_csv('../data_all/{0}/graph/ques_area.csv'.format(dataset))
    ques2area = torch.tensor(ques2area_df['area']).view(-1, 1)

    if dataset == 'JunYi':
        # Get the mapping of ques -> pre (n to n)
        pre2post = load_map('../data_all/{0}/graph/ques_pre.csv'.format(dataset), 'pre', 'post')
        post2pre = load_map('../data_all/{0}/graph/ques_pre.csv'.format(dataset), 'post', 'pre')

    folder_path = '../data_all/{0}/graph'.format(dataset)
    # Get all files in a folder
    files = os.listdir(folder_path)
    # Filter out all files with the extension '.csv'
    csv_files = [file for file in files if file.endswith('.csv')]
    # Extract file name (excluding suffix)
    file_names = [os.path.splitext(file)[0] for file in csv_files]
    print(csv_files)

    graph_dict = {}
    for csv_file in file_names:
        node1_name = csv_file.split('_')[0]
        node2_name = csv_file.split('_')[1]
        file_path = os.path.join(folder_path, csv_file + ".csv")
        df = pd.read_csv(file_path)
        if csv_file == 'ques_pre':
            graph_dict[('ques', csv_file, 'ques')] = (df['pre'], df['post'])
            graph_dict[('ques', node2_name + '_' + node1_name, 'ques')] = (df['post'], df['pre'])
        else:
            graph_dict[(node1_name, csv_file, node2_name)] = (df[node1_name], df[node2_name])
            graph_dict[(node2_name, node2_name + '_' + node1_name, node1_name)] = (df[node2_name], df[node1_name])

    # Get student test data
    question_path = '../data_all/{0}/train_test/train_all_feature.txt'.format(dataset)
    f = open(question_path)
    stu_id = 0
    stu_ids = []
    question_ids = []
    line = f.read().split('\n')
    for count in range(0, len(line), 6):
        print(count)
        length = int(line[count])
        question_ids += list(map(lambda x: int(x), line[count + 1].split(',')))
        stu_ids += [stu_id for i in range(length)]
        stu_id += 1
    graph_dict[('stu', 'stu_ques', 'ques')] = (stu_ids, question_ids)
    graph_dict[('ques', 'ques_stu', 'stu')] = (question_ids, stu_ids)

    # Create a DGL heterogeneous graph
    hetero_graph = dgl.heterograph(graph_dict)
    print(hetero_graph)

    # Get the number of nodes
    num_nodes = {node_type: hetero_graph.number_of_nodes(node_type) for node_type in hetero_graph.ntypes}
    # Randomly initialize node features
    for node_type, num in num_nodes.items():
        # Map all type nodes to the same dimension and initialize feature values ​​randomly
        features = nn.Parameter(torch.relu(torch.rand(num, embed_dim)))
        # Set features to nodes in the graph
        hetero_graph.nodes[node_type].data['h'] = features
    if dataset == 'JunYi':
        return (hetero_graph.to('cuda'), ques2skill.to('cuda'), ques2diff.to('cuda'),
                ques2area.to('cuda'), pre2post.to('cuda'), post2pre.to('cuda'))
    else:
        return (hetero_graph.to('cuda'), ques2skill.to('cuda'), ques2diff.to('cuda'),
                ques2area.to('cuda'))
