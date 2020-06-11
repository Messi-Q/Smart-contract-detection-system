import sys
import json
import numpy as np
from vec2onehot import vec2onehot
from AutoExtractGraph import generate_graph

"""
Node self property + Incoming Var + Outgoing Var + Incoming Edge + Outgoing Edge
"""

dict_NodeName = {"NULL": 0, "S0": 1, "W0": 2, 'VAR0': 3}

dict_VarOpName = {"NULL": 0, "BOOL": 1, "ASSIGN": 2}

dict_EdgeOpName = {"FW": 0, "IF": 1, "FOR": 2, "RE": 3, "AH": 4, "RG": 5}

dict_AC = {"NoLimit": 0, "LimitedAC": 1}

dict_NodeOpName = {"NULL": 0, "INNADD": 1, "MSG": 2}

dict_MarkName = {"compliance": 0, "warning": 1, "violation": 2}

node_convert = {"S0": 0, "W0": 1, "VAR0": 2}

v2o = vec2onehot()  # create the one-bot dicts


# extract the features of each node from input file #
def extract_node_features(nodeFile):
    nodeNum = 0
    node_list = []
    node_attribute_list = []

    f = open(nodeFile)
    lines = f.readlines()
    f.close()

    for line in lines:
        node = list(map(str, line.split()))
        verExist = False
        for i in range(0, len(node_list)):
            if node[0] == node_list[i]:
                verExist = True
            else:
                continue
        if verExist is False:
            node_list.append(node[0])
            nodeNum += 1
        node_attribute_list.append(node)

    return nodeNum, node_list, node_attribute_list


def embedding_node(node_attribute_list):
    # embedding each node after elimination #
    node_encode = []
    var_encode = []
    node_embedding = []
    var_embedding = []
    main_point = ['S0', 'S1', 'S2', 'S3', 'S4', 'W0', 'W1', 'W2', 'W3', 'W4']

    for j in range(0, len(node_attribute_list)):
        v = node_attribute_list[j][0]
        if v in main_point:
            vf0 = node_attribute_list[j][0]
            vf1 = dict_NodeName[node_attribute_list[j][0]]
            vfm1 = v2o.node2vecEmbedding(node_attribute_list[j][0])
            vf2 = dict_AC[node_attribute_list[j][1]]
            vfm2 = v2o.nodeAC2vecEmbedding(node_attribute_list[j][1])
            vf3 = dict_NodeName[node_attribute_list[j][2]]
            vfm3 = v2o.node2vecEmbedding(node_attribute_list[j][2])
            vf4 = int(node_attribute_list[j][3])
            vfm4 = v2o.sn2vecEmbedding(str(node_attribute_list[j][3]))
            nodeEmbedding = vfm1.tolist() + vfm2.tolist() + vfm3.tolist() + vfm4.tolist()
            node_embedding.append([vf0, np.array(nodeEmbedding)])
            temp = [vf1, vf2, vf3, vf4]
            node_encode.append([vf0, temp])
        else:
            vf0 = node_attribute_list[j][0]
            vf1 = dict_NodeName[node_attribute_list[j][0]]
            vfm1 = v2o.node2vecEmbedding(node_attribute_list[j][0])
            vf2 = dict_NodeName[node_attribute_list[j][1]]
            vfm2 = v2o.node2vecEmbedding(node_attribute_list[j][1])
            vf3 = int(node_attribute_list[j][2])
            vfm3 = v2o.sn2vecEmbedding(str(node_attribute_list[j][2]))
            vf4 = dict_MarkName[node_attribute_list[j][3]]
            vfm4 = v2o.mark2vecEmbedding(node_attribute_list[j][3])
            varEmbedding = vfm1.tolist() + vfm2.tolist() + vfm3.tolist() + vfm4.tolist()
            var_embedding.append([vf0, np.array(varEmbedding)])
            temp = [vf1, vf2, vf3, vf4]
            var_encode.append([vf0, temp])

    return node_encode, var_encode, node_embedding, var_embedding


def elimination_edge(edgeFile):
    # eliminate edge #
    edge_list = []  # all edge
    extra_edge_list = []  # eliminated edge

    f = open(edgeFile)
    lines = f.readlines()
    f.close()

    for line in lines:
        edge = list(map(str, line.split()))
        edge_list.append(edge)

    # The ablation of multiple edge between two nodes, taking the edge with the edge_operation priority
    for k in range(0, len(edge_list)):
        if k + 1 < len(edge_list):
            start1 = edge_list[k][0]  # start node
            end1 = edge_list[k][1]  # end node
            op1 = edge_list[k][3]
            start2 = edge_list[k + 1][0]
            end2 = edge_list[k + 1][1]
            op2 = edge_list[k + 1][3]
            if start1 == start2 and end1 == end2:
                op1_index = dict_EdgeOpName[op1]
                op2_index = dict_EdgeOpName[op2]
                # extract edge attribute based on priority
                if op1_index < op2_index:
                    extra_edge_list.append(edge_list.pop(k))
                else:
                    extra_edge_list.append(edge_list.pop(k + 1))

    return edge_list, extra_edge_list


def embedding_edge(edge_list):
    # extract & embedding the features of each edge from input file #
    edge_encode = []
    edge_embedding = []

    for k in range(len(edge_list)):
        start = edge_list[k][0]  # start node
        end = edge_list[k][1]  # end node
        a, b = edge_list[k][2], edge_list[k][3]

        ef1 = int(a)
        ef2 = dict_EdgeOpName[b]

        ef_temp = [ef1, ef2]
        edge_encode.append([start, end, ef_temp])

        efm1 = v2o.sn2vecEmbedding(str(a))
        efm2 = v2o.edgeOP2vecEmbedding(b)

        efm_temp = efm1.tolist() + efm2.tolist()
        edge_embedding.append([start, end, np.array(efm_temp)])

    return edge_encode, edge_embedding


def construct_vec(edge_list, node_embedding, var_embedding, edge_embedding, edge_encode):
    # Vec: core/var node + Incoming Edge + Outgoing Edge
    edge_vec_length = len(edge_embedding[0][2])
    edge_in_node = []
    edge_in = []
    edge_out_node = []
    edge_out = []
    node_vec = []
    var_point = ['VAR0']
    # node_embedding_dim_without_edge = 180

    for i in range(len(edge_embedding)):
        # The input/output edge vector of VAR0
        if edge_list[i][0] == "VAR0":
            edge_out.append([edge_embedding[i][0], edge_embedding[i][2]])
        elif edge_list[i][1] == "VAR0":
            edge_in.append([edge_embedding[i][1], edge_embedding[i][2]])

    for i in range(len(edge_in)):
        edge_in_node.append(edge_in[i][0])
    for i in range(len(edge_out)):
        edge_out_node.append(edge_out[i][0])

    for i in range(len(var_point)):
        if var_point[i] not in edge_out_node:
            edge_out.append([var_point[i], np.zeros(edge_vec_length, dtype=int)])
        if var_point[i] not in edge_in_node:
            edge_in.append([var_point[i], np.zeros(edge_vec_length, dtype=int)])

    edgeIn_dict = dict(edge_in)
    edgeOut_dict = dict(edge_out)
    var_dict = dict(var_embedding)

    for i in range(len(var_embedding)):
        if var_embedding[i][0] == 'VAR0':
            var_feature_0 = var_dict[var_embedding[i][0]].tolist() + np.array(
                edgeIn_dict[var_embedding[i][0]]).tolist() + \
                            np.array(edgeOut_dict[var_embedding[i][0]]).tolist()
            node_vec.append([var_embedding[i][0], var_feature_0])
            var_embedding[i][1] = var_feature_0

    node_vec_length = len(node_embedding[0][1])
    var_vec_length = len(var_embedding[0][1])

    if node_vec_length > var_vec_length:
        vec_length = node_vec_length
    else:
        vec_length = var_vec_length

    for i in range(len(node_embedding)):
        vec = np.zeros(vec_length, dtype=int)
        vec[0:len(np.array(node_embedding[i][1]))] = np.array(node_embedding[i][1])
        node_embedding[i][1] = vec.tolist()

    for i in range(len(var_embedding)):
        vec = np.zeros(vec_length, dtype=int)
        vec[0:len(np.array(var_embedding[i][1]))] = np.array(var_embedding[i][1])
        var_embedding[i][1] = vec.tolist()

    for i in range(len(node_embedding)):
        node_vec.append([node_embedding[i][0], node_embedding[i][1]])

    for i in range(len(node_vec)):
        node_vec[i][0] = node_convert[node_vec[i][0]]

    for i in range(len(edge_encode)):
        edge_encode[i][0] = node_convert[edge_encode[i][0]]
        edge_encode[i][1] = node_convert[edge_encode[i][1]]

    graph_edge = []
    for i in range(len(edge_encode)):
        graph_edge.append([edge_encode[i][0], edge_encode[i][2][1], edge_encode[i][1]])

    return node_vec, graph_edge


if __name__ == "__main__":
    current_detected_contract = sys.argv[1]  # 这里就是输入的参数(合约名)
    # current_detected_contract = "6798.sol"
    node_feature, edge_feature = generate_graph("../data/timestamp/contract/" + current_detected_contract)

    node_encode, var_encode, node_embedding, var_embedding = embedding_node(node_feature)

    edge_encode, edge_embedding = embedding_edge(edge_feature)
    node_vec, graph_edge = construct_vec(edge_feature, node_embedding, var_embedding, edge_embedding, edge_encode)

    graph_output = open('../data/timestamp/valid.json', 'w')
    contract_name = open("./timestamp_contract_name.txt", "r")  # contracts list
    contract_label = open("./timestamp_contract_label.txt", "r")  # contracts label
    names = contract_name.readlines()
    labels = contract_label.readlines()
    label = None
    for i in range(len(names)):
        if names[i].strip('\n') == current_detected_contract:
            label = labels[i].strip('\n')

    corenodes_feature_list = []
    for i in range(len(node_vec)):
        corenodes_feature_list.append(node_vec[i][1])

    edge_dict = {
        "graph": graph_edge
    }

    node_feature_dict = {
        "node_features": corenodes_feature_list,
    }

    graph_dict = ({
        "targets": label,
        "graph": graph_edge,  # graph_edge,
        "node_features": corenodes_feature_list,  # corenodes_feature_list
    })

    result = json.dumps(graph_dict)
    graph_output.write("[" + result + "]")
    graph_output.close()
    print("图向量提取成功，可执行检测！！！")
