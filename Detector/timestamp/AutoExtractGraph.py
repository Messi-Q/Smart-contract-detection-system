
# map user-defined variables to symbolic names(var)
var_list = ['balances', 'userBalance[msg.sender]', '[msg.sender]', '[from]', '[to]', 'msg.sender']

# function limit type
function_limit = ['private', 'onlyOwner', 'internal', 'onlyGovernor', 'onlyCommittee', 'onlyAdmin', 'onlyManager',
                  'only_owner', 'preventReentry', 'onlyProxyOwner', 'ownerExists', 'noReentrancy']

# Boolean condition expression:
var_op_bool = ['~', '**', '!=', '<', '>', '<=', '>=', '<<=', '>>=', '==', '<<', '>>', '||', '&&']

# Assignment expressions
var_op_assign = ['|=', '&=', '+=', '-=', '*=', '/=', '%=', '++', '--', '=']

# function call restrict
core_call_restrict = ['NoLimit', 'LimitedAC']

# Suspicious reentrancy mark
var_reentrancy_mark = ['compliance', 'warning', 'violation']


# split all functions of contracts
def split_function(filepath):
    function_list = []
    f = open(filepath, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    flag = -1

    for line in lines:
        text = line.strip()
        if len(text) > 0 and text != "\n":
            if text.split()[0] == "function" or text.split()[0] == "function()":
                function_list.append([text])
                flag += 1
            elif len(function_list) > 0 and ("function" in function_list[flag][0]):
                function_list[flag].append(text)

    return function_list


def generate_graph(filepath):
    allFunctionList = split_function(filepath)  # Store all functions
    timeStampList = []  # Store all W functions that call call.value
    otherFunctionList = []  # Store functions other than W functions
    node_list = []  # Store all the points
    edge_list = []  # Store edge and edge features
    node_feature_list = []  # Store nodes feature
    params = []  # Store the parameters of the W functions
    param = []
    timeStampFlag = 0
    key_count = 0  # Number of core nodes S and W

    # Store other functions without W functions (with block.timestamp)
    for i in range(len(allFunctionList)):
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if 'block.timestamp' in text:
                timeStampList.append(allFunctionList[i])
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])

    # ======================================================================
    # ---------------------------  store S and W    ------------------------
    # ======================================================================
    # Traverse all functions, find the block.timestamp keyword, store the S and W nodes
    for i in range(len(timeStampList)):
        node_list.append("S" + str(i))
        node_list.append("W" + str(i))
        timeStampFlag += 1

        for j in range(len(timeStampList[i])):
            text = timeStampList[i][j]

            # Handling W function access restrictions, which can be used for access restriction properties
            if 'block.timestamp' in text:
                limit_count = 0
                for k in range(len(function_limit)):
                    if function_limit[k] in timeStampList[i][0]:
                        limit_count += 1
                        node_feature_list.append(
                            ["S" + str(i), "LimitedAC", ["W" + str(i)], 2])
                        node_feature_list.append(
                            ["W" + str(i), "LimitedAC", ["NULL"], 1])
                        edge_list.append(["W" + str(i), "S" + str(i), 1, 'FW'])
                        break
                    elif k + 1 == len(function_limit) and limit_count == 0:
                        node_feature_list.append(
                            ["S" + str(i), "NoLimit", ["W" + str(i)], 2])
                        node_feature_list.append(
                            ["W" + str(i), "NoLimit", ["NULL"], 1])

                    edge_list.append(["W" + str(i), "S" + str(i), 1, 'FW'])

    # ======================================================================
    # ---------------------------  store var nodes  ------------------------
    # ======================================================================
    for i in range(len(timeStampList)):
        TimestampFlag1 = 0
        TimestampFlag2 = 0
        VarTimestamp = None
        varFlag = 0

        for j in range(len(timeStampList[i])):
            text = timeStampList[i][j]
            if 'block.timestamp' in text and "=" in text:
                TimestampFlag1 += 1
                VarTimestamp = text.split("=")[0]
            elif 'block.timestamp' in text:
                TimestampFlag2 += 1
                if 'return' in text:
                    node_feature_list.append(["VAR" + str(varFlag), "S" + str(i), 3, 'violation'])
                    edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'RE'])
                    node_list.append("VAR" + str(varFlag))
                    varFlag += 1
                    break
                if TimestampFlag1 == 0:
                    if "assert" in text:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'AH'])
                    elif "require" in text:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'RG'])
                    elif "if" in text:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'IF'])
                    elif "for" in text:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'FOR'])
                    else:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'FW'])

                    var_node = 0
                    for a in range(len(var_op_assign)):
                        if var_op_assign[a] in text:
                            node_feature_list.append(["VAR" + str(varFlag), "S" + str(i), 3, 'warning'])
                            var_node += 1
                            break
                    if var_node == 0:
                        node_feature_list.append(["VAR" + str(varFlag), "S" + str(i), 3, 'compliance'])

                    node_list.append("VAR" + str(varFlag))
                    varFlag += 1
                    break
            if TimestampFlag1 != 0 and TimestampFlag2 == 0:
                if VarTimestamp != " " or "":
                    if "return" in text and VarTimestamp in text:
                        node_feature_list.append(["VAR" + str(varFlag), "S" + str(i), 3, 'violation'])
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'RE'])
                        node_list.append("VAR" + str(varFlag))
                        varFlag += 1
                        break
                    elif VarTimestamp in text:
                        node_feature_list.append(["VAR" + str(varFlag), "S" + str(i), 3, 'warning'])
                        if "assert" in text:
                            edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'AH'])
                        elif "require" in text:
                            edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'RG'])
                        elif "if" in text:
                            edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'IF'])
                        elif "for" in text:
                            edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'FOR'])
                        else:
                            edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'FW'])
                        node_list.append("VAR" + str(varFlag))
                        varFlag += 1
                        break
                else:
                    node_feature_list.append(["VAR" + str(varFlag), "S" + str(i), 3, 'compliance'])
                    if "assert" in text:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'AH'])
                    elif "require" in text:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'RG'])
                    elif "if" in text:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'IF'])
                    elif "for" in text:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'FOR'])
                    else:
                        edge_list.append(["S" + str(i), "VAR" + str(varFlag), 2, 'FW'])
                    node_list.append("VAR" + str(varFlag))
                    varFlag += 1
                    break

    if timeStampFlag == 0:
        node_feature_list.append(["S0", "NoLimit", ["NULL"], 0])
        node_feature_list.append(["W0", "NoLimit", ["NULL"], 0])
        node_feature_list.append(["VAR0", "NULL", 0, "compliance"])
        edge_list.append(["W0", "S0", 0, 'FW'])
        edge_list.append(["S0", "VAR0", 0, 'FW'])

    # Handling some duplicate elements, the filter leaves a unique
    edge_list = list(set([tuple(t) for t in edge_list]))
    edge_list = [list(v) for v in edge_list]
    node_feature_list_new = []
    [node_feature_list_new.append(i) for i in node_feature_list if not i in node_feature_list_new]

    node_feature, edge_feature = nomralize_result(node_feature_list_new, edge_list)

    main_point = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
                  'S16', 'S17',
                  'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15',
                  'W16', 'W17']
    node_feature1 = []
    edge_feature1 = []
    node_flag = 0
    edge_flag = 0

    for i in range(len(node_feature)):
        for j in range(len(node_feature[i])):
            if node_feature[i][j] in main_point:
                node_flag = 1
            elif node_flag == 0 and j + 1 == len(node_feature[i]):
                node_feature1.append(node_feature[i])
                node_flag = 0
            elif j + 1 == len(node_feature[i]):
                node_flag = 0

    for i in range(len(edge_feature)):
        for j in range(len(edge_feature[i])):
            if edge_feature[i][j] in main_point:
                edge_flag = 1
            elif edge_flag == 0 and j + 1 == len(edge_feature[i]):
                edge_feature1.append(edge_feature[i])
                edge_flag = 0
            elif j + 1 == len(edge_feature[i]):
                edge_flag = 0

    return node_feature1, edge_feature1


def nomralize_result(node_feature, edge_feature):
    main_point = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
                  'S16', 'S17',
                  'W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15',
                  'W16', 'W17']

    for i in range(len(node_feature)):
        if node_feature[i][0] in main_point:
            for j in range(0, len(node_feature[i][2]), 2):
                if j + 1 < len(node_feature[i][2]):
                    tmp = node_feature[i][2][j] + "," + node_feature[i][2][j + 1]
                elif len(node_feature[i][2]) == 1:
                    tmp = node_feature[i][2][j]

            node_feature[i][2] = tmp

    return node_feature, edge_feature


if __name__ == "__main__":
    test_contract = "../data/timestamp/contract/1456.sol"
    node_feature, edge_feature = generate_graph(test_contract)
    node_feature = sorted(node_feature, key=lambda x: (x[0]))
    edge_feature = sorted(edge_feature, key=lambda x: (x[2], x[3]))
    # print(node_feature)
    # print(edge_feature)
