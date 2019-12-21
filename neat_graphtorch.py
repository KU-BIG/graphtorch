import graphtorch
import numpy as np
import pandas as pd
import random
import copy

def change_activation(SparseMatrix_original, activations):
    
    SparseMatrix = copy.deepcopy(SparseMatrix_original)
    in_dim = SparseMatrix.in_dim
    out_dim = SparseMatrix.out_dim
    mat = SparseMatrix.mat  
    indices = (mat != 0) #0이 아닌 index들만 activation 바꿀 수 있으니까 true false matrix 받기   
    possible = [] # activation을 바꿀 수 있는 index 리스트. [[0, 0], [1, 3], [1, 5]] 이런 식으로 나옴 
    
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            if indices[i,j] == True:
                possible += [[i,j]]
    
    ran_idx = random.choice(possible) # [1,5] 이런 식으로 나오는데     
    
    possible_activation = [i for i in range(len(activations))]
    ran_activation = random.choice(possible_activation[1:])
    
    #change activation
    mat[ran_idx[0],ran_idx[1]] = ran_activation 
    
    new_SparseMatrix = graphtorch.SparseMatrix(mat, in_dim, out_dim) #이렇게 해야 callable 하더라고.. 
    

    return new_SparseMatrix

def add_connection(SparseMatrix):
    
    mat = SparseMatrix.mat 
    in_dim = SparseMatrix.in_dim
    out_dim = SparseMatrix.out_dim  
    hidden_dim = SparseMatrix.hidden_dim 
    
    #random 하게 connection 추가 
    # 같은 layer끼리는 connection x -> hidden layer끼리만 봐주면 됨 (input끼리 output끼리는 matrix에 없으니까) 
    # 0인 element 중에서 임의로 하나 바꾸기 -> 1,2,3 중에서(linear, relu, sigmoid)   
    # 가능한 element 중에서 랜덤하게 하나의 element 뽑고, 거기서 또 (1,2,3) 중에서 랜덤하게 하나 뽑기  
    
    ###################
    #hidden layer끼리 연결되어 있는 indices 들 뽑아내기
    hidden_wise_indices = []
    #print(hidden_dim[0])
    row_start = 0
    row_end = 0 + hidden_dim[0] - 1
    col_start = in_dim 
    col_end = in_dim + hidden_dim[0] - 1 
    #print(col_start, col_end)
    
    for hidden_count, current_hidden_dim in enumerate(hidden_dim): 
        #이전 laye의 dimension
        for i in range(row_start, row_end + 1):
            #print(col_start, col_end)
            for j in range(col_start, col_end + 1):
                
                hidden_wise_indices += [[i,j]]
        
        row_start += current_hidden_dim  
        col_start += current_hidden_dim 
        row_end += (current_hidden_dim - 1)  
        col_end += (current_hidden_dim - 1)

    ##################################    
    #우선 0인 element들의 indices 받아오기   
    indices = (mat == 0)     
    zero_indices = [] # 0이 아닌 애들 indexes 우선 받아오기 
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            if indices[i,j] == True:
                zero_indices += [[i,j]]
    
    
    #################################
    # hidden_wise_indices에 해당되지 않는 zero_indices만이 possible에 포함될 수 있음  
    # (zero_indices) and (! hidden_wise_indices)
    # 같은 hidden layer끼리 있지 않은 인덱스들 
    possible = []
    possible = [value for value in zero_indices if value not in hidden_wise_indices]  
    
    
    # 우선 랜덤하게 어떤 element 바꿀 지 선택
    ran_idx = random.choice(possible)
    # select random activation -> 안하고 바로 liear 로 세팅
    #possible_activation = [1, 2, 3] #linear, relu, sigmoid   
    #ran_activation = random.choice(possible_activation)
    
    mat[ran_idx[0], ran_idx[1]] = 1   
    
    new_SparseMatrix = graphtorch.SparseMatrix(mat, in_dim, out_dim)  
    
    
    
    return new_SparseMatrix



#function 1 to add node
def which_layer(total_dim, from_node_num, to_node_num):
    #선택된 index가 어느 layer와 어느 layer에 해당하는지 list return 
    layer_idx = [0]*len(total_dim)
    
    #아래 for문으로 짜야할듯 
    for idx in range(len(total_dim)):
        if from_node_num <= sum(total_dim[0:(idx+1)]):  
            layer_idx[idx] = 1
            break
            
    for idx in range(len(total_dim)):  
        if to_node_num <= sum(total_dim[0:(idx+1)]):
            layer_idx[idx] = 1
            break 
    
    return layer_idx # [1,0,1,0,0] 형태로 return

#function 2 to add node 
def extra_hidden_idx(layer_idx, total_dim, in_dim):
    idx_total = None
    from_layer_idx = None
    to_layer_idx = None
    
    for i in range(len(layer_idx)):
        if from_layer_idx == None and layer_idx[i] == 1:
            from_layer_idx = i
        elif from_layer_idx != None and layer_idx[i] == 1:
            to_layer_idx = i
    print("from_layer_idx:{}".format(from_layer_idx))
    print("to_layer_idx:{}".format(to_layer_idx))
    
    idx_total = sum(total_dim[0:(to_layer_idx)]) #이건 number은 아니고 0부터 시작하는 index 
    #num_total = idx_total + 1 #이건 앞에서부터의 순 노드 number(1부터 시작하는)
    idx_hidden = idx_total - in_dim
    #num_hidden = idx_hidden + 1
    print("idx_total:{}".format(idx_total))
    #print("num_total:{}".format(num_total))
    print("idx_hidden:{}".format(idx_hidden))
    #print("num_hidden:{}".format(num_hidden))
    
        
    return idx_hidden # 몇 번째 hidden node에 삽입될 것인가

#fuction 3 to add node 
def expand_dim_mat(mat, idx_hidden, in_dim):
    col_idx = in_dim + idx_hidden
    row_idx = idx_hidden
    mat = np.insert(mat, row_idx, 0, axis = 0) #index, value to fill, x axis 
    mat = np.insert(mat, col_idx, 0, axis = 1) #index, value to fill, y axis 

    return mat

#function 4 to add node 
def change_element(mat, idx_hidden, ran_idx, in_dim, original_activation, ran_activation, front_or_back):
    #끊기는 connection의 새로운 matrix에서의 index 
    from_idx = ran_idx[1]  
    to_idx = ran_idx[0] + 1 #hidden에 하나 추가가 되니까 하나 뒤로 밀림 
    #삽입된 노드의 index
    col_idx = in_dim + idx_hidden
    row_idx = idx_hidden
    print("from_idx:{}".format(from_idx))
    print("to_idx:{}".format(to_idx))
    print("col_idx:{}".format(col_idx))
    print("row_idx:{}".format(row_idx))
    
    if front_or_back == 0: #front에 새로운 activation
        mat[row_idx, from_idx] = ran_activation
        mat[to_idx, col_idx] = original_activation
    elif front_or_back == 1:
        mat[row_idx, from_idx] = original_activation
        mat[to_idx, col_idx] = ran_activation
    
    return mat

def add_node(SparseMatrix, activations):
    
    mat = SparseMatrix.mat 
    in_dim = SparseMatrix.in_dim
    out_dim = SparseMatrix.out_dim  
    hidden_dim = SparseMatrix.hidden_dim 
    
    #이미 connection이 존재하는 부분 사이에 node 추가하기 
    #원래 connection에 존재하던 activation 앞으로 넘길지 뒤로 넘길지 정하고, 나머지에는 랜덤하게 정해주기
    #노드 추가해주면 dimension 추가해주고 connection 다시 계산해 줘야 하는데.. 
    
    # connection이 존재하는 indices추출 
    indices = (mat != 0)
    possible = [] #node를 add할 수 있는 connection들 index 리스트 
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            if indices[i,j] == True:
                possible += [[i,j]]    
    #가능한 인덱스 중에서 하나 선택 
    ran_idx = random.choice(possible) #[i,j] 형태로 나옴 
    #랜덤하게 정할 activation 뽑기  
    possible_activation = [i for i in range(len(activations))]
    ran_activation = random.choice(possible_activation[1:]) 
    #원래 connection에 존재하던 activation type 앞으로 할지 뒤로 할지
    front = 0 
    back = 1
    front_or_back = random.choice([front, back]) 
    #선택된 index의 원래 actavation 가지고있기
    original_activation = mat[ran_idx[0], ran_idx[1]] 
    #선택된 index의 element 값 0으로 만들어주기
    mat[ran_idx[0], ran_idx[1]] = 0
    
    ## 몇번째 layer에 해당하는지만   
    #그 layer 마지막에 추가해주기   
    total_dim = [in_dim]
    for i in range(len(hidden_dim)):   
        total_dim += [hidden_dim[i]]
    total_dim += [out_dim]  
    
    #선택된 index 사이에 추가할 노드가 몇번째 hidden layer의 몇번째 hidden node인지 알아내야 -> mat의 index로 나오도록 
    # 선택된 index의 from = input + 그 이전의 hidden 노드 개수 
    # 선택된 index의 to = input + 그 이전의 hidden 노드 개수 + 그 이전의 output 노드 개수 
    # 우선 from이 앞에서부터 몇번째 노드인지 알아내기 (1부터 시작하는 node number)  
    from_node_num = ran_idx[1] + 1 
    to_node_num = in_dim + ran_idx[0] + 1 
    layer_idx = which_layer(total_dim, from_node_num, to_node_num) #function 1
    idx_hidden = extra_hidden_idx(layer_idx, total_dim, in_dim) #function 2
    
    #matrix dimension 늘리기(앞의 index 활용해서 늘려야함)     
    #추가 col idx = in_dim + idx_hidden
    #추가 row idx = idx_hidden 
    mat = expand_dim_mat(mat, idx_hidden, in_dim) #function 3
    
    #element 값 할당하기  
    mat = change_element(mat, idx_hidden, ran_idx, in_dim, original_activation, ran_activation, front_or_back) #function 4  
    
    
    #sparsematrix 객체 update
    new_SparseMatrix = graphtorch.SparseMatrix(mat, in_dim, out_dim) 
    
    return new_SparseMatrix