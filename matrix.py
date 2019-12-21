#import modules needed  
import numpy as np
import torch 
import torch.nn as nn

#class for matrix information 
class SparseMatrix():
    
    def __init__(self, mat, in_dim, out_dim):
        # get when initialized
        self.mat = mat  
        self.in_dim = in_dim  
        self.out_dim = out_dim  
        
        #calculate
        self.num_hidden_nodes = self.mat.shape[1] - self.in_dim 
        
        #calculate total number of connection in matrix 
        self.connection_count = np.count_nonzero(self.mat)
        
        #when matrix has hidden layer  
        if self.num_hidden_nodes == 1:  
            self.hidden_dim = [1] 
        elif self.num_hidden_nodes == 0:
            self.hidden_dim = []
        else:
            self.hidden_dim = self.get_hidden_dim()   
            
            
    def get_hidden_dim(self):
        in_dim = self.in_dim
        out_dim = self.out_dim
        mat_mask = self.mat
        
        hidden_dim_list = []
        start_col_idx = 0
        finish_col_idx = in_dim -1   
        
        while(True):
            
            if finish_col_idx >= mat_mask.shape[1]:   
                #print(finish_col_idx)
                break  
            
            if ((mat_mask.shape[0] - sum(hidden_dim_list)) == out_dim):  #example4 해결
                 break  #지금 hidden dimension들 합이랑 output dim 합이 row길이랑 같으면 더이상 탐색 필요 x
            
            for i in range(sum(hidden_dim_list), len(mat_mask)): #이부분이상한데..?   
    
                #밑에처럼 하면 example 2에서 오류가 남.
                #skip connection에 대한 예외처리 해줘야 함   
                if(mat_mask[i,start_col_idx:(finish_col_idx + 1)].sum() == 0):   
                
                    hidden_dim = i - sum(hidden_dim_list)
                    hidden_dim_list += [hidden_dim]
                    start_col_idx = finish_col_idx + 1
                    finish_col_idx += hidden_dim   
                    break    
                    
        return hidden_dim_list     