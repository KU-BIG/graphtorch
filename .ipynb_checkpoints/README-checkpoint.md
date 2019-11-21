# Weight Agnostic Neural Networks PyTorch Implementation



# What we need

## 1. Class converts matrix representating graph to PyTorch NN model

- shape(matrix) : [# of nodes x # of nodex] ?
  - As limitation exists for input and output layer, it might be reduced
  - activation fuction can be also represented in this format like below. In matrix, from node to node
    - 0 : no connection 
    - 1 : connection with linear activation function
    - 2 : connection with ReLU activation function
    - and so on..

## 2. NEAT Algorithm

- It might be already implemented in other packages
- If it is, just use it
- If not, implement like [this( sklearn-genetic)](https://pypi.org/project/sklearn-genetic/)
  - it should have estimator which caculate the fitness
  - GeneticSelectionCV() class 
  - GeneticSelectionCV().fit() performs search, it saves history and the optimal result inside the class
  - and so on .. (need further digging)
  

## 3. Experiment

### 1) MNIST에 대해서 

- 논문에서 사용한것처럼 다양한 activation사용하면서 backprop -> 논문에서 제시된 1~4 (random weight, .., ..., fine-tune)과 비교 -> 성능이 얼마나 떨어지는지
- ReLU만 사용해서 MNIST에 대한 best topology 찾고, backprop으로 fine tune -> 논문에서 제시한 성능이랑 얼마나 차이나는지 + params. 가 얼마나 차이나는지
- Gradient Vanshing을 크게 일으킬것으로 기대되는 activation빼고 사용하여 위와 같은 방법으로 비교



