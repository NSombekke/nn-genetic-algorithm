import numpy as np
from utils import chrom2weights, weights2numpy

class NeuralNetwork:
  """
  A class to represent a neural network from a chromosome.
  
  Arguments:
    chrom: The chromosome.
    num_inputs: The number of inputs to the neural network.
    num_hidden: The number of hidden layers in the neural network.
    num_outputs: The number of outputs from the neural network.
    bias: Whether or not to include bias in the neural network.
    num_bits_w: The number of bits to be used for each weight (bit-precision).
    min_w: The minimum possible value of a weight.
    max_w: The maximum possible value of a weight.
    gen_num: The generation number.
    pop_id: The individual population ID.
  """
  def __init__(self, chrom: str, num_inputs: int, num_hidden: int|list|tuple, num_outputs: int, bias: bool = True, 
               num_bits_w: int = 8, min_w: float = -3.0, max_w: float = 3.0,
               gen_num: None|int = None, pop_id: None|int = None) -> None:
    self.chrom = chrom
    self.gen_num = gen_num
    self.pop_id = pop_id
    self.weights = chrom2weights(chrom, num_inputs, num_hidden, num_outputs, bias=bias,
                            num_bits_w=num_bits_w, min_w=min_w, max_w=max_w)
    self.layers = weights2numpy(self.weights, num_inputs, num_hidden, num_outputs, bias=bias)
    
  def __call__(self, x: np.array) -> np.array:
    return self.forward(x)
    
  def forward(self, x: np.array) -> np.array:
    for i in range(len(self.layers) - 1):
      x = np.dot(self.layers[i], x)
      x = sigmoid(x)
    x = np.dot(self.layers[-1], x)
    return softmax(x)
  
def sigmoid(x: np.array) -> np.array:
  return 1 / (1 + np.exp(-x))

# def relu(x: np.array) -> np.array:
#   return np.maximum(0, x)

def softmax(x: np.array) -> np.array:
  return np.exp(x) / np.sum(np.exp(x), axis=0)