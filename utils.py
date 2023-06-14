def bin2weight(bin_string: str, min_w: float = -3.0, max_w: float = 3.0) -> float:
  """
  Convert a bit string to a weight in the neural network.
  
  Each bit is converted to a value between min_w and max_w.
  
  Arguments:
    bin_string: The inputted bit string of arbitrary length (8-bit, 16-bit, ...)
    min_w: The minimum possible value of a weight.
    max_w: The maximum possible value of a weight.
    
  Returns:
    weight: The weight to be used for the neural network.
    
  """
  num_bits = len(bin_string)
  bin2int = int(bin_string, 2)
  weight = (max_w - min_w) * (bin2int / ((1 << num_bits) - 1)) + min_w
  return weight

def weight2bin(weight: float, num_bits_w: int = 8, min_w: float = -3.0, max_w: float = 3.0) -> str:
  """
  Convert a weight in the neural network to a bit string.
  
  Each weight is converted to a bit string of length num_bits.
  
  Arguments:
    weight: The inputted weight of the neural network.
    num_bits_w: The number of bits to be used for the bit string (bit-precision).
    min_w: The minimum possible value of a weight.
    max_w: The maximum possible value of a weight.
  
  Returns:
    bin_string: The bit string of length num_bits.
  """
  bin2int = int(((1 << num_bits_w) - 1) * (weight - min_w) / (max_w - min_w) + 0.5)
  bin_string = format(bin2int, 'b').zfill(num_bits_w)
  return bin_string

def chrom2weights(chrom: str, num_inputs: int, num_hidden: int|list|tuple, num_outputs: int, bias: bool = True,
                  num_bits_w: int = 8, min_w: float = -3.0, max_w: float = 3.0) -> list:
  """
  Convert a chromosome to a list of weights for the neural network.
  
  Arguments:
    chrom: The inputted chromosome.
    num_inputs: The number of inputs to the neural network.
    num_hidden: The number of hidden layers in the neural network.
    num_outputs: The number of outputs from the neural network.
    bias: Whether or not to include bias in the neural network.
    num_bits_w: The number of bits to be used for each weight (bit-precision).
    min_w: The minimum possible value of a weight.
    max_w: The maximum possible value of a weight.
  
  Returns:
    layer_weights: The list of weights for each layer in the neural network.
  """
  layer_weights = []
  index = 0
  layers = flatten_list([num_inputs, num_hidden, num_outputs])
  for i in range(len(layers) - 1):
    num_weights = (layers[i] + 1) * layers[i + 1] if bias else layers[i] * layers[i + 1]
    bin_weights = chrom[index:index + num_weights * num_bits_w]
    index += num_weights * num_bits_w
    weights = [bin2weight(bin_weights[i:i + num_bits_w], min_w, max_w) for i in range(0, len(bin_weights), num_bits_w)]
    layer_weights.append(weights)
  return layer_weights

def weights2chrom(weights: list, num_bits_w: int = 8, min_w: float = -3.0, max_w: float = 3.0) -> str:
  """
  Convert a list of weights to a chromosome.
  
  Arguments:
    weights: The list of weights for each layer in the neural network.
    num_bits_w: The number of bits to be used for each weight (bit-precision).
    min_w: The minimum possible value of a weight.
    max_w: The maximum possible value of a weight.
    
  Returns:
    chrom: The chromosome.
  """
  weights = flatten_list(weights)
  chrom = ''.join([weight2bin(weight, num_bits_w, min_w, max_w) for weight in weights])
  return chrom

def chrom_length(num_inputs: int, num_hidden: int|list|tuple, num_outputs: int, bias: bool = True,
                 num_bits_w: int = 8):
  """
  Calculate the length of the chromosome for neural network architecture and number of bits per weight.
  
  Arguments:
    num_inputs: The number of inputs to the neural network.
    num_hidden: The number of hidden layers in the neural network.
    num_outputs: The number of outputs from the neural network.
    bias: Whether or not to include bias in the neural network.
    num_bits_w: The number of bits to be used for each weight (bit-precision).
    
  Returns:
    chrom_length: The length of the chromosome.
  """
  layers = flatten_list([num_inputs, num_hidden, num_outputs])
  num_weights = 0
  for i in range(len(layers) - 1):
    num_weights += (layers[i] + 1) * layers[i + 1] if bias else layers[i] * layers[i + 1]
  chrom_length = num_weights * num_bits_w
  return chrom_length

def flatten_list(l: list) -> list:
  """
  Flatten a heterogenous list of integers and lists.
  
  Arguments:
    l: The list of lists to be flattened.
    
  Returns:
    flat_list: The flattened list.
  """
  flatten_l = []
  if isinstance(l, (list, tuple)):
      for x in l:
          flatten_l.extend(flatten_list(x))
  else:
      flatten_l.append(l)
  return flatten_l