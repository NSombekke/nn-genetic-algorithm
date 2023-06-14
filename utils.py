def bin2weight(bin_string: str, min_w: float = -3.0, max_w: float = 3.0) -> float:
  """
  Function to convert a bit string to a weight in the neural network.
  
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

def weight2bin(weight: float, num_bits: int = 8, min_w: float = -3.0, max_w: float = 3.0) -> str:
  """
  Function to convert a weight in the neural network to a bit string.
  
  Each weight is converted to a bit string of length num_bits.
  
  Arguments:
    weight: The inputted weight of the neural network.
    num_bits: The number of bits to be used for the bit string.
    min_w: The minimum possible value of a weight.
    max_w: The maximum possible value of a weight.
  
  Returns:
    bin_string: The bit string of length num_bits.
  """
  bin2int = int(((1 << num_bits) - 1) * (weight - min_w) / (max_w - min_w) + 0.5)
  bin_string = format(bin2int, 'b').zfill(num_bits)
  return bin_string

print((8 * 100 * 100 * 4) * 8)