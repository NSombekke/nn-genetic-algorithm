import numpy as np

from neural_network import NeuralNetwork
from utils import chrom_length

def generate_population(num_inputs: int, num_hidden: int|list|tuple, num_outputs: int, bias: bool = True,
                        num_bits_w: int = 8, min_w: float = -1.0, max_w: float = 1.0, 
                        gen_num: None|int = None, pop_size: int = 100) -> list:
  """
  Generates a population of neural networks.
  """
  population = []
  chrom_len = chrom_length(num_inputs, num_hidden, num_outputs, bias, num_bits_w)
  for id in range(pop_size):
    chrom = "".join([str(np.random.randint(0, 2)) for _ in range(chrom_len)])
    population.append(NeuralNetwork(chrom, num_inputs, num_hidden, num_outputs, bias=bias, 
                                    num_bits_w=num_bits_w, min_w=min_w, max_w=max_w, 
                                    gen_num=gen_num, pop_id=id))
  return population

def calculate_fitness_child(child: NeuralNetwork) -> float:
  """
  """
  pass

def calculate_fitness_population(population: list) -> list:
  """
  """
  pass
  
def selection(population: list, fitness: list, num_parents: int = 2) -> list:
  """
  """
  pass

def crossover(parents: list, num_offspring: int = 2) -> list:
  """
  """
  pass

def mutation(chrom: str, mutation_rate: float = 0.008) -> str:
  """
  """
  for i in range(len(chrom)):
    if np.random.rand() < mutation_rate:
      chrom[i] = str(1 - int(chrom[i]))
  return chrom

def create_next_population():
  """
  """
  

if __name__ == "__main__":
  print(generate_population(2, 2, 2, bias=True, num_bits_w=8, pop_size=1000))