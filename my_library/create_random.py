import os
import random
import numpy as np
import torch

def set_random(random_path, seed):
    """
    Initialize and manage random numbers for reproducibility.
    
    Args:
        random_path (str): Path to store/load random numbers file
        seed (int): Seed value for random number generation
    
    Behavior:
        - If a file with random numbers exists, loads them
        - If not, generates new random numbers and saves them
        - Always sets the Python random seed
    """
    import random
    random.seed(seed)  # Set Python's random seed
    print(random.random())  # Print a test random number
    
    random_numbers = []  # Initialize empty list for random numbers

    # Check if random numbers file exists and is not empty
    if os.path.isfile(random_path) and os.path.getsize(random_path) > 0:
        random_numbers = []
        with open(random_path, 'r') as fp:
            for line in fp:
                # Remove linebreak and convert to integer
                x = line[:-1]
                random_numbers.append(int(x))

        print(random_numbers)  # Display loaded numbers
    else:
        # Generate 25 new random numbers between 1 and 100,000,000
        random_numbers = [random.randint(1, 100000000) for _ in range(25)]
        print(random_numbers)
        
        # Save numbers to file
        file = open(random_path+'/random_numbers.txt', 'w')
        for item in random_numbers:
            file.write("%s\n" % item)
        file.close()


def load_random(random_path):
    """
    Load previously saved random numbers from file.
    
    Args:
        random_path (str): Path to random numbers file
    
    Returns:
        list: List of integers loaded from file
    """
    with open(random_path+'/random_numbers.txt', "r") as file:
        # Read each line, strip whitespace, and convert to integer
        numbers = [int(line.strip()) for line in file]
    return numbers


def set_random_seed(seed_value):
    """
    Set all random seeds across different libraries for complete reproducibility.
    
    Args:
        seed_value (int): Seed value to use for all random number generators
    
    Returns:
        int: The same seed value that was input
    
    Sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA if available)
    """
    import random
    # Set seeds for all relevant libraries
    random.seed(seed_value)  # Python random
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch CPU
    
    # Set seed for all CUDA devices if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    
    print(f"Random seed set to: {seed_value}")
    return seed_value