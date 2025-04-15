import os
import numpy as np
from sklearn import base

def calculate_mean_of_matrices(base_folder, subfolders):
    """
    Calcola la media delle matrici con lo stesso nome presenti in diverse cartelle.
    
    Args:
        base_folder (str): Il percorso della cartella principale che contiene le sottocartelle.
        subfolders (list): Lista delle sottocartelle da processare.

    Returns:
        dict: Un dizionario con i nomi dei file come chiavi e la media delle matrici come valori.
    """
    # Inizializza un dizionario per salvare le medie finali
    mean_results = {}

    #subfolder = [name for name in os.listdir(base_folder) if os.path.isdir(os.path.join(base, name))]

    # Recupera tutti i nomi dei file dalla prima cartella
    folder_path = os.path.join(base_folder, subfolders[0])
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Itera su ogni file
    for filename in filenames:
        matrices = []  # Per salvare tutte le matrici con lo stesso nome
        shape_check = None  # Variabile per controllare che tutte le matrici abbiano la stessa shape
        
        for subfolder in subfolders:

            file_path = os.path.join(base_folder, subfolder, filename)

            # Pseudocodice: carica il file e convertilo in un numpy array
            # Ad esempio: matrix = np.load(file_path)
            matrix = load_numpy_array_from_npzfile(file_path)  # Sostituisci con il codice effettivo
            
            # Controlla che tutte le matrici abbiano la stessa shape
            if shape_check is None:
                shape_check = matrix.shape
            elif matrix.shape != shape_check:
                raise ValueError(f"Shape mismatch for file '{filename}' in folder '{subfolder}'.")

            matrices.append(matrix)

        # Calcola la media delle matrici
        mean_matrix = np.mean(matrices, axis=0)
        mean_results[filename] = mean_matrix

    return mean_results


def calculate_mean_matrices_and_names(base_folder, subfolders):
    """
    Calcola la media delle matrici con lo stesso nome presenti in diverse cartelle.
    
    Args:
        base_folder (str): Il percorso della cartella principale che contiene le sottocartelle.
        subfolders (list): Lista delle sottocartelle da processare.

    Returns:
        dict: Un dizionario con i nomi dei file come chiavi e la media delle matrici come valori.
    """
    # Inizializza un dizionario per salvare le medie finali
    mean_results = {}


    # Recupera tutti i nomi dei file dalla prima cartella
    folder_path = os.path.join(base_folder, subfolders[0])
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Itera su ogni file
    for filename in filenames:
        matrices = []  # Per salvare tutte le matrici con lo stesso nome
        shape_check = None  # Variabile per controllare che tutte le matrici abbiano la stessa shape
        
        for subfolder in subfolders:

            file_path = os.path.join(base_folder, subfolder, filename)

            #Loads numpy array
            matrix = load_numpy_array_from_npzfile(file_path)
            
            # Controlla che tutte le matrici abbiano la stessa shape
            if shape_check is None:
                shape_check = matrix.shape
            elif matrix.shape != shape_check:
                raise ValueError(f"Shape mismatch for file '{filename}' in folder '{subfolder}'.")

            matrices.append(matrix)

        # Calcola la media delle matrici
        mean_matrix = np.mean(matrices, axis=0)
        mean_results[filename.replace("_att.npz", "")] = mean_matrix


    return mean_results

# Funzione di esempio per aprire un file e caricare un numpy array
def load_numpy_array_from_npzfile(file_path):
    """
    Carica un numpy array da un file. (Sostituire con l'implementazione reale)
    """
    data = np.load(file_path)

    array = data["arr_0"]
    return array


def filter_and_count_dict(input_dict, k):
    """
    Conta e filtra gli elementi di un dizionario in base alla shape dei valori numpy array.

    Args:
        input_dict (dict): Dizionario con valori numpy array.
        k (int): Parametro che definisce il limite minimo per y in shape (y,).

    Returns:
        tuple: Un dizionario filtrato, il numero di scalari, e il numero di array con shape (y,) dove y < k.
    """
    filtered_dict = {}
    count_scalars = 0
    count_small_arrays = 0

    for key, value in input_dict.items():
        if value.shape == ():  # Caso scalare
            count_scalars += 1
        elif len(value.shape) == 1 and value.shape[0] < k:  # Caso (y,) con y < k
            count_small_arrays += 1
        else:
            filtered_dict[key] = value  # Mantieni il valore nel nuovo dizionario

    return filtered_dict, count_scalars, count_small_arrays



import os
import torch

def create_k_embeddings_tensor(attention_slide_name, weighted_attention_tensor, bag_path, k, verbose=False):
    """
    Process attention and embedding tensors to return a reshaped tensor of the top-k embeddings, 
    weighted by their respective attention values.

    Args:
        attention_slide_name (str): Name of the attention file (e.g., 'example_att.npz').
        weighted_attention_tensor (numpy.ndarray): Array containing attention values.
        bag_path (str): Path to the folder containing the embedding .pt file.
        k (int): The number of top-k attention values to consider.
        verbose (bool): If True, print detailed progress.

    Returns:
        torch.Tensor: 1D tensor of top-k embeddings weighted by attention values.
    """
    if verbose:
        print(f"Processing attention slide: {attention_slide_name}")

    # Step 1: Remove "_att.npz" from the attention file name to match embedding filename
    match = attention_slide_name.replace("_att.npz", "")
    
    # Step 2: Construct the path to the corresponding embedding .pt file
    embedding_file = os.path.join(bag_path, match + ".pt")
    if verbose:
        print(f"Loading embeddings from: {embedding_file}")
    
    # Load the embeddings (assuming they're stored in a .pt file)
    embeddings = torch.load(embedding_file)

    # Step 3: Convert the weighted attention tensor to a PyTorch tensor
    weighted_attention_tensor = torch.from_numpy(weighted_attention_tensor)

    # Step 4: Get the top-k attention values and their corresponding indices
    top_k_values, top_k_indices = torch.topk(weighted_attention_tensor, k)
    if verbose:
        print(f"Top-k attention values: {top_k_values}")
        print(f"Top-k indices: {top_k_indices}")
    
    # Step 5: Extract the top-k embeddings corresponding to the top-k indices
    top_k_embeddings = embeddings[top_k_indices]
    if verbose:
        print(f"Top-k embeddings shape: {top_k_embeddings.shape}")
    
    # Step 6: Multiply each top-k embedding by its corresponding attention value
    weighted_embeddings = top_k_embeddings * top_k_values.unsqueeze(1)
    if verbose:
        print(f"Weighted embeddings shape: {weighted_embeddings.shape}")
    
    # Step 7: Flatten the resulting tensor into a 1D vector
    reshaped_tensor = weighted_embeddings.reshape(-1)
    if verbose:
        print(f"Reshaped tensor shape: {reshaped_tensor.shape} \n")
    
    return reshaped_tensor


import os
import pandas as pd

def create_dataframe_from_filenames(directory_path):
    """
    Create a pandas DataFrame containing file names in the specified directory.

    Args:
        directory_path (str): Path to the directory containing the files.

    Returns:
        pd.DataFrame: DataFrame with a column "FileName" listing all file names.
    """
    # List all files in the directory
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    file_names = [f.replace("_att.npz", "") for f in file_names]

    # Create the DataFrame
    df = pd.DataFrame(file_names, columns=["slide"])
    
    return df

