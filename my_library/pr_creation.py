import os
import slideflow as sf

def create_or_load_project(project_path):
    """
    Wrapper function to create or load a SlideFlow project.
    
    Args:
        project_path (str): Path to the project directory
        
    Returns:
        Project: Initialized SlideFlow Project object
    """
    return create_project(project_path)

def create_project(project_path):
    """
    Create or load a SlideFlow project at the specified path.
    
    Args:
        project_path (str): Path to the project directory
        
    Returns:
        Project: Initialized SlideFlow Project object
        
    Behavior:
        - Creates new project if path doesn't exist
        - Loads existing project if path exists
    """
    # Check if project directory exists
    if not os.path.exists(project_path):
        print("Directory created successfully!")
    else:
        print("Directory already exists!")

    # Initialize SlideFlow project
    if not os.path.exists(project_path):
        P = sf.create_project(root=project_path)
    else:
        P = sf.load_project(project_path)
    
    return P

def sf_extract_tiles(P, tile_px, tile_um, qc, num_threads, report):
    """
    Extract tiles from whole slide images using default settings.
    
    Args:
        P (Project): SlideFlow Project object
        tile_px (int): Tile size in pixels
        tile_um (int/int): Tile size in microns (or 'auto')
        qc (str): Quality control method ('otsu', 'blur', 'both', or None)
        num_threads (int): Number of CPU threads to use
        report (bool): Whether to generate extraction report
    """
    P.extract_tiles(
        tile_px=tile_px, 
        tile_um=tile_um, 
        qc=qc,
        num_threads=num_threads, 
        report=report
    )
    
def sf_extract_tiles_s(P, tile_px, tile_um, qc, num_threads, report, source):
    """
    Extract tiles from specific source(s) in the project.
    
    Args:
        P (Project): SlideFlow Project object
        tile_px (int): Tile size in pixels
        tile_um (int/int): Tile size in microns (or 'auto')
        qc (str): Quality control method
        num_threads (int): Number of CPU threads to use
        report (bool): Whether to generate extraction report
        source (str/list): Source name(s) to extract from
    """
    P.extract_tiles(
        tile_px=tile_px, 
        tile_um=tile_um, 
        qc=qc,
        num_threads=num_threads, 
        report=report,
        source=source
    )
    
def build_dataset(P, tile_px, tile_um, filters):
    """
    Build a dataset with basic parameters.
    
    Args:
        P (Project): SlideFlow Project object
        tile_px (int): Tile size in pixels
        tile_um (int/int): Tile size in microns (or 'auto')
        filters (dict): Dictionary of filters to apply
        
    Returns:
        Dataset: Configured SlideFlow Dataset object
    """
    dataset = P.dataset(
        tile_px=tile_px,
        tile_um=tile_um,
        filters=filters
    )
    return dataset

def build_dataset_s(P, tile_px, tile_um, filters, source):
    """
    Build a dataset from specific source(s) with no minimum tiles requirement.
    
    Args:
        P (Project): SlideFlow Project object
        tile_px (int): Tile size in pixels
        tile_um (int/int): Tile size in microns (or 'auto')
        filters (dict): Dictionary of filters to apply
        source (str/list): Source name(s) to include
        
    Returns:
        Dataset: Configured SlideFlow Dataset object
    """
    dataset = P.dataset(
        tile_px=tile_px,
        tile_um=tile_um,
        filters=filters,
        sources=source,
        min_tiles=0,
    )
    dataset.summary()
    return dataset

def build_dataset_sm(P, tile_px, tile_um, filters, source, min_tiles):
    """
    Build a dataset from specific source(s) with custom minimum tiles requirement.
    
    Args:
        P (Project): SlideFlow Project object
        tile_px (int): Tile size in pixels
        tile_um (int/int): Tile size in microns (or 'auto')
        filters (dict): Dictionary of filters to apply
        source (str/list): Source name(s) to include
        min_tiles (int): Minimum number of tiles required per slide
        
    Returns:
        Dataset: Configured SlideFlow Dataset object
    """
    dataset = P.dataset(
        tile_px=tile_px,
        tile_um=tile_um,
        filters=filters,
        sources=source,
        min_tiles=min_tiles,
    )
    dataset.summary()
    return dataset

def uni_encodings(dataset, uni_bags_path, magnification, qc, normalization):
    """
    Generate UNI feature embeddings for dataset tiles.
    
    Args:
        dataset (Dataset): SlideFlow Dataset object
        uni_bags_path (str): Base path to store UNI embeddings
        magnification (str): Magnification level (e.g., '20x')
        qc (str): Quality control method used
        normalization (str): Normalization method (or empty string)
        
    Returns:
        str: Path to directory containing generated feature bags
        
    Behavior:
        - Builds UNI feature extractor
        - Generates feature bags with specified normalization
        - Handles special case for no normalization
    """
    from slideflow.model import build_feature_extractor
    
    # Handle special case for no normalization
    if normalization == '':
        temp_normalization = normalization
        normalization = "No Gaussian"
        
    # Create output directory path
    dir_uni = os.path.join(uni_bags_path, f"{magnification} {normalization}")

    # Initialize UNI feature extractor
    uni = sf.build_feature_extractor('uni', weights="pytorch_model.bin")

    # Generate feature bags with appropriate normalization
    if normalization == "No Gaussian":
        normalization = temp_normalization 
        dataset.generate_feature_bags(uni, outdir=dir_uni)
    else:
        dataset.generate_feature_bags(uni, outdir=dir_uni, normalizer=normalization)
    
    return dir_uni