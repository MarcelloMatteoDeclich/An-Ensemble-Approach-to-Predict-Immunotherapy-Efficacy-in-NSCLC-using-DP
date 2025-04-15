import os

# Define the path to the slides directory
# Combines '/mnt', 'd', and 'Digital_path' into a platform-independent path
slide_path = os.path.join("/mnt", 'd', "Digital_path")

# Define the project name and create a project path
project_name = "I3lung-sqad-project"  # Name of the project
project_path = os.path.join("projects", project_name)  # Path to project directory

def list_extensions(folder_path):
    """
    Get a list of unique file extensions present in a specified folder.
    
    Args:
        folder_path (str): Path to the directory to scan for file extensions
        
    Returns:
        list: Alphabetically sorted list of unique file extensions (including the dot)
    
    Example:
        >>> list_extensions("/path/to/files")
        ['.svs', '.zip', '.tiff']
    """
    # List all files and directories in the specified folder
    folder_files = os.listdir(folder_path)
    
    # Extract file extensions for all files (ignoring directories)
    # os.path.splitext splits filename into (root, ext) - we take the extension part
    # We only process actual files (not directories) by checking with os.path.isfile
    extensions = [os.path.splitext(file)[1] 
                 for file in folder_files 
                 if os.path.isfile(os.path.join(folder_path, file))]
    
    # Get unique extensions:
    # - Convert to set to remove duplicates
    # - Filter out empty strings (files without extensions)
    unique_extensions = set([ext for ext in extensions if ext])

    # Return as sorted list for consistent ordering
    return sorted(list(unique_extensions))