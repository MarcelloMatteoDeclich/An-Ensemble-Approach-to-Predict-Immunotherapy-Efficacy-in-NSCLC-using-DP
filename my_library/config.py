# Import necessary libraries
import os 
import platform

# Initialize empty strings for path variables
luad_string = ""  # Will store path to LUAD slides
lscc_string = ""  # Will store path to LSCC slides
sandisk = ""      # Will store path to Sandisk drive
samsung = ""      # Will store path to Samsung drive
thesis_folder = "" # Will store path to thesis code folder

### Device names used for path configuration
desktop_ubuntu = 'mmd-MS-7D98'  # Name of Ubuntu desktop device
lab_workstation = 'ArselaGPU'   # Name of lab workstation device

# Root directory (will be '/' on Unix systems)
root ="/"

# Configure the thesis folder path where code is located
# Contains a subfolder 'Uni' with the UNI encoder
thesis_folder = os.path.join("GitHub", "UMD")

# Get the current device name to automatically configure paths
device_name = platform.node()

# Flags for environment configuration
wsl = True    # Windows Subsystem for Linux
home = True   # Home computer vs work computer

# Set environment flags based on device name
if device_name == 'mmd-MS-7D98':
    home = True
    wsl = False

if device_name == 'mmd-Inspiron-7580':
    home = False
    wsl = False

lab = True  # Flag for lab environment

# Configure paths for Unix-like systems
if os.name == "posix": 
    mnt = os.path.join(root, "mnt")  # Mount point directory
    
    # Path configuration for WSL at home
    if wsl & home:
        samsung = os.path.join(mnt, 'd')
        sandisk = os.path.join(mnt, 'e')

    # Path configuration for WSL at work
    elif wsl & ~home:
        samsung = os.path.join(mnt, 'e')
        sandisk = os.path.join(mnt, 'f')

    # Path configuration for native Linux
    else:
        samsung = os.path.join(root, 'home', 'mmd', 'Documents')
        sandisk = samsung

# Special configuration for lab workstation
if device_name == lab_workstation:
    print("Server Workstation")
    subfolder = "Declich"
    sandisk =  os.path.join(mnt, 'd', subfolder)
    thesis_folder = os.path.join(sandisk, "GitHub", "UMD")
    samsung = sandisk 

# Get current working directory
current_directory = os.getcwd()

# Additional configuration for Ubuntu desktop
if device_name == 'mmd-MS-7D98':
    ubuntu_media = os.path.join(root, "media", "mmd")
    sandisk = os.path.join(ubuntu_media, "Extreme SSD")
    samsung = os.path.join(ubuntu_media, "Samsung_T5")

# Print current path configuration for verification
print("this is the current directory:  " + current_directory)
print("This is the thesis folder, where all the code is located:  " + thesis_folder)    
print("Questo è samsung:" + samsung)

# Configure paths to slide datasets
lscc_string = os.path.join(samsung, "Slides", "PKG - CPTAC-LSCC_v10","LSCC")
luad_string = os.path.join(samsung, "Slides", "PKG - CPTAC-LUAD_v12","LUAD")

# Configure paths for different types of data bags
uni_bags_path = os.path.join(sandisk,'work', 'bags', 'uni')          # UNI bags path
phikon_bags_path = os.path.join(sandisk,'bags', 'histossl')          # Phikon bags path

# Configure working directories and model paths
working_directory = os.path.join(samsung, thesis_folder)
int_models_path = os.path.join(working_directory, "int-models")      # Intermediate models path

# Configuration for reproducibility
default_seed = "123456"                                              # Default random seed
random_path = os.path.join(working_directory,"projects")             # Path for project files