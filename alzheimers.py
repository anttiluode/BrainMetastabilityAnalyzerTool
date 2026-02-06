import openneuro
import os

# Define target directory
target_dir = r"E:\DocsHouse\575 45 degree angle is real in brain\ds004504"
os.makedirs(target_dir, exist_ok=True)

# Download dataset ds004504 (latest version)
openneuro.download(dataset="ds004504", target_dir=target_dir)