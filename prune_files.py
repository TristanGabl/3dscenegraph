import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def prune_files(folder_path):
    # Get all files in the folder
    all_files = os.listdir(folder_path)
    
    
    jpg_files_numbers = {os.path.splitext(f)[0].split('_')[1] for f in all_files if f.endswith('.jpg') and f.startswith('frame_')}
    
    # Iterate through all files and delete json and png files without jpg equivalent
    for file in all_files:
        file_name, file_ext = os.path.splitext(file)
        if not file_name.startswith(('frame_', 'depth_', 'conf_')):
            continue
        file_name, file_name_number = file_name.split('_')
        if file_ext in ['.json', '.png'] and file_name_number not in jpg_files_numbers:
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# Example usage
if len(sys.argv) != 2:
    print("Usage: python prune_files.py <folder_path>")
    sys.exit(1)

folder_path = sys.argv[1]
prune_files(folder_path)