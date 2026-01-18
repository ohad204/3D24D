import os
import glob

def delete_last_ckpt(root_dir):
    """Recursively deletes all files named 'last.ckpt' in the given root directory."""
    for file_path in glob.iglob(os.path.join(root_dir, '**', 'last.ckpt'), recursive=True):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Example usage
root_directory = "/home/lab/rahamiob/ohadr/projects/object_animation/output-ablation"  # Change this to your target directory
delete_last_ckpt(root_directory)
