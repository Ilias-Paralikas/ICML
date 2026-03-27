import os

def get_folder_size(folder_path,human_readable=True):

    def human_readable_size(size, decimal_places=2):
        # Convert bytes to KB, MB, GB, etc.
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.{decimal_places}f} {unit}"
            size /= 1024
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it's a symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    if human_readable:
        total_size = human_readable_size(total_size)
    return total_size

