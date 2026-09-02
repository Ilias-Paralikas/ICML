import os


def get_version_folder(root):
    """Create and return a fresh auto-incrementing run folder under ``root``.

    ``root/index.txt`` records the last-used index; the new folder is that + 1.
    If that folder somehow already exists (index.txt deleted, or out of sync with
    what's on disk), keep incrementing until an unused name is found — an existing
    run is never reused or clobbered. index.txt is only rewritten once the folder
    has been picked.
    """
    os.makedirs(root, exist_ok=True)
    index_file = os.path.join(root, 'index.txt')

    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = int(f.read())
    else:
        index = 0

    index += 1
    index_folder = os.path.join(root, str(index))
    while os.path.exists(index_folder):
        index += 1
        index_folder = os.path.join(root, str(index))

    os.makedirs(index_folder)
    with open(index_file, 'w') as f:
        f.write(str(index))

    return index_folder
