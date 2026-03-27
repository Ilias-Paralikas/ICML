import os
def get_version_folder(root):
    os.makedirs(root,exist_ok=True)
    index_file = os.path.join(root,'index.txt')

    if not os.path.exists(index_file):
            index = 0
    else:
        with open(index_file, 'r') as f:
            index = int(f.read())

    index = index+1
    with open(index_file, 'w') as f:
        f.write(str(index))

    index_folder = os.path.join(root,str(index))
    os.makedirs(index_folder)

    return index_folder