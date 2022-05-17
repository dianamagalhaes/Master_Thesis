import os


def create_dir_recursively(path):
    """
    Creating folders and directories recursively,
    to assure that the path exists. If the directory
    already exists the method does not overwrite.

    Parameters
    ----------

    path: str
        Full path to the file ought to be saved.
    """
    parts = path.split(os.sep)
    path = os.sep

    for part in parts:
        path = os.path.join(path, part)
        if not os.path.exists(path):
            os.mkdir(path)
