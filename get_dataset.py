import os, sys

dir_tree = './MillionSongSubset'

for dir_path, dir_names, file_names in os.walk(dir_tree):
    for file_name in file_names:
        try:
            os.rename(os.path.join(dir_path, file_name), os.path.join(dir_tree, file_name))
        except OSError:
            print ("Could not move %s " % os.join(dir_path, file_name))
