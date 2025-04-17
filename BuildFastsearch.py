import argparse
from os import listdir, path, system

parser = argparse.ArgumentParser(description='Create fast search indices.')
parser.add_argument('libdir', type=str, help='The name of the directory containing the smi files to build for.')
args = parser.parse_args()

toplevel_directory = "../data_and_results"
lib_directory = path.join(toplevel_directory, args.libdir)

files = listdir(lib_directory)
relevant_files = list()
for f in files:
    if f[-4:] == ".smi":
        fast_index = f[:-4] + ".fs"
        if fast_index not in files:
            complete_filename = path.join(lib_directory, f)
            relevant_files.append(complete_filename)

for f in relevant_files:
    system(f"obabel {f} -ofs")
