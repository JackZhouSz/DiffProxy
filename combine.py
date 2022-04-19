import argparse
from sbs.sbs_generators import *


def combine_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sub_folders', type=str, required=True)
    parser.add_argument('--target_folder', type=str, required=True)
    args = parser.parse_args()

    sub_folders = args.sub_folders.split(",")

    args.target_folder = pth.join(args.data_path, args.target_folder)
    os.makedirs(args.target_folder, exist_ok=True)

    for i, folder in enumerate(sub_folders):
        # move data folder
        os.rename(pth.join(args.data_path, folder), pth.join(args.target_folder, folder))
        # copy and rename json
        shutil.copyfile(pth.join(args.target_folder, folder, 'dataset.json'),
                        pth.join(args.target_folder, f'dataset{i}.json'))

    # combine json
    n_subfolder = len(sub_folders)
    SBSGenerators.combine_params(args.target_folder, n_batch=n_subfolder)

    # delete json
    for i in range(n_subfolder):
        os.remove(pth.join(args.target_folder, f'dataset{i}.json'))


if __name__ == '__main__':
    combine_data()