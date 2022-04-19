import argparse
from sbs.sbs_generators import *

sat_dir = 'C:/Program Files/Allegorithmic/Substance Automation Toolkit'


def synthesize_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--generator_name', type=str, required=True)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--index', type=str, default='')
    parser.add_argument('--image_res', type=int, default=256)
    parser.add_argument('--verify', type=bool, default=False)
    args = parser.parse_args()

    data_path = args.data_path
    generator_name = args.generator_name
    index = args.index
    output_dir = f'{generator_name}{index}'
    graph_name = 'generator'
    n_samples = args.n_samples
    vis_every = 50

    output_dir = pth.join(data_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    graph_filename = pth.join(data_path, f'{generator_name}.sbs')
    sampler = generator_lookup_table[generator_name](graph_filename, graph_name, sat_dir, args.image_res)
    sampler.sample(output_dir, n_samples=n_samples, vis_every=vis_every)

    # verify
    if args.verify:
        verify_output_dir = output_dir + '_verify'
        os.makedirs(verify_output_dir, exist_ok=True)
        sampler.sample_with_json(verify_output_dir, pth.join(output_dir, 'dataset.json'))


if __name__ == '__main__':
    synthesize_data()