import os
import os.path as pth
import json
import numpy as np
import torch
import sbs.sbs_generators as sbs_generators
from sbs.differentiable_generator import StyleGANCond
from utils import read_image, write_image

sat_dir = 'C:/Program Files/Allegorithmic/Substance Automation Toolkit'


def compare2real():
    generator_name = 'arc_pavement'
    graph_filename = f'./data/sbs/{generator_name}16.sbs'
    model_path = './models/arc_pavement_20000_nogan.pkl'
    image_res = 256
    n_samples = 32

    out_dir = pth.join('./models/synthesis', generator_name)
    if pth.exists(out_dir) is False:
        os.makedirs(out_dir)

    sampler = sbs_generators.generator_lookup_table[generator_name](graph_filename, 'generator', sat_dir, image_res)
    sampler.sample(out_dir, n_samples=n_samples, vis_every=1)

    json_file = pth.join(out_dir, 'dataset.json')
    with open(json_file) as f:
        sampled_params = json.load(f)['labels']

    init = {'method': 'avg'}
    G = StyleGANCond(generator_name, model_path, init, model_type='norm')

    for params in sampled_params:
        image_name = pth.basename(params[0])
        real_np = read_image(pth.join(out_dir, image_name))
        parameters = params[1]
        p = torch.as_tensor(parameters, dtype=torch.float64, device=G.device).unsqueeze(0)

        comp = real_np.copy()
        G.set_params(p)
        fake = G().detach().squeeze().cpu().numpy()
        comp = np.concatenate((comp, fake), axis=1)
        write_image(pth.join(out_dir, f'fake_{image_name}'), fake)
        write_image(pth.join(out_dir, f'comp_{image_name}'), comp)


if __name__ == "__main__":
    compare2real()