import os
import os.path as pth
import shutil
import glob
import random
import torch
import numpy as np
from abc import ABC, abstractmethod
import json
import subprocess
from collections import OrderedDict
from sbs.sbs_graph import SBSGraph
from sbs.sbs_graph_nodes import SBSNodeParameter
from utils import Timer, read_image, write_image


class Sampler(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def size(self):
        pass


class RandomSampler(Sampler):
    def __init__(self, min_val, max_val, default_val=None, is_discrete=False):
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default_val if default_val is not None else self.min_val
        self.is_discrete = is_discrete
        if is_discrete:
            self.func = random.randint
        else:
            self.func = random.uniform

    def sample(self):
        val = []
        for min_v, max_v in zip(self.min_val, self.max_val):
            val.append(self.func(min_v, max_v))
        return val

    def size(self):
        return len(self.min_val)


class GaussianRandomSampler(Sampler):
    def __init__(self, min_val, max_val, mean_val, std_val, default_val=None, is_discrete=False):
        self.min_val = min_val
        self.max_val = max_val
        self.mean_val = mean_val
        self.std_val = std_val
        self.default_val = default_val if default_val is not None else mean_val
        self.is_discrete = is_discrete

    def get_sample_np(self):
        val = np.random.normal(self.mean_val, self.std_val)
        val = np.clip(val, self.min_val, self.max_val)
        return val

    def sample(self):
        val = self.get_sample_np()
        if self.is_discrete:
            val = np.rint(val).astype(np.int)
        return val.tolist()

    def size(self):
        return len(self.mean_val)


class ParameterNormalizer:
    def __init__(self, min_, max_):
        self.min_ = min_.clone()
        self.max_ = max_.clone()
        self.range = self.max_ - self.min_

    def normalize(self, x):
        return torch.nan_to_num((x - self.min_) / self.range)

    def denormalize(self, x):
        return x * self.range + self.min_


class ParameterStandarizer:
    def __init__(self, mean, std):
        self.mean = mean.clone()
        self.std = std.clone()

    def normalize(self, x):
        return torch.nan_to_num((x - self.mean) / self.std)

    def denormalize(self, x):
        return x * self.std + self.mean


class ParameterRegularizer:
    def __init__(self, min_, max_):
        self.min_ = min_.clone()
        self.max_ = max_.clone()

    def regularize(self, x):
        return torch.clamp(x, self.min_, self.max_)

    def regularize_(self, x):
        x.clamp_(self.min_, self.max_)

    def check_valid(self, x):
        all_min = x >= self.min_
        if not torch.all(all_min):
            l = x.shape[1]
            for k in range(l):
                i, j = x[0, k], self.min_[0, k]
                if i < j:
                    print(f'For {k}th params: {i} < {j}')
            raise RuntimeError('Invalid parameters')

        all_max = x <= self.max_
        if not torch.all(all_max):
            l = x.shape[1]
            for k in range(l):
                i, j = x[0, k], self.max_[0, k]
                if i > j:
                    print(f'For {k}th params: {i} > {j}')
            raise RuntimeError('Invalid parameters')


class SBSGenerators:
    @staticmethod
    def get_params():
        pass

    @staticmethod
    def get_normalizer(params, batch_size, device):
        min_, max_ = [], []
        for param_name, param_sampler in params.items():
            min_val = param_sampler.min_val
            max_val = param_sampler.max_val
            min_.extend(min_val)
            max_.extend(max_val)

        min_tensor = torch.as_tensor(min_, dtype=torch.float64, device=device)
        max_tensor = torch.as_tensor(max_, dtype=torch.float64, device=device)
        min_tensor = min_tensor.expand((batch_size, -1))
        max_tensor = max_tensor.expand((batch_size, -1))

        return ParameterNormalizer(min_tensor, max_tensor)

    @staticmethod
    def get_standarizer(params, batch_size, device):
        mean, std = [], []
        for param_name, param_sampler in params.items():
            mean_v = param_sampler.mean_val
            std_v = param_sampler.std_val
            mean.extend(mean_v)
            std.extend(std_v)

        mean_tensor = torch.as_tensor(mean, dtype=torch.float64, device=device)
        std_tensor = torch.as_tensor(std, dtype=torch.float64, device=device)
        mean_tensor = mean_tensor.expand((batch_size, -1))
        std_tensor = std_tensor.expand((batch_size, -1))
        return ParameterStandarizer(mean_tensor, std_tensor)

    @staticmethod
    def get_regularizer(params, batch_size, device):
        min_, max_ = [], []
        for param_name, param_sampler in params.items():
            min_val = param_sampler.min_val
            max_val = param_sampler.max_val
            min_.extend(min_val)
            max_.extend(max_val)

        min_tensor = torch.as_tensor(min_, dtype=torch.float64, device=device)
        max_tensor = torch.as_tensor(max_, dtype=torch.float64, device=device)
        min_tensor = min_tensor.expand((batch_size, -1))
        max_tensor = max_tensor.expand((batch_size, -1))

        return ParameterRegularizer(min_tensor, max_tensor)

    def __init__(self, graph_filename, graph_name, sat_dir, image_res):
        self.grah_filename = graph_filename
        self.graph_name = graph_name
        self.sat_dir = sat_dir
        self.image_res = image_res

        # load graph
        graph = SBSGraph.load_sbs(graph_name=graph_name, filename=graph_filename,
                                  sbs_resource_dir=os.path.join(sat_dir, 'resources', 'packages'), res=image_res)

        self.graph = graph

        # load parameters
        self.params = self.get_params()

        # collect all nodes
        node_list = []
        output_name_list = []
        for node in graph.nodes:
            if node.type == 'Unsupported':
                node_list.append(node)
                output_name_list.append(node.get_connected_child_inputs()[0].node.graph_output.name)

        self.output_name_list = output_name_list

        self.n_nodes = len(node_list)
        print("{} nodes are found".format(self.n_nodes))

        # check params in nodes
        node_params_list = []
        for i, node in enumerate(node_list):
            node_params = {}
            for param in node.params:
                node_params[param.name] = param

            # add unregistered params to nodes
            unregistered_param_names = set(self.params) - set(node_params)
            print(f"In Node {i}, found registered params:{set(node_params)}")
            print(f"In Node {i}, found unregistered params: {unregistered_param_names}")

            for param_name in unregistered_param_names:
                param = SBSNodeParameter(name=param_name, val=list(self.params[param_name].default_val), name_xml=param_name)
                node.add_param(param)
                node_params[param_name] = param

            node_params_list.append(node_params)

        self.node_params_list = node_params_list

    @staticmethod
    def save_params(all_params, all_image_names, output_dir, i_batch):
        assert len(all_params) == len(all_image_names)
        data = dict()
        data['labels'] = []

        for params, image_name in zip(all_params, all_image_names):
            data['labels'].append([image_name, params])

        with open(pth.join(output_dir, f'dataset{i_batch}.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)

    @staticmethod
    def combine_params(input_path, n_batch, move_to_folder=None):
        assert (n_batch >= 1)
        if move_to_folder is not None:
            os.makedirs(move_to_folder, exist_ok=True)

        data_path = pth.join(input_path, 'dataset0.json')
        with open(data_path) as f:
            data = json.load(f)

        if move_to_folder is not None:
            shutil.move(data_path, pth.join(move_to_folder, f'dataset0.json'))

        for i in range(1, n_batch):
            data_path = pth.join(input_path, f'dataset{i}.json')
            with open(data_path) as f:
                data_i = json.load(f)
            data['labels'].extend(data_i['labels'])

            if move_to_folder is not None:
                shutil.move(data_path, pth.join(move_to_folder, f'dataset{i}.json'))

        output_path = pth.join(input_path, 'dataset.json')
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    @staticmethod
    def change_param_directory(json_file, dir_name):
        with open(json_file) as f:
            data = json.load(f)

        n_samples = len(data['labels'])

        for i in range(n_samples):
            file_path = data['labels'][i][0]
            basename = pth.basename(file_path)
            new_file_path = f'{dir_name}/{basename}'
            data['labels'][i][0] = new_file_path

        # save a copy just in case
        shutil.copyfile(json_file, json_file + '.bak')

        with open(json_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def sample(self, output_dir, n_samples, vis_every):
        n_samples = n_samples // self.n_nodes * self.n_nodes
        n_batch = n_samples // self.n_nodes

        timer = Timer()
        timer.begin("Begin Sampling")

        for i in range(n_batch):
            params_list = []

            # sample parameters
            for j in range(self.n_nodes):
                params = []
                for param_name, param_sampler in self.params.items():
                    val = param_sampler.sample()
                    self.node_params_list[j][param_name].val = val
                    params.extend(val)

                params_list.append(params)

            # save sbs
            output_graph_filename = pth.join(output_dir, f'tmp{i}.sbs')
            self.save_graph(output_graph_filename)
            image_names_list = self.save_sample(output_graph_filename, output_dir, i)
            self.save_params(params_list, image_names_list, output_dir, i)

            if i % vis_every == 0 or i == n_batch - 1:
                timer.end('Generated {}/{} samples'.format((i+1)*self.n_nodes, n_samples))
                timer.begin()

        # combine parameter json into one file
        self.combine_params(output_dir, n_batch, move_to_folder=pth.join(output_dir, 'params'))

        # move generated sbs and sbsar files to an sbs folder
        sbs_files = glob.glob(pth.join(output_dir, '*.sbs')) + glob.glob(pth.join(output_dir, '*.sbsar'))
        sbs_out_dir = pth.join(output_dir, 'sbs')
        if pth.exists(sbs_out_dir):
            shutil.rmtree(sbs_out_dir)
        os.makedirs(sbs_out_dir)
        for sbs_file in sbs_files:
            shutil.move(sbs_file, sbs_out_dir)

    def sample_with_json(self, output_dir, json_file):
        if isinstance(json_file, str):
            with open(json_file) as f:
                params = json.load(f)['labels']
        else:
            params = json_file

        n_samples = len(params)
        n_batch = n_samples // self.n_nodes

        timer = Timer()
        timer.begin("Begin Sampling")

        for i in range(n_batch):
            image_names = []
            for j in range(self.n_nodes):
                idx = i*self.n_nodes + j

                # set parameters
                s = 0
                for param_name, param_sampler in self.params.items():
                    r = param_sampler.size()
                    val = [int(np.rint(x)) if param_sampler.is_discrete else x for x in params[idx][1][s:s + r]]
                    self.node_params_list[j][param_name].val = val
                    s += r

                # record image name
                image_name = pth.join(output_dir, pth.basename(params[idx][0]))
                image_names.append(image_name)

            # save sbs
            output_graph_filename = pth.join(output_dir, f'tmp{i}.sbs')
            self.save_graph(output_graph_filename)
            self.save_sample(output_graph_filename, output_dir, i, image_names)

        # move generated sbs and sbsar files to an sbs folder
        sbs_files = glob.glob(pth.join(output_dir, '*.sbs')) + glob.glob(pth.join(output_dir, '*.sbsar'))
        sbs_out_dir = pth.join(output_dir, 'sbs')
        if pth.exists(sbs_out_dir):
            shutil.rmtree(sbs_out_dir)
        os.makedirs(sbs_out_dir)
        for sbs_file in sbs_files:
            shutil.move(sbs_file, sbs_out_dir)

    # save sbs graph back to an sbs file
    def save_graph(self, output_graph_filename):
        self.graph.clear_xml()  # needed for legacy reasons
        self.graph.clamp_trainable_node_params()  # ensure optimized parameters are in the valid parameter ranges
        self.graph.save_sbs(filename=output_graph_filename)

    # cook and output images
    def save_sample(self, input_graph_filename, output_dir, i_batch, image_names=None):
        tmp_output_dir = pth.join(output_dir, 'tmp')
        os.makedirs(tmp_output_dir, exist_ok=True)

        command_cooker = (
            f'"{os.path.join(self.sat_dir, "sbscooker")}" '
            f'--inputs "{input_graph_filename}" '
            f'--alias "sbs://{os.path.join(self.sat_dir, "resources", "packages")}" '
            f'--output-path {{inputPath}}')

        completed_process = subprocess.run(command_cooker, shell=True, capture_output=True, text=True)
        if completed_process.returncode != 0:
            raise RuntimeError(f'Error while running sbs cooker:\n{completed_process.stderr}')
        # import pdb; pdb.set_trace()

        cooked_input_graph_filename = pth.splitext(input_graph_filename)[0] + '.sbsar'
        image_format = 'png'
        command_render = (
            f'"{os.path.join(self.sat_dir, "sbsrender")}" render '
            f'--inputs "{cooked_input_graph_filename}" '
            f'--input-graph "{self.graph_name}" '
            f'--output-format "{image_format}" '
            f'--output-path "{tmp_output_dir}" '
            f'--output-name "{{outputNodeName}}"')

        completed_process = subprocess.run(command_render, shell=True, capture_output=True, text=True)
        if completed_process.returncode != 0:
            raise RuntimeError(f'Error while running sbs render:\n{completed_process.stderr}')

        image_list = [pth.join(tmp_output_dir, f'{output_name}.png') for output_name in self.output_name_list]

        assert len(image_list) == self.n_nodes

        image_names_list = []
        dir_name = pth.basename(output_dir)

        for i, image_filename in enumerate(image_list):
            if image_names is None:
                image_name = pth.join(output_dir, '{:06d}.png'.format(i_batch*self.n_nodes + i))
            else:
                image_name = image_names[i]
            shutil.move(image_filename, image_name)

            # convert to 8bit
            self.convert(image_name)

            image_name = f'{dir_name}/{pth.basename(image_name)}'
            image_names_list.append(image_name)

        os.rmdir(tmp_output_dir)

        return image_names_list

    @staticmethod
    def convert(image_file):
        im = read_image(image_file)
        write_image(image_file, im)


class ArcPavement(SBSGenerators):
    @staticmethod
    def get_params():
        params = OrderedDict([('pattern_amount', RandomSampler((4,), (32,), (12,), True)),
                              ('arcs_amount', RandomSampler((4,), (20,), (14,), True)),
                              ('pattern_scale', RandomSampler((0.9,), (1.0,), (1.0,), False)),
                              ('pattern_width', RandomSampler((0.7,), (0.9,), (0.8,), False)),
                              ('pattern_height', RandomSampler((0.8,), (1.0,), (0.9,), False)),
                              ('pattern_width_random', RandomSampler((0.0,), (0.2,), (0.0,), False)),
                              ('pattern_height_random', RandomSampler((0.0,), (0.2,), (0.0,), False)),
                              ('global_pattern_width_random', RandomSampler((0.0,), (0.2,), (0.0,), False)),
                              ('pattern_height_decrease', RandomSampler((0.0,), (0.5,), (0.25,), False)),
                              ('color_random', RandomSampler((0.0,), (1.0,), (0.0,), False)),
                              ])
        return params


generator_lookup_table = {'arc_pavement': ArcPavement}