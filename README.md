# Node Graph Optimization Using Differentiable Proxies
This repo contains code for "Node Graph Optimization Using Differentiable Proxies"
## Requirements
See requirements for StyleGAN2: https://github.com/NVlabs/stylegan2-ada. 
If you want to synthesize your own data for a Substance generator, please install MATch(http://match.csail.mit.edu/) in the `./sbs` folder.
## Generate Training Data (ArcPavement Generator as Example)
Note: if you want to generate training data for a Substance generator, please install MATch. 
If you want to train on your own data, you can skip this section and jump to training section.

All commands are listed in `cmd.py` for convenience. 

**Step 1**: Please specify a correct path `sat_dir` to substance automatic toolkit in `synthesis.py` and then synthesize training data:
```bash 
python synthesis.py --data_path=./data/sbs --generator_name=tile_generator --index=0 --n_samples=102400
python synthesis.py --data_path=./data/sbs --generator_name=tile_generator --index=1 --n_samples=102400
python synthesis.py --data_path=./data/sbs --generator_name=tile_generator --index=2 --n_samples=102400
```
102400x3 images will be synthesized into three folders [***tile_generator0***, ***tile_generator1***, ***tile_generator2***]. 

**Step 2**: Combine these generated images into one folder
```bash 
python combine.py --data_path=./data/sbs --sub_folders=tile_generator0,tile_generator1,tile_generator2 --target_folder=tile_generator
```
This will give you a single folder ***tile_generator*** with all the images and their parameters.

**Step 3**: Build the training dataset:
```bash 
python dataset_tool.py --source=./data/sbs/tile_generator --dest=./data/train/tile_generator_300k.zip
```
This will generate a final zip package for training located at `./data/train`.

## Train a Differentiable Proxy
If you want to train your own data, please make sure you have a json file that stores parameters for each image, 
and use `dataset_tool.py` to build a dataset for training. We include an example of such json file in `./data/example`. 

Please check `cmd.py` for the training command. Generally, there are two versions of networks: with GAN loss and without GAN loss, which is specified by an option `--no_gan`. The training log and intermediate results can be found in  `training-runs` subfolder. 
In order to run the code, please put the pretrained VGG19 model into `./pretrained` folder. The VGG19 model can be downloaded from here:  https://drive.google.com/file/d/13TTR61wQ3OVJUKk6P39HegoWBQ-_MUQq/view?usp=sharing.

Loss weights can be adjusted in the `./training/loss_gan.py` or `./training/loss_nogan.py` . 

## Validate a Differentiable Proxy
`generate.py` is a simple script describing how to use the trained differentiable proxy to generate images and compare to the real generator.
##  Train a New Substance Generator
To train a new generator, you need to implement an appropriate parameter sampling method for that generator. The code can be found in `./sbs/sbs_generator.py`. Define you own generator class by overwriting method `get_params()`. Two basic samplers are defined to sample parameters: `RandomSampler` (uniform sampling based on min and max values) and `GaussianRandomSampler` (Gaussian-like sampling based on mean and std, and the sampled values are clipped between min and max). What you need to do is to create your own class and then register that class in the variable `generator_lookup_table`. 

The samplers can be arbitrary for each parameter. `GaussianRandomSampler` is recommended if you have the reasonable statistics for that parameter. But note that during training phase, there is an option `--norm_type` specifies the normalization method applied on the parameters during the training phase. Currently, there are two options `norm` and `std`. The `norm` method rescales parameters between `0` and `1` based on min and max values, while `std` works by normalizing the parameters to `mean=0` and `std=1` so `std` option only works when all the samplers are `GaussianRandomSampler`.

After defining the sampling method, you should prepare a .sbs file similar to `./data/sbs/arc_pavement.sbs`. You can find a template file in `./data/sbs` named as `basic.sbs`. All you need to do is to open that file and replace the placeholder generator by your own generator and rename the sbs file properly. However, only 1 generator can be very slow when generating data. We recommend you copy-paste that "two-node" node graph 512 times or 1024 times. This will make the program synthesize 512 mask maps or 1024 mask maps for each step.

You should now be able to generate the data and train the networks. To test if the data is corrected generated, we recommend to first output a couple of mask maps (e.g. 128 or 256) to see if their general appearances looks reasonable. To see if your sampled parameters are correctly saved as a json file, you can specify `--verify=True` when you call `synthesis.py`. The program will re-generate all the mask maps to another folder based on the parameters your just sampled. This can help you examine whether the parameter you sampled and the generated mask map is a correct one-on-one mapping. Note `verify=True` should be only used for verification and prevent using it when you generating the whole dataset.

## Contact
If you have any question, feel free to contact yiwei.hu@yale.edu
