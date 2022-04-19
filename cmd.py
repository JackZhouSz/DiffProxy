import os

cmd_type = 'train'
assert cmd_type in ['synthesize', 'combine', 'build', 'train']

generator_name = 'arc_pavement'

if cmd_type == 'synthesize':
    cmd = 'python synthesis.py ' \
          '--data_path=./data/sbs ' \
          f'--generator_name={generator_name} ' \
          '--index=0 ' \
          '--n_samples=512 '

elif cmd_type == 'combine':
    cmd = 'python combine.py ' \
          '--data_path=./data/sbs ' \
          f'--sub_folders={generator_name}0,{generator_name}1,{generator_name}2 ' \
          f'--target_folder={generator_name} ' \

elif cmd_type == 'build':
    cmd = f'python dataset_tool.py --source=./data/sbs/{generator_name} ' \
          f'--dest=./data/train/{generator_name}_300k.zip'

elif cmd_type == 'train':
    cmd = 'python train.py ' \
        '--cfg stylegan2 ' \
        '--outdir=./training-runs ' \
        f'--data=./data/train/{generator_name}_300k.zip ' \
        f'--generator_name={generator_name} ' \
        '--gpus=1 ' \
        '--batch=8 ' \
        '--gamma=10 ' \
        '--map-depth=4 ' \
        '--glr=0.0025 ' \
        '--dlr=0.001 ' \
        '--cbase=16384 ' \
        '--snap=50 ' \
        '--cond=true ' \
        '--aug=noaug ' \
        '--kimg=20000 ' \
        '--metrics=none ' \
        '--mirror=false ' \
        '--norm_type=norm ' \
        '--no_gan=True ' \
        '--cond_d=True ' \
        # '--dry-run'
else:
    raise NotImplementedError

print('*' * 50)
print(cmd)
os.system(cmd)