import argparse, importlib, os
import feht.utils.io as uio 
from loguru import logger


parser = argparse.ArgumentParser("Higest level configuration settings")
parser.add_argument("-c", "--config", type=str, help="configuration path", required=True)
parser.add_argument("-g", "--gpu_override", help="GPU override", required=False)
args = parser.parse_args()


if __name__ == '__main__':

    exp_name, _ = os.path.splitext(os.path.basename(args.config))

    config = uio.load_yaml(args.config)
    config['exp_name'] = exp_name

    # GPU override:
    # If the user specifies a GPU, override the config file
    # But will not change the config file
    # This is a solution for adding flexibility for the GPU configuration
    # And be useful for running on clusters which cannot assign GPU.
    if args.gpu_override == 'x':
        logger.info(f"GPU override -> Device {args.gpu_override}")
        config['gpu_ids'] = args.gpu_override
    elif args.gpu_override is not None:
        logger.info(f"GPU override -> Device {args.gpu_override}")
        config['gpu_ids'][0] = int(args.gpu_override)
    else: pass
        
        
    arch_module, arch_attr = config['arch']
    arch_module = f"feht.arch.{arch_module}"
    Arch = getattr(
        importlib.import_module(arch_module), 
        arch_attr
    )
    arch = Arch(config)
    arch.train()
    