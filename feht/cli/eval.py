import argparse, importlib, os
import feht.utils.io as uio 
from loguru import logger


def main():
    parser = argparse.ArgumentParser("Higest level configuration settings")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="assign gpu device", required=False)
    parser.add_argument("-p", "--path", type=str, help="path to the experiment folder", required=True)
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="batch size", required=False)
    parser.add_argument("-c", "--checkpoint", default='latest' ,type=str, help="specify the checkpoint number", required=False)
    parser.add_argument("-v", "--validation", action='store_true', help="validate the model", required=False)
    parser.add_argument("-e", "--entry", default='', help="fexibly call a function in the arch", required=False)
    args = parser.parse_args()

    config = uio.load_yaml(
        os.path.join(args.path, 'config.yaml')
    )

    if isinstance(config['gpu_ids'], str):
        logger.info(f">>> GPU was used as {config['gpu_ids']} in training")
        pass
    else:
        config['gpu_ids'][0] = args.gpu

    if args.batch_size:
        config['batch_size'] = args.batch_size

    arch_module, arch_attr = config['arch']
    arch_module = f"feht.arch.{arch_module}"
    Arch = getattr(
        importlib.import_module(arch_module), 
        arch_attr
    )

    arch = Arch(config)
    arch.load_state(args.checkpoint)

    if args.validation:  
        # run on validation set, for debugging purposes
        arch.validate()
    elif args.entry:
        # run a specific function in the arch
        getattr(arch, args.entry)()
    else:
        arch.inference()