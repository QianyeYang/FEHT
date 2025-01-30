import argparse, os, feht, torch
from loguru import logger
from feht.utils.time import current_time


def main():
    parser = argparse.ArgumentParser("Higest level configuration settings")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="assign gpu device", required=False)
    parser.add_argument("-m", "--model", type=str, default='feht-l2-pretrained', help="name of the model need to use", required=True)
    parser.add_argument("-c", "--csv", default='', help="csv index for ", required=True)
    parser.add_argument("-o", "--output", default='', help="output path", required=True)
    args = parser.parse_args()

    # sanity checks
    assert os.path.isfile(args.csv), \
        f"CSV file {args.csv} not found! Please check the file."
    assert args.csv.endswith('.csv'), \
        f"CSV file {args.csv} is not in csv format! Please check the file."
    
    if args.output == '':
        args.output = "./" + current_time()
        logger.info(f">>> Output path is not specified. Saving the output to {args.output}")
    os.makedirs(args.output, exist_ok=True)
    if len(os.listdir(args.output)) > 0:
        logger.warning(f">>> Output directory {args.output} is not empty! Files may be overwritten.")

    # load the model
    model = feht.Model(name=args.model)
    model = model.to(torch.device(f"cuda:{args.gpu}"))

    # set dataloader
    

