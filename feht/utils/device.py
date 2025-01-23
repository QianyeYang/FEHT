import torch


def get_gpu_info():
    '''will be useful when computing on HPC :) '''
    gpu_id = torch.cuda.current_device()
    gpu_type = torch.cuda.get_device_name(gpu_id)
    print(f'>>> Computing on GPU: {gpu_type} <<<')


if __name__ == "__main__":
    get_gpu_info()