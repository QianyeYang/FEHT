import torch, sys, os
from datetime import datetime
from glob import glob
from loguru import logger
from icecream import ic

from abc import ABC, abstractmethod
from typing import Any
import src.utils.io as uio
from src.utils.time import current_time
from src import utils

from torch.optim.lr_scheduler import StepLR, LinearLR
import torch.distributed as dist


class BaseArch(ABC):
    def __init__(self, config: dict|str) -> None:
        """
        Base class for all architectures.
        """

        # load configuration
        if isinstance(config, str):
            self.config = uio.load_yaml(config)
        else:
            self.config = config

        # model variables
        self.net = None
        self.optimizer = None
        self.scheduler = None

        # set logger
        self.log_dir = self.get_log_dir()
        self.logger = self.get_logger()
        self.logger.info(f'>>> Configurations:')
        for key, value in self.config.items():
            self.logger.info(f'>>> {key}: {value}')
        
        # training variables
        self._epoch : int = 0
        self._val_warmup : int = self.config.get('val_warmup', 0)
        self.iteration : int = 0
        self.phase = 'train'
        self.epoch_loss = 0

        # utils
        self.vis_dir = None
        self.res_dir = None

    
    @property
    def exp_name(self):
        return self.config['exp_name']


    @property
    def epoch(self):
        return self._epoch
    @epoch.setter
    def epoch(self, value):
        if value == 0:
            self._epoch = value 
            return
        if self.phase == 'train' and self._epoch > self._val_warmup:
            self.validate()
        if not value % self.save_frequency and self.phase != 'test':
            self.save_state()
        self._epoch = value
            

    @abstractmethod
    def parse_config(self, config) -> None:
        '''
        Load the configuration file.
        '''
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.save_frequency = config['save_frequency']
        self.device = self.get_device()



    @abstractmethod
    def set_dataloader(self) -> None:
        '''
        Set the dataloader.
        '''
        pass

    
    @abstractmethod
    def train(self) -> None:
        '''
        Training process
        '''
        self.logger.info(f'>>> Start training...')
        self.to_train_mode()
        self.save_configure()


    @abstractmethod
    def validate(self) -> None:
        '''
        Validation process
        '''
        self.logger.info(f'>>> Start validation...')
        self.to_val_mode()


    @abstractmethod
    def inference(self) -> None:
        '''
        Inference process
        '''
        self.logger.info(f'>>> Start inference...')
        self.vis_dir = os.path.join(self.log_dir, f"epoch-{self.epoch}-results", 'visual.results')
        os.makedirs(self.vis_dir, exist_ok=True)

        self.res_dir = os.path.join(self.log_dir, f"epoch-{self.epoch}-results", 'quant.Results')
        os.makedirs(self.res_dir, exist_ok=True)
        self.to_test_mode()


    @abstractmethod
    def loss(self):
        '''
        Define loss function
        '''
        pass


    def set_lr_scheduler(self, scheduler: str|None) -> None:
        if scheduler is None: return
        elif scheduler == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)
        elif scheduler == 'linear':
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1,
                end_factor=0.25,
                total_iters=self.num_epochs
                )
        else:
            raise NotImplementedError


    def to_train_mode(self) -> None:
        '''
        Set the model to train mode.
        Check before call model.train() to save time.
        '''
        logger.configure(
            handlers=self.__get_logger_handler('TRAIN')
        )
        self.phase = 'train'
        if self.net.training:
            logger.info(f'>>> Stay in train mode')
        else:
            logger.info(f'>>> Switch to train mode')
            self.net.train()


    def to_val_mode(self) -> None:
        '''
        Set the model to validation mode
        '''
        logger.configure(
            handlers=self.__get_logger_handler('VALIDATION')
        )
        self.phase = 'val'
        if self.net.training:
            logger.info(f'>>> Switch to val mode')
            self.net.eval()
        else:
            logger.info(f'>>> Stay in val mode')


    def to_test_mode(self) -> None:
        '''
        Set the model to test mode
        '''
        logger.configure(
            handlers=self.__get_logger_handler('TEST')
        )
        self.phase = 'test'
        if self.net.training:
            logger.info(f'>>> Switch to test mode')
            self.net.eval()
        else:
            logger.info(f'>>> Stay in test mode')


    def get_current_lr(self) -> float:
        '''
        Get the current learning rate
        '''
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


    def get_device(self) -> None:
        '''
        TODO: Need to consider when gpu_ids are set but not avaliable.
        '''
        device = self.config['gpu_ids'][0]
        if device == -1:
            return torch.device('cpu')
        elif device == 'x':
            return torch.device('cuda')
        else:
            return torch.device('cuda', device)
        

    def save_state(self, type: None | str = None) -> None:
        '''
        Save the model state.
        '''
        ckpt_path = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)

        if type is None:
            torch.save(self.net, os.path.join(ckpt_path, f'epoch-{self.epoch}.pt'))
            uio.save_checkpoint(
                model = self.net, 
                optimizer = self.optimizer, 
                epoch = self.epoch, 
                checkpoint_path = os.path.join(ckpt_path, f'epoch-{self.epoch}.pt')
            )
            logger.info(f'>>> Save model at epoch {self.epoch}')
        elif type == 'best':
            exist_best_models = glob(os.path.join(ckpt_path, 'best*.pt'))
            [os.remove(i) for i in exist_best_models]
            uio.save_checkpoint(
                model = self.net, 
                optimizer = self.optimizer, 
                epoch = self.epoch, 
                checkpoint_path = os.path.join(ckpt_path, f'best-epoch-{self.epoch}.pt')
            )
            logger.info(f'>>> Save best model at epoch {self.epoch}')
        else:
            raise NotImplementedError
        


    def load_state(self, num_epoch: int | str = 'best') -> None:
        '''
        Load the model state.
        '''
        if isinstance(num_epoch, int):
            num_epoch = str(num_epoch)

        if num_epoch == 'best':
            checkpoint_path = glob(os.path.join(self.log_dir, 'checkpoints', 'best*'))
            assert len(checkpoint_path) != 0, f"no best ckpt found in {self.log_dir}..."
            assert len(checkpoint_path) == 1, f"multiple best ckpt found in {self.log_dir}..."
            checkpoint_path = checkpoint_path[0]
        elif num_epoch == 'latest':
            checkpoint_path = glob(os.path.join(self.log_dir, 'checkpoints', 'epoch*.pt'))
            checkpoint_path = [i for i in checkpoint_path if 'best' not in i]
            assert len(checkpoint_path) > 0, f"no ckpt found in {self.log_dir}..."
            checkpoint_path.sort(
                key = lambda x: int(x.split('-')[-1].split('.')[0])
            )
            checkpoint_path = checkpoint_path[-1]
        else:
            checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'epoch-{num_epoch}.pt')
            assert os.path.exists(checkpoint_path), f"{checkpoint_path} not exists..."
        
        state_dict = uio.load_checkpoint(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.to_test_mode()  # default to test mode unless specified otherwise
        self.epoch = state_dict['epoch']
        self.logger.info(f'>>> Load model from best epoch {self.epoch}')


    def save_configure(self):
        utils.io.save_yaml(
            data = self.config, 
            path = os.path.join(self.log_dir, 'config.yaml')
            )


    def get_log_dir(self) -> str:
        '''
        Get the log directory.
        '''
        assert self.exp_name is not None, "exp_name should not be None."
        log_dir = os.path.join('./logs', self.exp_name)
        while os.path.exists(log_dir) and 'train.py' in sys.argv[0]:
            log_dir = os.path.join(
                './logs', 
                self.exp_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        return log_dir
    

    def get_logger(self) -> Any:
        '''
        Get the logger.
        '''
        logger.remove()
        logger_format = "{time:HH:mm:ss} {level} {message}"
        logger.add(sys.stdout, format=logger_format, level="INFO")
        self.logger_path = os.path.join(self.log_dir, f'{current_time()}.log')
        logger.add(
            self.logger_path,
            rotation="100 MB", 
            format=logger_format,
            level="INFO"
            )
        return logger
    

    def __get_logger_handler(self, phase: str) -> list[dict]:
        '''
        Get the logger handler.
        
        :param phase: A string indicating the phase.
        :type phase: str

        :return: dict
        :rtype: dict
        '''
        return [
            {
                "sink": self.logger_path,
                "rotation": "100 MB",
                "format": f"[{phase}] {{time:HH:mm:ss}} {{level}} {{message}}",
                "level": "INFO",
                # "colorize": True
            }, 
            {
                "sink": sys.stdout,
                "format": f"[{phase}] {{time:HH:mm:ss}} {{level}} {{message}}",
                "level": "INFO"
            }
        ]
    
