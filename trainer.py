#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext
import wandb

from datapipe.datasets import create_dataset

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

def configs_to_dict(configs):
    """
    将 configs 转为 dict，供 wandb.init 使用
    """
    if configs is None:
        return {}

    # 如果本身就是 dict
    if isinstance(configs, dict):
        return configs

    # 如果是 argparse.Namespace
    if hasattr(configs, "__dict__"):
        return vars(configs)

    # 如果是 dataclass
    try:
        from dataclasses import asdict
        return asdict(configs)
    except:
        pass

    # 兜底
    return dict(configs)

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module.'+key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            torch.cuda.empty_cache()

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)
            torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)  # wrap the network
        else:
            self.model = model

        # EMA
        # if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
        #     self.ema_model = deepcopy(model).cuda()
        #     self.ema_state = OrderedDict(
        #         {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
        #         )
        #     self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        # for ii in range(self.iters_start, self.configs.train.iterations):
        for ii in tqdm(
                range(self.iters_start, self.configs.train.iterations),
                initial=self.iters_start,
                total=self.configs.train.iterations,
                desc="Training"
            ):

            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']))

            # training phase
            self.training_step(data)

            # validation phase
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0:
                self.save_ckpt()
            # 分布式训练更新epcoch
            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            ckpt = {
                    'iters_start': self.current_iters,
                    'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                    'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                    'state_dict': self.model.state_dict(),
                    }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_state, ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def load_model(self, model, ckpt_path=None, tag='model', strict=True):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if strict:
            util_net.reload_model(model, ckpt)
        else:
            model.load_state_dict(ckpt, strict=False)
        if self.rank == 0:
            self.logger.info('Loaded Done')

class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        super().setup_optimizaton()
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )

    def build_model(self):
        super().build_model()
        # if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
        #     self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        # autoencoder
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            if self.configs.autoencoder.tune_decoder:
                self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
                if self.rank == 0:
                    num_params = 0
                    for key, value in autoencoder.named_parameters():
                        if 'decoder' in key or 'post_quant_conv' in key:
                            num_params += value.numel()
                        else:
                            value.requires_grad = False
                    self.logger.info(f'Finetuning Decoder module: {num_params/10**6:.2f}M...')
            else:
                self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
                self.freeze_model(autoencoder)
                autoencoder.eval()
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        if self.configs.autoencoder.params.lora_tune_decoder or self.configs.autoencoder.tune_decoder:
            self.freeze_model(self.model)

        # LPIPS metric
        if self.configs.train.get('use_lpips', False):
            if hasattr(self.configs, 'lpips'):
                lpips_net = self.configs.lpips.net
            else:
                lpips_net = 'vgg'
            if self.rank == 0:
                self.logger.info(f"Loading LIIPS Metric: {lpips_net}...")
            lpips_loss = lpips.LPIPS(net=lpips_net).to(f"cuda:{self.rank}")
            for params in lpips_loss.parameters():
                params.requires_grad_(False)
            lpips_loss.eval()
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling LPIPS Metric...")
                lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.lpips_loss = lpips_loss

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        if realesrgan is None:
            realesrgan = self.configs.data.get(phase, dict).type == 'realesrgan'
        if realesrgan and phase == 'train':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if not hasattr(self, 'use_sharpener'):
                self.use_sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
            if isinstance(self.configs.degradation.sf, int):
                sf = self.configs.degradation.sf
            else:
                assert len(self.configs.degradation.sf) == 2
                sf = random.uniform(*self.configs.degradation.sf)

            if self.configs.degradation.use_sharp:
                im_gt = self.use_sharpener(im_gt)

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob']
            if random.random() < self.configs.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.configs.degradation['second_order_prob']:
                # blur
                if random.random() < self.configs.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.configs.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.configs.degradation['gray_noise_prob2']
                if random.random() < self.configs.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.configs.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.configs.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)

            # resize back
            if self.configs.degradation.resize_back:
                out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')
                temp_sf = self.configs.degradation['sf']
            else:
                temp_sf = self.configs.degradation['sf']

            # clamp and round
            im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.configs.degradation['gt_size']
            im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, temp_sf)
            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        elif phase == 'val':
            offset = self.configs.train.get('val_resolution', 256)
            for key, value in data.items():
                h, w = value.shape[2:]
                if h > offset and w > offset:
                    h_end = int((h // offset) * offset)
                    w_end = int((w // offset) * offset)
                    data[key] = value[:, :, :h_end, :w_end]
                else:
                    h_pad = math.ceil(h / offset) * offset - h
                    w_pad = math.ceil(w / offset) * offset - w
                    padding_mode = self.configs.train.get('val_padding_mode', 'reflect')
                    data[key] = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        else:
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
                    )
            latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            if 'autoencoder' in self.configs:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (latent_resolution, ) * 2,
                    device=micro_data['gt'].device,
                    )
            if self.configs.model.params.cond_lq:
                model_kwargs = {'lq':micro_data['lq'],}
                if 'mask' in micro_data:
                    model_kwargs['mask'] = micro_data['mask']
            else:
                model_kwargs = None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)

            # make logging
            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach())

        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # grad zero
        self.model.zero_grad()

        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['loss'][jj].item(),
                            self.loss_mean['mse'][jj].item(),
                            )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                x0_pred = self.base_diffusion.decode_first_stage(
                        z0_pred,
                        self.autoencoder,
                        )
                self.logging_image(x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)

    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            indices = np.linspace(
                    0,
                    self.base_diffusion.num_timesteps,
                    self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']
                num_iters = 0
                if self.configs.model.params.cond_lq:
                    model_kwargs = {'lq':data['lq'],}
                    if 'mask' in data:
                        model_kwargs['mask'] = data['mask']
                else:
                    model_kwargs = None
                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,
                        ):
                    sample_decode = {}
                    if num_iters in indices:
                        for key, value in sample.items():
                            if key in ['sample', ]:
                                sample_decode[key] = self.base_diffusion.decode_first_stage(
                                        value,
                                        self.autoencoder,
                                        ).clamp(-1.0, 1.0)
                        im_sr_progress = sample_decode['sample']
                        if num_iters + 1 == 1:
                            im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                    num_iters += 1
                    tt -= 1

                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                            sample_decode['sample'] * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ycbcr=self.configs.train.val_y_channel,
                            )
                    mean_lpips += self.lpips_loss(
                            sample_decode['sample'],
                            im_gt,
                            ).sum().item()

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')

                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    self.logging_image(
                            im_sr_all,
                            tag='progress',
                            phase=phase,
                            add_global_step=False,
                            nrow=len(indices),
                            )
                    if 'gt' in data:
                        self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                    self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)

            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
                self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips, tag='LPIPS', phase=phase, add_global_step=True)

            self.logger.info("="*100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

class TrainerDifIRLPIPS(TrainerDifIR):
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        loss_coef = self.configs.train.get('loss_coef')
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        # diffusion loss
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                    self.autoencoder,
                    ) # f16
            self.current_x0_pred = x0_pred.detach()

            # lpips loss
            losses["lpips"] = self.lpips_loss(
                    x0_pred,
                    micro_data['gt'],
                    ).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)
            losses["lpips"] *= loss_coef[1]

            if loss_coef[0] > 0:    # calculate mse in latent space
                losses["mse"] *= loss_coef[0]
            else:                   # calculate mse in pixel space
                assert loss_coef[2] > 0
                losses["mse"] = mean_flat((x0_pred - micro_data['gt']) ** 2)
                losses["mse"] *= loss_coef[2]

            assert losses["mse"].shape == losses["lpips"].shape
            if flag_nan:
                losses["loss"] = losses["mse"]
            else:
                losses["loss"] = losses["mse"] + losses["lpips"]
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    assert value.shape == mask.shape
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, MSE/LPIPS: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['mse'][jj].item(),
                            self.loss_mean['lpips'][jj].item(),
                            )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                self.logging_image(self.current_x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)

class TrainerDifadapter(TrainerDifIR):

    def build_model(self):
        super().build_model()
        # load autoencoder wrapper if provided
        if self.configs.get("autoencoderwrapper", None) is not None:
            params = OmegaConf.to_container(
                self.configs.autoencoderwrapper.params, resolve=True
            )
            params["base_ae"] = self.autoencoder
            autoencoder_wrapper = util_common.get_obj_from_str(
                self.configs.autoencoderwrapper.target
            )(**params)
            autoencoder_wrapper = autoencoder_wrapper.cuda()

            # if self.configs.autoencoderwrapper.params.get("shared_encoder", None) is not None:
            #     ckpt_path = self.configs.autoencoderwrapper.params.shared_encoder.get("ckpt_path", None)
            #     if ckpt_path is not None:
            #         if self.rank == 0:
            #             self.logger.info(f"Loading AutoEncoder Wrapper shared encoder from {ckpt_path}...")
            #         self.load_model(autoencoder_wrapper.shared_encoder, ckpt_path, tag="shared_encoder")

            if self.configs.autoencoderwrapper.params.get("private_encoder", None) is not None:
                ckpt_path = self.configs.autoencoderwrapper.params.private_encoder.get("ckpt_path", None)
                if ckpt_path is not None:
                    if self.rank == 0:
                        self.logger.info(f"Loading AutoEncoder Wrapper private encoder from {ckpt_path}...")
                    self.load_model(autoencoder_wrapper.private_encoder, ckpt_path, tag="private_encoder")

            if self.configs.autoencoderwrapper.params.get("decoder_adapter", None) is not None:
                ckpt_path = self.configs.autoencoderwrapper.params.decoder_adapter.get("ckpt_path", None)
                if ckpt_path is not None:
                    if self.rank == 0:
                        self.logger.info(f"Loading AutoEncoder Wrapper decoder adapter from {ckpt_path}...")
                    self.load_model(autoencoder_wrapper.decoder_adapter, ckpt_path, tag="decoder_adapter")

            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder wrapper...")
                autoencoder_wrapper = torch.compile(
                    autoencoder_wrapper, mode=self.configs.train.compile.mode
                )
                if self.rank == 0:
                    self.logger.info("Compiling Done")

            self.autoencoder = autoencoder_wrapper

        # load discriminator if provided
        # if self.configs.get("discriminator_shared", None) is not None:
        #     params = OmegaConf.to_container(
        #         self.configs.discriminator_shared.params, resolve=True
        #     )
        #     discriminator_shared = util_common.get_obj_from_str(
        #         self.configs.discriminator_shared.target
        #     )(**params)
        #     discriminator_shared = discriminator_shared.cuda()

        #     ckpt_path = self.configs.discriminator_shared.get("ckpt_path", None)
        #     if ckpt_path is not None:
        #         if self.rank == 0:
        #             self.logger.info(f"Loading Discriminator Shared from {ckpt_path}...")
        #         self.load_model(discriminator_shared, ckpt_path, tag="discriminator_shared")

        #     if self.configs.train.compile.flag:
        #         if self.rank == 0:
        #             self.logger.info("Begin compiling discriminator shared...")
        #         discriminator_shared = torch.compile(
        #             discriminator_shared, mode=self.configs.train.compile.mode
        #         )
        #         if self.rank == 0:
        #             self.logger.info("Compiling Done")

        #     self.discriminator_shared = discriminator_shared

        # load private discriminator if provided
        if self.configs.get("discriminator_private", None) is not None:
            params = OmegaConf.to_container(
                self.configs.discriminator_private.params, resolve=True
            )
            discriminator_private = util_common.get_obj_from_str(
                self.configs.discriminator_private.target
            )(**params)
            discriminator_private = discriminator_private.cuda()

            ckpt_path = self.configs.discriminator_private.get("ckpt_path", None)
            if ckpt_path is not None:
                if self.rank == 0:
                    self.logger.info(f"Loading Discriminator Private from {ckpt_path}...")
                self.load_model(discriminator_private, ckpt_path, tag="discriminator_private")

            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling discriminator private...")
                discriminator_private = torch.compile(
                    discriminator_private, mode=self.configs.train.compile.mode
                )
                if self.rank == 0:
                    self.logger.info("Compiling Done")

            self.discriminator_private = discriminator_private


        #load unet wrapper
        if self.configs.get("unet_wrapper", None) is not None:
            params = OmegaConf.to_container(
                self.configs.unet_wrapper.params, resolve=True
            )
            params["unet"] = self.model
            unet_wrapper = util_common.get_obj_from_str(
                self.configs.unet_wrapper.target
            )(**params)
            unet_wrapper = unet_wrapper.cuda()
            # if self.configs.unet_wrapper.params.get("shared_proj", None) is not None:
            #     ckpt_path = self.configs.unet_wrapper.params.shared_proj.get("ckpt_path", None)
            #     if ckpt_path is not None:
            #         if self.rank == 0:
            #             self.logger.info(f"Loading Unet Wrapper shared projector from {ckpt_path}...")
            #         self.load_model(unet_wrapper.shared_proj_16, ckpt_path, tag="unet_shared_proj")
            if self.configs.unet_wrapper.params.get("private_proj", None) is not None:
                ckpt_path = self.configs.unet_wrapper.params.private_proj.get("ckpt_path", None)
                if ckpt_path is not None:
                    if self.rank == 0:
                        self.logger.info(f"Loading Unet Wrapper private projector from {ckpt_path}...")
                    self.load_model(unet_wrapper.private_proj, ckpt_path, tag="unet_private_proj")
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder wrapper...")
                autoencoder_wrapper = torch.compile(
                    autoencoder_wrapper, mode=self.configs.train.compile.mode
                )
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.model = unet_wrapper

        # freeze the diffusion model
        self.freeze_model(self.model)
        # 解冻projector
        # for p in self.model.shared_proj_16.parameters():
        #     p.requires_grad = True
        for p in self.model.private_proj.parameters():
            p.requires_grad = True
        # unfreeze unet module
        train_keywords = [
            # # "middle_block"
            # "input_blocks[0]",
            # "input_blocks[1]",
            # "input_blocks[2]",

        ]
        if train_keywords:
            for n, p in self.model.unet.named_parameters():
                if any(k in n for k in train_keywords):
                    p.requires_grad = True
        # unfreeze the adapter modules
        # num_params = 0
        # for name, module in self.model.named_modules():
        #     if 'adapter' in name:
        #         for param in module.parameters():
        #             param.requires_grad = True
        #             num_params += param.numel()
        # if self.rank == 0:
        #     self.logger.info(f'Tuning adapter modules: {num_params/10**6:.2f}M...')

        # trainable parameters
        print("Unet trainable parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print("Diffusion model trainable parameters:", sum(p.numel() for p in self.model.unet.parameters() if p.requires_grad) if hasattr(self.model, "unet") and self.model.unet is not None else 0)
        print("AutoEncoder Wrapper trainable parameters:", sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad))
        print("Autoencoder base ae trainable parameters:", sum(p.numel() for p in self.autoencoder.base_ae.parameters() if p.requires_grad) if hasattr(self.autoencoder, "base_ae") and self.autoencoder.base_ae is not None else 0)
        # print("shared encoder trainable parameters:", sum(p.numel() for p in self.autoencoder.shared_encoder.parameters() if p.requires_grad) if hasattr(self.autoencoder, "shared_encoder") and self.autoencoder.shared_encoder is not None else 0)
        print("private encoder trainable parameters:", sum(p.numel() for p in self.autoencoder.private_encoder.parameters() if p.requires_grad) if hasattr(self.autoencoder, "private_encoder") and self.autoencoder.private_encoder is not None else 0)
        # print("Discriminator Shared trainable parameters:", sum(p.numel() for p in self.discriminator_shared.parameters() if p.requires_grad) if hasattr(self, "discriminator_shared") and self.discriminator_shared is not None else 0)
        print("Discriminator Private trainable parameters:", sum(p.numel() for p in self.discriminator_private.parameters() if p.requires_grad) if hasattr(self, "discriminator_private") and self.discriminator_private is not None else 0)

        print(type(configs_to_dict))
        print(type(self.configs))
        #init wandb
        wandb.init(
                    project=getattr(self.configs, "wandb_project", "difadapter"),
                    name=getattr(self.configs, "wandb_name", None),
                    config=OmegaConf.to_container(self.configs, resolve=True),
                    dir="/share/huayunpeng-local/wandb",
                )

    def training_step(self, data):
        self.model.train()
        if hasattr(self, "autoencoder") and self.autoencoder is not None:
            self.autoencoder.train()
        # if hasattr(self, "discriminator_shared") and self.discriminator_shared is not None:
        #     self.discriminator_shared.train()
        if hasattr(self, "discriminator_private") and getattr(self, "discriminator_private") is not None:
            self.discriminator_private.train()

        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

       # 清梯度：G/D 都清
        self.optimizer.zero_grad(set_to_none=True)
        # if getattr(self, "optimizer_D", None) is not None:
        #     self.optimizer_D.zero_grad(set_to_none=True)
        if getattr(self, "optimizer_Dp", None) is not None:
            self.optimizer_Dp.zero_grad(set_to_none=True)

        # lambda_adv = float(getattr(self.configs.train.loss_coef, "lambda_adv", 0.0))
        # lambda_align = float(getattr(self.configs.train.loss_coef, "lamdba_align", 0.0))
        lambda_private = float(getattr(self.configs.train.loss_coef, "lambda_private", 0.0))


        # shared_branch = bool(getattr(self.configs.train, "shared_branch", True))
        private_branch = bool(getattr(self.configs.train, "private_branch", True))

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {k: v[jj:jj+micro_batchsize] for k, v in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)

            # timestep
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=micro_data['gt'].device,
                )
            #noise shape (latent space)
            latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            noise_chn = self.configs.autoencoder.params.embed_dim
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (latent_resolution, ) * 2,
                    device=micro_data['gt'].device,
                )
            
            _, p_ir = self.autoencoder.encode(micro_data["lq"], return_features=True)
            # model kwargs
            model_kwargs = {}
            # if shared_branch:
            #     model_kwargs['shared_feat'] = s_ir
            if private_branch:
                model_kwargs['private_feat'] = p_ir
            if self.configs.model.params.cond_lq:
                model_kwargs['lq'] = micro_data['lq']
            else:
                model_kwargs = None
            
            # 1) diffusion 主损失（只用红外）
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],          # IR gt
                micro_data['lq'],          # IR lq
                tt,
                first_stage_model=self.autoencoder,  # 你的 wrapper
                model_kwargs=model_kwargs,
                noise=noise,
            )

            # ---------- diffusion loss（只 IR） ----------
            # 保持父类 DDP no_sync 的写法
            if (not last_batch) and (self.num_gpus > 1):
                with self.model.no_sync():
                    loss_dict, z_t, z0_pred = compute_losses()
            else:
                loss_dict, z_t, z0_pred = compute_losses()

            # 这里你按你工程返回键选择：有的返回 loss/mse
            if loss_dict['mse'].dim() > 0:
                loss_dict['mse'] = loss_dict['mse'].mean()
            diff_loss = loss_dict['mse']
            lambda_diff = float(getattr(self.configs.train.loss_coef, "lambda_diff", 1.0))
            diff_loss = lambda_diff * diff_loss

            # ---------- 计算 features（IR/VIS） ----------
            """
            损失函数：
            mse: diffusion噪声损失
            loss_D: share分支的判别器损失
            loss_adv: 对抗损失，用于特征不可分训练shared encoder
            loss_Dp: private分支的交叉熵损失
            loss_private: private encoder的损失
            """
            # loss_D = None
            # loss_adv = None
            loss_Dp = None
            loss_private = None

            if 'vis_lq' in micro_data:
                _, p_vis = self.autoencoder.encode(micro_data["vis_lq"], return_features=True)

                # print("shared_ir shape:", s_ir.shape)
                # print("private_ir shape:", p_ir.shape)

                lab_ir = torch.zeros(p_ir.size(0), dtype=torch.long, device=p_ir.device)  # 0=IR
                lab_vis = torch.ones (p_vis.size(0), dtype=torch.long, device=p_vis.device)  # 1=VIS

                # ---- shared 分支：训练 D + 对抗 encoder ----
                # if shared_branch and (self.discriminator_shared is not None):
                #      # (1) 训练 D：detach feature
                #     for p in self.discriminator_shared.parameters():
                #         p.requires_grad_(True)
                #     logits_ir_D = self.discriminator_shared(s_ir.detach())
                #     logits_vis_D = self.discriminator_shared(s_vis.detach())
                #     loss_D = F.cross_entropy(logits_ir_D, lab_ir) + F.cross_entropy(logits_vis_D, lab_vis)
                
                #     # (2) 训练 encoder（shared）骗 D：冻结 D 参数，不 detach feature
                #     if lambda_adv != 0.0:
                #         for p in self.discriminator_shared.parameters():
                #             p.requires_grad_(False)

                #         logits_ir_E = self.discriminator_shared(s_ir)
                #         logits_vis_E = self.discriminator_shared(s_vis)
                #         # 取负号：最大化分类损失 -> 域不可分
                #         # loss_adv = -(F.cross_entropy(logits_ir_E, lab_ir) + F.cross_entropy(logits_vis_E, lab_vis))
                #         loss_adv = kl_to_uniform(logits_ir_E) + kl_to_uniform(logits_vis_E)

                #         #l2 loss
                #         # s_ir_vec  = l2norm(pool_feat(s_ir.float()))
                #         # s_vis_vec = l2norm(pool_feat(s_vis.float()))
                #         # loss_align = (s_ir_vec - s_vis_vec).pow(2).sum(dim=1).mean()

                #         # 建议使用 L1，对异常值更鲁棒，更有利于保留边缘
                #         loss_align = F.l1_loss(s_ir, s_vis)

                #         loss_adv = lambda_adv * loss_adv + lambda_align * loss_align
                        
                #         for p in self.discriminator_shared.parameters():
                #             p.requires_grad_(True)
                    
                # ---- private 分支（可选）----
                if private_branch and getattr(self, "discriminator_private", None) is not None and p_ir is not None and p_vis is not None:
                    for p in self.discriminator_private.parameters():
                        p.requires_grad_(True)

                    logits_p_ir = self.discriminator_private(p_ir.detach())
                    logits_p_vis = self.discriminator_private(p_vis.detach())
                    loss_Dp = F.cross_entropy(logits_p_ir, lab_ir) + F.cross_entropy(logits_p_vis, lab_vis)

                    if lambda_private != 0.0:
                        # private 想“更可分”：正向 CE
                        for p in self.discriminator_private.parameters():
                            p.requires_grad_(False)

                        logits_p_ir_e = self.discriminator_private(p_ir)
                        logits_p_vis_e = self.discriminator_private(p_vis)
                        loss_private = F.cross_entropy(logits_p_ir_e, lab_ir) + F.cross_entropy(logits_p_vis_e, lab_vis)

                        for p in self.discriminator_private.parameters():
                            p.requires_grad_(True)

            # ---------- backward：先 D，再 G ----------
            # D backward（只让 D 收到梯度）
            # if loss_D is not None:
            #     self.backward_step_D(loss_D, num_grad_accumulate)
            if loss_Dp is not None:
                self.backward_step_Dp(loss_Dp, num_grad_accumulate)

            # G backward：diff + adv + (可选 private 可分 loss_private)
            # adv_term = loss_adv if (loss_adv is not None and lambda_adv != 0.0) else None
            extra_private = lambda_private * loss_private if (loss_private is not None) else None

            # 把 adv_term 和 extra_private 合并成一个 extra（保持 backward_step_G 简洁）
            extra = None
            # if adv_term is not None and extra_private is not None:
            #     extra = adv_term + extra_private
            # elif adv_term is not None:
            #     extra = adv_term
            if extra_private is not None:
                extra = extra_private

            # for p in self.discriminator_shared.parameters():
            #     p.requires_grad_(False)
            for p in self.discriminator_private.parameters():
                p.requires_grad_(False)
            self.backward_step_G(diff_loss, extra, num_grad_accumulate)

            # logging（你原来的 log_step_train 依赖 z_t/z0_pred，这里没拿；你如果要保留就继续用 dif_loss_wrapper 返回那套）
            if last_batch:
                # 你可以把 loss_dict 加上 adv/D 再 log
                # if loss_adv is not None: loss_dict["adv_loss"] = loss_adv.detach()
                # if loss_D is not None:   loss_dict["loss_D"] = loss_D.detach()
                if loss_Dp is not None:  loss_dict["loss_Dp"] = loss_Dp.detach()
                # self.log_step_train(...) 需要你按原函数签名补 z_t/z0_pred（如果必须的话，你就还是用你父类 backward_step 那种 compute_losses() 返回 z_t,z0_pred 的版本）
                # 这里先不强行调用，避免你接口对不上
                # self.log_step_train(loss_dict, tt, micro_data, z_t, z0_pred.detach())
                if self.current_iters%20 == 0: 
                    wandb.log({
                        k: v.item() if torch.is_tensor(v) else v
                        for k, v in loss_dict.items()
                    }, step=self.current_iters)
        
        # ---------- step：G/D 都 step ----------
        if self.configs.train.use_amp:
            # G
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
            # D
            # if self.optimizer_D is not None and loss_D is not None:
            #     self.amp_scaler_D.step(self.optimizer_D)
            #     self.amp_scaler_D.update()
            if getattr(self, "optimizer_Dp", None) is not None and loss_Dp is not None:
                self.amp_scaler_Dp.step(self.optimizer_Dp)
                self.amp_scaler_Dp.update()
        else:
            self.optimizer.step()
            # if self.optimizer_D is not None and loss_D is not None:
            #     self.optimizer_D.step()
            if getattr(self, "optimizer_Dp", None) is not None and loss_Dp is not None:
                self.optimizer_Dp.step()

         # 清梯度
        self.optimizer.zero_grad(set_to_none=True)
        if self.optimizer_D is not None:
            self.optimizer_D.zero_grad(set_to_none=True)
        if getattr(self, "optimizer_Dp", None) is not None:
            self.optimizer_Dp.zero_grad(set_to_none=True)

        # scheduler（如果你每 step 调一次）
        if getattr(self, "lr_scheduler", None) is not None:
            self.lr_scheduler.step()
        # if getattr(self, "lr_scheduler_D", None) is not None:
        #     self.lr_scheduler_D.step()
        if getattr(self, "lr_scheduler_Dp", None) is not None:
            self.lr_scheduler_Dp.step()

        # EMA
        # if hasattr(self.configs.train, 'ema_rate'):
        #     self.update_ema_model()
                
    def setup_optimizaton(self):
        # ========= 1) 选择 G 的参数（diffusion + autoencoder parts） =========
        params_G = []

        train_diffusion = bool(getattr(self.configs.train, "train_diffusion", False))
        if train_diffusion:
            params_G += [p for p in self.model.parameters() if p.requires_grad]

        if hasattr(self, "autoencoder") and self.autoencoder is not None:
            params_G += [p for p in self.autoencoder.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            params_G,
            lr=self.configs.train.lr,
            weight_decay=self.configs.train.weight_decay,
        )

        # ========= 2) 判别器 optimizer_D =========
        if hasattr(self, "discriminator_shared") and self.discriminator_shared is not None:
            self.optimizer_D = torch.optim.AdamW(
                self.discriminator_shared.parameters(),
                lr=getattr(self.configs.train, "lr_D", self.configs.train.lr),
                weight_decay=getattr(self.configs.train, "weight_decay_D", self.configs.train.weight_decay),
            )
        else:
            self.optimizer_D = None

        # private 判别器（如果你有）
        if hasattr(self, "discriminator_private") and getattr(self, "discriminator_private") is not None:
            self.optimizer_Dp = torch.optim.AdamW(
                self.discriminator_private.parameters(),
                lr=getattr(self.configs.train, "lr_Dp", getattr(self.configs.train, "lr_D", self.configs.train.lr)),
                weight_decay=getattr(self.configs.train, "weight_decay_Dp", getattr(self.configs.train, "weight_decay_D", self.configs.train.weight_decay)),
            )
        else:
            self.optimizer_Dp = None

        # ========= 3) AMP =========
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None
        self.amp_scaler_D = amp.GradScaler() if (self.configs.train.use_amp and self.optimizer_D is not None) else None
        self.amp_scaler_Dp = amp.GradScaler() if (self.configs.train.use_amp and self.optimizer_Dp is not None) else None

        # ========= 4) Scheduler =========
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                eta_min=self.configs.train.lr_min,
            )
            self.lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer_D,
                T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                eta_min=getattr(self.configs.train, "lr_min_D", self.configs.train.lr_min),
            ) if self.optimizer_D is not None else None

            self.lr_scheduler_Dp = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer_Dp,
                T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                eta_min=getattr(self.configs.train, "lr_min_Dp", getattr(self.configs.train, "lr_min_D", self.configs.train.lr_min)),
            ) if self.optimizer_Dp is not None else None
        else:
            self.lr_scheduler = None
            self.lr_scheduler_D = None
            self.lr_scheduler_Dp = None

    def backward_step_G(self, diff_loss, adv_loss, num_grad_accumulate):
        """
        只负责对 G 做 backward。
        diff_loss/adv_loss 都应是标量 tensor（未 mean 则这里会 mean）
        """
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext

        with context():
            # 保证是标量
            loss = diff_loss
            if loss.dim() > 0:
                loss = loss.mean()

            if adv_loss is not None:
                loss = loss + adv_loss

            loss = loss / num_grad_accumulate

        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return loss.detach()

    # def backward_step_D(self, loss_D, num_grad_accumulate):
    #     if loss_D is None or self.optimizer_D is None:
    #         return None
    #     loss = loss_D
    #     if loss.dim() > 0:
    #         loss = loss.mean()
    #     loss = loss / num_grad_accumulate

    #     if self.amp_scaler_D is None:
    #         loss.backward()
    #     else:
    #         self.amp_scaler_D.scale(loss).backward()
    #     return loss.detach()

    def backward_step_Dp(self, loss_Dp, num_grad_accumulate):
        if loss_Dp is None or self.optimizer_Dp is None:
            return None
        loss = loss_Dp
        if loss.dim() > 0:
            loss = loss.mean()
        loss = loss / num_grad_accumulate

        if self.amp_scaler_Dp is None:
            loss.backward()
        else:
            self.amp_scaler_Dp.scale(loss).backward()
        return loss.detach()  

    def validation(self, phase='val'):
        pass

    def save_ckpt(self):
        if self.rank == 0:
            # shared_proj_ckpt = {
            #         'iters_start': self.current_iters,
            #         # 'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
            #         # 'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
            #         'state_dict': self.model.shared_proj_16.state_dict(),  #unet
            #         }
            # torch.save(shared_proj_ckpt, self.ckpt_dir/f"shared_proj_{self.current_iters}.pth")
            private_proj_ckpt = {
                    'iters_start': self.current_iters,
                    # 'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                    # 'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                    'state_dict': self.model.private_proj.state_dict(),  #unet
                    }
            torch.save(private_proj_ckpt, self.ckpt_dir/f"private_proj_{self.current_iters}.pth")
            # 保存encoder
            if getattr(self, "autoencoder", None) is not None:
                # if getattr(self.autoencoder, "shared_encoder", None) is not None:
                #     shared_encoder_ckpt = {
                #         'iters_start': self.current_iters,
                #         'state_dict': self.autoencoder.shared_encoder.state_dict(),
                #     }
                #     torch.save(shared_encoder_ckpt, self.ckpt_dir/f"shared_encoder_{self.current_iters}.pth")
                if getattr(self.autoencoder, "private_encoder", None) is not None:
                    private_encoder_ckpt = {
                        'iters_start': self.current_iters,
                        'state_dict': self.autoencoder.private_encoder.state_dict(),
                    }
                    torch.save(private_encoder_ckpt, self.ckpt_dir/f"private_encoder_{self.current_iters}.pth")
                if getattr(self.autoencoder, "decoder_adapter", None) is not None:
                    decoder_adapter_ckpt = {
                        'iters_start': self.current_iters,
                        'state_dict': self.autoencoder.decoder_adapter.state_dict(),
                    }
                    torch.save(decoder_adapter_ckpt, self.ckpt_dir/f"decoder_adapter_{self.current_iters}.pth")
            # 保存判别器
            # if getattr(self, "discriminator_shared", None) is not None:
            #     discriminator_shared_ckpt = {
            #         'iters_start': self.current_iters,
            #         'state_dict': self.discriminator_shared.state_dict(),
            #     }
            #     torch.save(discriminator_shared_ckpt, self.ckpt_dir/f"discriminator_shared_{self.current_iters}.pth")
            if getattr(self, "discriminator_private", None) is not None:
                discriminator_private_ckpt = {
                    'iters_start': self.current_iters,
                    'state_dict': self.discriminator_private.state_dict(),
                }
                torch.save(discriminator_private_ckpt, self.ckpt_dir/f"discriminator_private_{self.current_iters}.pth")
            
            # unet_ckpt = {
            #     'iters_start': self.current_iters,
            #     'state_dict': self.model.unet.state_dict(),
            # }
            # torch.save(unet_ckpt, self.ckpt_dir/f"unet_{self.current_iters}.pth")

            if self.amp_scaler is not None:
                amp_scaler_ckpt = {
                    'iters_start': self.current_iters,
                    'state_dict': self.amp_scaler.state_dict(),
                }
                torch.save(amp_scaler_ckpt, self.ckpt_dir/f"amp_scaler_{self.current_iters}.pth")

            # if hasattr(self, 'ema_rate'):
            #     ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
            #     torch.save(self.ema_state, ema_ckpt_path)

def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def kl_to_uniform(logits):
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    return (p * (log_p + math.log(logits.size(1)))).sum(dim=1).mean()

def pool_feat(f):
    # f: [B,C,H,W] -> [B,C]
    return f.mean(dim=(2,3))
def l2norm(x, eps=1e-6):
    return x / (x.norm(dim=1, keepdim=True) + eps)
def info_nce_loss(z_ir, z_vis, temperature=0.1):
    """
    z_ir: [B, D]  (already projected)
    z_vis: [B, D]
    """
    z_ir = F.normalize(z_ir, dim=-1)
    z_vis = F.normalize(z_vis, dim=-1)

    logits = (z_ir @ z_vis.t()) / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2v = F.cross_entropy(logits, labels)
    loss_v2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2v + loss_v2i)

if __name__ == '__main__':
    from utils import util_image
    from  einops import rearrange
    im1 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00012685_crop000.png',
                            chn = 'rgb', dtype='float32')
    im2 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00014886_crop000.png',
                            chn = 'rgb', dtype='float32')
    im = rearrange(np.stack((im1, im2), 3), 'h w c b -> b c h w')
    im_grid = im.copy()
    for alpha in [0.8, 0.4, 0.1, 0]:
        im_new = im * alpha + np.random.randn(*im.shape) * (1 - alpha)
        im_grid = np.concatenate((im_new, im_grid), 1)

    im_grid = np.clip(im_grid, 0.0, 1.0)
    im_grid = rearrange(im_grid, 'b (k c) h w -> (b k) c h w', k=5)
    xx = vutils.make_grid(torch.from_numpy(im_grid), nrow=5, normalize=True, scale_each=True).numpy()
    util_image.imshow(np.concatenate((im1, im2), 0))
    util_image.imshow(xx.transpose((1,2,0)))

