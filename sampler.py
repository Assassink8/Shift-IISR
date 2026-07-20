#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27
# Copyright (c) 2022 S-Lab
# Modified by Yunpeng Hua for Shift-IISR in 2026.

import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from contextlib import nullcontext

from utils import util_net
from utils import util_image
from utils import util_common
from utils.shift_iisr_checkpoint import load_shift_iisr_checkpoint

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from utils.util_image import ImageSpliterTh

class BaseSampler:
    def __init__(
            self,
            configs,
            sf=4,
            use_amp=True,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            padding_offset=16,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.sf = sf
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = padding_offset

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        assert num_gpus== 1, 'Please assign only one available GPU using CUDA_VISIBLE_DEVICES during sampling!'
        self.rank = 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            util_net.reload_model(model, ckpt['state_dict'])
        else:
            util_net.reload_model(model, ckpt)
        self.freeze_model(model)
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder.params.get("lora_tune_decoder", False):
            lora_vae_state = ckpt['lora_vae']
        elif self.configs.autoencoder.get("tune_decoder", False):
            vae_state = ckpt['vae']
        if self.configs.autoencoder is not None:
            # params = self.configs.autoencoder.get('params', dict)
            params = self.configs.autoencoder.get('params', {})
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder = autoencoder.cuda()
            if self.configs.autoencoder.params.get("lora_tune_decoder", False):
                ckpt_path = self.configs.autoencoder.ckpt_path
                self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
                self.load_model_lora(autoencoder, ckpt_path, tag='autoencoder')
                autoencoder.load_state_dict(lora_vae_state, strict=False)
            elif self.configs.autoencoder.get("tune_decoder", False):
                ckpt_path = self.configs.autoencoder.ckpt_path
                self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
                self.load_model(autoencoder, ckpt_path)
                ckpt_path =self.configs.model.ckpt_path
                self.write_log(f'Loading Finetuned decoder from {ckpt_path}...')
                autoencoder.load_state_dict(vae_state, strict=False)
            else:
                ckpt_path = self.configs.autoencoder.ckpt_path
                self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
                self.load_model(autoencoder, ckpt_path)
            self.freeze_model(autoencoder)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        # load autoencoder wrapper
        if self.configs.get("autoencoderwrapper", None) is not None:
            # params = self.configs.autoencoderwrapper.get('params', dict)
            params = OmegaConf.to_container(
                self.configs.autoencoderwrapper.params, resolve=True
            )
            params['base_ae'] = self.autoencoder
            autoencoder_wrapper = util_common.get_obj_from_str(self.configs.autoencoderwrapper.target)(**params)
            autoencoder_wrapper = autoencoder_wrapper.cuda()
            if self.configs.autoencoderwrapper.params.get("decoder_adapter", None) is not None and self.configs.autoencoderwrapper.params.decoder_adapter.get("ckpt_path", None) is not None:
                ckpt_path = self.configs.autoencoderwrapper.params.decoder_adapter.ckpt_path
                self.write_log(f'Loading AutoEncoder Wrapper decoder adapter from {ckpt_path}...')
                self.load_model(autoencoder_wrapper.decoder_adapter, ckpt_path)
            self.freeze_model(autoencoder_wrapper)
            self.autoencoder = autoencoder_wrapper.eval()
        
        #load unet wrapper
        if self.configs.get("unet_wrapper", None) is not None:
            # params = self.configs.unetwrapper.get('params', dict)
            params = OmegaConf.to_container(
                self.configs.unet_wrapper.params, resolve=True
            )
            params['unet'] = self.model
            unet_wrapper = util_common.get_obj_from_str(self.configs.unet_wrapper.target)(**params)
            unet_wrapper = unet_wrapper.cuda()
            self.freeze_model(unet_wrapper)
            self.model = unet_wrapper.eval()

        shift_iisr_path = self.configs.shift_iisr.ckpt_path
        if shift_iisr_path is None:
            raise ValueError("configs.shift_iisr.ckpt_path must be provided for inference.")
        self.write_log(f"Loading Shift-IISR model from {shift_iisr_path}...")
        self.shift_iisr_checkpoint = load_shift_iisr_checkpoint(
            shift_iisr_path,
            self.autoencoder.grm_feature_extractor,
            self.model.grm_projector,
            map_location=f"cuda:{self.rank}",
            expected_lsr_strength=self.configs.diffusion.params.lsr_strength,
        )

    def load_model_lora(self, model, ckpt_path=None, tag='model'):
        if self.rank == 0:
            self.write_log(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        num_success = 0
        for key, value in model.named_parameters():
            if key in ckpt:
                value.data.copy_(ckpt[key])
                num_success += 1
            else:
                key_parts = key.split('.')
                if 'conv' in key_parts:
                    key_parts.remove('conv')
                new_key = '.'.join(key_parts)
                if new_key in ckpt:
                    value.data.copy_(ckpt[new_key])
                    num_success += 1
        assert num_success == len(ckpt)
        if self.rank == 0:
            self.write_log('Loaded Done')

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class ResShiftSampler(BaseSampler):
    @torch.inference_mode()
    def sample_func(self, y0, noise_repeat=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        offset = self.padding_offset
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % offset == 0 and ori_w % offset == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / offset)) * offset - ori_h
            pad_w = (math.ceil(ori_w / offset)) * offset - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        _, grm_feat = self.autoencoder.encode(y0, return_features=True)

         # model kwargs
        model_kwargs = {}
        if self.model.grm_projector is not None:
            model_kwargs['grm_feat'] = grm_feat
        if self.configs.model.params.cond_lq:
            model_kwargs['lq'] = y0
        else:
            model_kwargs = None

        results = self.base_diffusion.p_sample_loop(
                y=y0,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                )    # This has included the decoding for latent space

        if flag_pad:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]

        return results.clamp_(-1.0, 1.0)

    @torch.inference_mode()
    def inference(self, in_path, out_path, mask_back=True, bs=1, noise_repeat=False):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        def _process_per_image(im_lq_tensor):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [-1, 1], RGB
                mask: image mask for inpainting, [-1, 1], 1 for unknown area
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.sf,
                        extra_bs=self.chop_bs,
                        )
                for im_lq_pch, index_infos in im_spliter:
                    with context():
                        im_sr_pch = self.sample_func(
                                im_lq_pch,
                                noise_repeat=noise_repeat,
                                )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                # print(im_lq_tensor.shape)
                with context():
                    im_sr_tensor = self.sample_func(
                            im_lq_tensor,
                            noise_repeat=noise_repeat,
                            )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5

            return im_sr_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if self.rank == 0:
            assert in_path.exists()
            if not out_path.exists():
                out_path.mkdir(parents=True)

        # if self.num_gpus > 1:
        #     dist.barrier()

        if in_path.is_dir():
            data_config = {'type': 'base',
                            'params': {'dir_path': str(in_path),
                                        'transform_type': 'default',
                                        'transform_kwargs': {
                                            'mean': 0.5,
                                            'std': 0.5,
                                            },
                                        'need_path': True,
                                        'recursive': True,
                                        'length': None,
                                        }
                            }
            dataset = create_dataset(data_config)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    )
            for data in dataloader:
                batch_names = [Path(path).stem for path in data['path']]
                results = _process_per_image(data['lq'].cuda())    # b x h x w x c, [0, 1], RGB
                for jj in range(results.shape[0]):
                    im_sr = util_image.tensor2img(results[jj], rgb2bgr=True, min_max=(0.0, 1.0))
                    im_name = batch_names[jj]
                    im_path = out_path / f"{im_name}.png"
                    util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
        else:
            im_lq = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
            im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w
            im_sr_tensor = _process_per_image((im_lq_tensor - 0.5) / 0.5)

            im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))
            im_path = out_path / f"{in_path.stem}.png"
            util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

if __name__ == '__main__':
    pass
