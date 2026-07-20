# Copyright (c) 2022 S-Lab
# Modified by Yunpeng Hua for Shift-IISR in 2026.

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from contextlib import contextmanager
from typing import Optional, Tuple, Any
from omegaconf import DictConfig


import loralib as lora

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

class VQModelTorch(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 rank=8,    # rank for lora
                 lora_alpha=1.0,
                 lora_tune_decoder=False,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        if lora_tune_decoder:
            conv_layer = partial(lora.Conv2d, r=rank, lora_alpha=lora_alpha)
        else:
            conv_layer = nn.Conv2d

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(rank=rank, lora_alpha=lora_alpha, lora_tune=lora_tune_decoder, **ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = conv_layer(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, force_not_quantize=True)
        return dec

    def forward(self, input, force_not_quantize=False):
        h = self.encode(input)
        dec = self.decode(h, force_not_quantize)
        return dec

class VQModelTorchWrapper(nn.Module):
    """
    Wrap a VQModelTorch-like autoencoder and optionally insert:
      - latent_encoder: applied after base.encode(x)
      - latent_decoder: applied before base.decode(h)

    This wrapper keeps the original VQModelTorch interface:
      - encode(x) -> h
      - decode(h, force_not_quantize=False) -> x_rec
      - decode_code(code_b) -> x_rec
      - forward(x, force_not_quantize=False) -> x_rec
    """

    def __init__(
        self,
        base_ae: nn.Module,
        grm_feature_extractor: Optional[nn.Module] = None,
        decoder_adapter: Optional[nn.Module] = None,
        freeze_base: bool = True,
    ):
        super().__init__()
        if isinstance(base_ae, (dict, DictConfig)):
            base_ae = instantiate_from_config(base_ae)
        self.base_ae = base_ae
        if isinstance(grm_feature_extractor, (dict, DictConfig)):
            grm_feature_extractor = instantiate_from_config(grm_feature_extractor)
        self.grm_feature_extractor = grm_feature_extractor

        if isinstance(decoder_adapter, (dict, DictConfig)):
            decoder_adapter = instantiate_from_config(decoder_adapter)
        self.decoder_adapter = decoder_adapter
        
        if freeze_base:
            self.requires_grad_base_(False)
        
    # freezing helpers
    def requires_grad_base_(self, flag: bool):
        for p in self.base_ae.parameters():
            p.requires_grad = flag
        return self
    
    def requires_grad_grm_(self, flag: bool):
        if self.grm_feature_extractor is not None:
            for p in self.grm_feature_extractor.parameters():
                p.requires_grad  = flag
        return self
    
    def encode(self, x: torch.Tensor, return_features=False, *args, **kwargs) -> torch.Tensor:
        """
        Base: h = base_ae.encode(x)
        Then extract the GRM feature if a GRM feature extractor is provided.
        """
        h = self.base_ae.encode(x, *args, **kwargs)
        grm_feat = None
        if self.grm_feature_extractor is not None:
            grm_feat = self.grm_feature_extractor(h)
        if return_features:
            return h, grm_feat
        return h
    
    def decode(self, h: torch.Tensor, force_not_quantize: bool = False, *args, **kwargs) -> torch.Tensor:
        """
        Optionally map h via latent_decoder before base decode.
        IMPORTANT: do NOT change force_not_quantize semantics.
        """
        if self.decoder_adapter is not None:
            h = self.decoder_adapter(h)
        return self.base_ae.decode(h, force_not_quantize=force_not_quantize, *args, **kwargs)

    def decode_code(self, code_b: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Keep original behavior. base_ae.decode_code will call base_ae.decode(..., force_not_quantize=True)
        but that bypasses our latent_decoder. So we re-implement through base quantize embed_code + wrapper decode.
        """
        if not hasattr(self.base_ae, "quantize"):
            raise AttributeError("base_ae has no attribute 'quantize', cannot decode_code.")

        quant_b = self.base_ae.quantize.embed_code(code_b)
        # Go through wrapper decode so latent_decoder is applied if present.
        return self.decode(quant_b, force_not_quantize=True, *args, **kwargs)

    def forward(self, x: torch.Tensor, force_not_quantize: bool = False, *args, **kwargs) -> torch.Tensor:
        h = self.encode(x, *args, **kwargs)
        return self.decode(h, force_not_quantize=force_not_quantize, *args, **kwargs)

    @property
    def encoder(self):
        return getattr(self.base_ae, "encoder", None)

    @property
    def decoder(self):
        return getattr(self.base_ae, "decoder", None)

    @property
    def quantize(self):
        return getattr(self.base_ae, "quantize", None)
    

class AutoencoderKLTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input, sample_posterior, return_moments=False)
        dec = self.decode(z)
        return dec

class EncoderKLTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z
    def forward(self, x, sample_posterior=True, return_moments=False):
        return self.encode(x, sample_posterior, return_moments)

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
