import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint
from mambairv2_arch import MambaIRv2

MambaDenoiser_URL = 'https://huggingface.co/cguoh/MambaIR/resolve/main/MambaIRv2_ckpt/mambairv2_ColorDN_15.pth'

MambaDenoiser_dictionary = dict(upscale=1, in_chans=3, img_size=128, img_range=1.0, embed_dim=174,  d_state=16, depths=[6] * 6, num_heads=[6] * 6,
                  window_size=16, inner_rank=64, num_tokens=128, convffn_kernel_size=5, mlp_ratio=2.0) #copied from the pretrained model's test file

def build_pretrained_MambaDenoiser(checkpoint_path=None):
    model = MambaIRv2(**MambaDenoiser_dictionary)
    if checkpoint_path is None:
        checkpoint = torch.hub.load_state_dict_from_url(MambaDenoiser_URL, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['params'], strict=True)

    return model

class LabColouriser(nn.Module):
    def __init__(self, base, use_checkpoint=True):
        super().__init__()
        self.base = base
        self.use_checkpoint = use_checkpoint

        old_in = base.conv_first
        self.conv_in = nn.Conv2d(1, old_in.out_channels, kernel_size=old_in.kernel_size, stride=old_in.stride, padding=old_in.padding)

        with torch.no_grad():
            self.conv_in.weight.copy_(old_in.weight.sum(dim=1, keepdim=True))
            self.conv_in.bias.copy_(old_in.bias)

        old_out = base.conv_last
        self.conv_out = nn.Conv2d(old_out.in_channels, 2, kernel_size=old_out.kernel_size, stride=old_out.stride, padding=old_out.padding)

        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        for p in list(old_in.parameters()) + list(old_out.parameters()):
            p.requires_grad = False

    def forward_features(self, x, params):
        b = self.base
        x_size = (x.shape[2], x.shape[3])
        x = b.patch_embed(x)
        if b.ape:
            x = x + b.absolute_pos_embed

        for layer in b.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, x_size, params, use_reentrant=False)
            else:
                x = layer(x, x_size, params)

        x = b.norm(x)
        return b.patch_unembed(x, x_size)


    def forward(self, L):
        b = self.base

        h_old, w_old = L.shape[-2:]
        mod = b.window_size
        h = ((h_old + mod - 1) // mod) * mod #pad L channel to multiple of window size (16)
        w = ((w_old + mod - 1) // mod) * mod
        x = torch.cat([L, torch.flip(L, [2])], 2)[:, :, :h, :] #padding by flipping image upsidedown and concatenating it to itself
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w] #padding by flipping image left to right and concatenating it to itself

        x = (x - b.mean.type_as(x).mean()) * b.img_range #applies same scaling as pretrained model

        params = {'attn_mask': b.calculate_mask([h, w]).to(x.device), 'rpi_sa': b.relative_position_index_SA} #same as pretrained model

        x_first = self.conv_in(x) #makes L channel into 174 feature channels, same as pretrained model's first conv layer

        residual = b.conv_after_body(self.forward_features(x_first, params)) + x_first #skip connection, runs pretrained forward pass plus an extra conv layer
        # after the body to match the pretrained model's architecture

        ab = self.conv_out(residual)
        
        return ab[..., :h_old, :w_old] #output is cropped to match the original L channel's dimensions