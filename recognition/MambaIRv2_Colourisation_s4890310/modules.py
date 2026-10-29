import torch
import torch.nn as nn

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