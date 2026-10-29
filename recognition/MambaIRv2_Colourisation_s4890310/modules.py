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

class LabColouriser(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

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

    def forward(self, L):
        x = self.conv_in(L)

        residual = self.base.conv_after_body(self.base.forward_features(x)) + x
        return self.conv_out(residual)