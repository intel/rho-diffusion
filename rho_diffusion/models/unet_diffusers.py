from __future__ import annotations
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DConditionModel, UNet2DModel
from rho_diffusion.registry import registry

@registry.register_model("UNet_Diffuser")
class UNet_nd(nn.Module):
    def __init__(
        self,
        data_shape: torch.Size | int | list,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list = [16, 8],
        dropout: float = 0,
        channel_mult: list | tuple = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: int | bool = None,
        cond_fn: nn.Module = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.data_shape = data_shape 
        self.cond_fn = cond_fn

        self.model = UNet2DModel(
            sample_size=data_shape,           # the target image resolution
            in_channels=in_channels, # Additional input channels for class cond.
            out_channels=1,           # the number of output channels
            layers_per_block=num_res_blocks,       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64), 
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
            ),
            # num_class_embeds=None
            class_embed_type='identity'
        )

    def forward(self, x, timesteps, y=None):

        
        
        if y is not None:
            # import pdb; pdb.set_trace()
            # cond = self.cond_fn(y).view(x.shape[0], y.shape[0], 1, 1).expand(x.shape[0], y.shape[0], self.data_shape[0], self.data_shape[1])
            # net_input = torch.cat((x, cond), 1) 
            cond = self.cond_fn(y)
            net_input = x 
        else:
            net_input = x 
        model_out = self.model(sample=net_input, timestep=timesteps, class_labels=cond).sample 
        # if y is not None:
        #     print(model_out.shape, self.cond_fn(y).shape)
        #     model_out += self.cond_fn(y)

        return model_out 
