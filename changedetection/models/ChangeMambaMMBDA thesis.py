import torch
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

import torch.nn as nn
import torch.nn.functional as F
from MambaCD.changedetection.models.ChangeDecoder_BRIGHT import ChangeDecoder
from MambaCD.changedetection.models.SemanticDecoder import SemanticDecoder


class ChangeMambaMMBDA(nn.Module):
    def __init__(self, output_building, output_damage, pretrained, **kwargs):
        super(ChangeMambaMMBDA, self).__init__()
        # Using two separate encoders for multimodal data (e.g., Optical + SAR)
        self.encoder_1 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        self.encoder_2 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
       
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder_1.channel_first

        print(f"ChangeMambaMMBDA initialized with channel_first={self.channel_first}")

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
       
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}

        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder_1.dims,
            channel_first=self.encoder_1.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder_2.dims,
            channel_first=self.encoder_2.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
      
        self.main_clf = nn.Conv2d(in_channels=128, out_channels=output_damage, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_building, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder_1(pre_data)
        post_features = self.encoder_2(post_data)

        # Decoder processing - these are the intermediate features we need
        features_building = self.decoder_building(pre_features)
        features_damage = self.decoder_damage(pre_features, post_features)
       
        # Final predictions
        pred_building = self.aux_clf(features_building)
        pred_building = F.interpolate(pred_building, size=pre_data.size()[-2:], mode='bilinear')

        pred_damage = self.main_clf(features_damage)
        pred_damage = F.interpolate(pred_damage, size=post_data.size()[-2:], mode='bilinear')
       
        # During training, return intermediate features for the consistency loss
        if self.training:
            return pred_building, pred_damage, features_building, features_damage
        else:
            # During evaluation/inference, return only the final predictions
            return pred_building, pred_damage