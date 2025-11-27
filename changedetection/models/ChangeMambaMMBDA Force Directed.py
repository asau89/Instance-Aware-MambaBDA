import torch
import torch.nn as nn
import torch.nn.functional as F
from changedetection.models.Mamba_backbone import Backbone_VSSM
from classification.models.vmamba import LayerNorm2d
from changedetection.models.ChangeDecoder_BRIGHT import ChangeDecoder
from changedetection.models.SemanticDecoder import SemanticDecoder


class ChangeMambaMMBDA(nn.Module):
    def __init__(self, output_building, output_damage, pretrained, **kwargs):
        super(ChangeMambaMMBDA, self).__init__()

        # --- Encoders ---
        self.encoder_1 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        self.encoder_2 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        norm_layer = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}

        # --- Decoders ---
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

        # --- Classification heads (for BDA loss) ---
        self.main_clf = nn.Conv2d(128, output_damage, kernel_size=1)
        self.aux_clf = nn.Conv2d(128, output_building, kernel_size=1)

        # --- Projection layers for HR (1/4) feature maps ---
        in_channels_hr = self.encoder_1.dims[0]  # channel dim of 1/4 feature map
        self.proj_building_hr = nn.Conv2d(in_channels_hr, 128, kernel_size=1)
        self.proj_damage_hr = nn.Conv2d(in_channels_hr, 128, kernel_size=1)


    def forward(self, pre_data, post_data):
        # ----- encoders -----
        pre_feats = self.encoder_1(pre_data)  # list of [stage0, stage1, stage2, stage3]
        post_feats = self.encoder_2(post_data)

        # ----- decoders -----
        features_building = self.decoder_building(pre_feats)
        features_damage = self.decoder_damage(pre_feats, post_feats)

        # Use stage0 feature maps (1/4 resolution) for instance loss
        features_building_hr = self.proj_building_hr(pre_feats[0])
        features_damage_hr = self.proj_damage_hr(post_feats[0])

        # ----- final predictions (1/8 resolution outputs) -----
        pred_building = F.interpolate(self.aux_clf(features_building),
                                      size=pre_data.shape[-2:], mode='bilinear', align_corners=False)
        pred_damage = F.interpolate(self.main_clf(features_damage),
                                    size=post_data.shape[-2:], mode='bilinear', align_corners=False)

        if self.training:
            # Return both HR and low-res features for different losses
            return pred_building, pred_damage, features_building, features_damage, features_building_hr, features_damage_hr
        else:
            return pred_building, pred_damage
