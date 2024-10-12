import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch,config.input_res)

        # clip
        #  Current Best mAP 35.68, mAP 35.68                                                                                                                                                      
        #  Current Best AUC is 71.46, AUC 71.46
        # Transformer
        #  Current Best mAP 42.72, mAP 42.02
        #  Current Best AUC is 74.3, AUC 73.31
        # TODO: fixed this with sim_matrix_training
        # config.pooling_type = 'avg'
        self.pooling_type = config.pooling_type
        scale = self.clip.visual.embed_dim ** -0.5
        # print(self.clip.transformer_width, self.clip.visual.output_dim)
        self.patch_proj = nn.Parameter(scale * torch.randn(self.clip.visual.embed_dim, self.clip.visual.output_dim))
        self.pool_patches = Transformer(config)

        # custom
        if self.config.filter_threshold<=1.:
            self.addional_proj = nn.Parameter(scale * torch.randn(768, self.clip.visual.output_dim))
        


    def forward(self, data, return_all_patches=False):
        batch_size = data['image'].shape[0]
        text_data = data['text']
        image_data = data['image']
        addition_data = data['text_feat']
        # image_data = image_data.reshape(-1, 3, self.config.input_res[0], self.config.input_res[1])
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            image_features = self.clip.get_image_features(image_data)
        else:
            text_features = self.clip.encode_text(text_data)
            image_features, patch_features, tokens = self.clip.encode_image(image_data, return_tokens=True)
   
        # video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        
        # print(image_features[0].shape, text_features.shape, (image_features[1]@ self.patch_proj).shape, image_features[0].shape)#, (image_features[1]@ self.patch_proj).shape)
        if self.pooling_type == 'transformer':
            # print('use transformer')
            image_features_pooled = self.pool_patches(text_features , patch_features@ self.patch_proj)
            # image_features_pooled = self.pool_patches(text_features , tokens)
        else:
            image_features_pooled = image_features
        # image_features = image_features[0]
        # image_features_pooled = image_features
        addition_features = None
        if self.config.filter_threshold<=1.:
            addition_features = addition_data @ self.addional_proj
        # print(self.addional_proj)
        if self.pooling_type == 'transformer':
            return text_features, image_features, image_features_pooled, addition_features

        return text_features, image_features, image_features_pooled, addition_features
    
    def get_feature(self, data, condition=False, return_attn=False):

        batch_size = data['image'].shape[0]
        text_data = data['text']
        image_data = data['image']
        # image_data = image_data.reshape(-1, 3, self.config.input_res[0], self.config.input_res[1])
        
        if self.config.huggingface:
            if condition:
                text_features = self.clip.get_text_features(**text_data)
            image_features = self.clip.get_image_features(image_data)
        else:
            if condition:
                text_features = self.clip.encode_text(text_data)
            image_features, patch_features, tokens = self.clip.encode_image(image_data, return_tokens=True)
   
        # video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        
        # print(image_features[0].shape, text_features.shape, (image_features[1]@ self.patch_proj).shape, image_features[0].shape)#, (image_features[1]@ self.patch_proj).shape)

        # image_features = image_features[0]
        # image_features_pooled = image_features
        if condition:
            if return_attn:
                image_features_pooled, attn_weights = self.pool_patches(text_features , patch_features@ self.patch_proj,return_attn=return_attn)
                image_features_pooled = torch.diagonal(image_features_pooled,dim1=0, dim2=1).T
                # print(image_features_pooled.shape)
                return  image_features_pooled + image_features, attn_weights
            else:
                image_features_pooled =  self.pool_patches(text_features , patch_features@ self.patch_proj)
                image_features_pooled = torch.diagonal(image_features_pooled,dim1=0, dim2=1).T
                # image_features_pooled = self.pool_patches(text_features , tokens@ self.patch_proj)
                # print(image_features_pooled.shape)
                # image_features_pooled = torch.diagonal(image_features_pooled)
                # print(image_features_pooled.shape)
                # print(image_features_pooled.shape)
                return image_features_pooled + image_features

        return image_features

    def get_text_features(self, data, use_text_feats=False):
        batch_size = data['text'].shape[0]
        text_data = data['text']
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
        else:
            text_features = self.clip.encode_text(text_data)
        if use_text_feats:
            return text_feats, data['text_feat']
        return text_features
