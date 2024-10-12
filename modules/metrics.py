import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats
import math


def log_cosh(feat1,feat2,weight=None,scale=100):
    """exp(-logcosh(x))"""
    size1 = feat1.size(0)
    size2 = feat2.size(0)
    x = torch.unsqueeze(feat1, dim=1) - torch.permute(torch.reshape(torch.tile(feat2, dims=(size1,)),(size2,size1,-1)),dims=(1,0,2))
    if weight is not None:
        x = x * weight
    # np.mean(x + np.log(np.exp(-2. * x) + 1.) - np.log(2.), axis=-1)
    logits_ab = -torch.mean(x + F.softplus(-2. * x) - math.log(2.0),-1) * scale
    logits_ab_exp = torch.exp(logits_ab)

    # logits_ab = -torch.mean(x + F.softplus(-2. * x) - math.log(2.0),-1)/scale

    # correct self using the mean similarity to prevent large neg sample reduction
    # correction = torch.zeros_like(logits_ab)
    # correction.fill_diagonal_(1.) # ignore self
    # logits_ab = logits_ab * (1-correction) + correction * logits_ab.mean(dim=-1, keepdim=True)

    mask = torch.ones_like(logits_ab)
    mask.fill_diagonal_(0.) # ignore self
    # logits_ab = logits_ab * scale
    # print(logits_ab)
    return logits_ab_exp * mask , logits_ab_exp#logits_ab + 1

def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type, global_embeds = None, additional_feats = None, filter_threshold = 0.9, scaling = False, noisy_filter = False, epoch=11, sim_func = None):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    # print(text_embeds.max(),text_embeds.min())
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)
    additional_sims = None
    additional_sims_1 = None
    global_sims = None
    if global_embeds is not None:
        global_embeds = global_embeds / global_embeds.norm(dim=-1, keepdim=True)
        global_sims = torch.mm(text_embeds, global_embeds.t())
    
    if filter_threshold <= 1: #and epoch > 300: #TODO: change to a variable
        # check if texts are similar

        if additional_feats is not None:
            if sim_func is None:
                additional_feats = additional_feats / additional_feats.norm(dim=-1, keepdim=True)
        else:
            additional_feats = text_embeds

        if sim_func is None:
            additional_sims = torch.mm(additional_feats, additional_feats.t()).detach()
            if global_embeds is not None:
                additional_sims_1 = torch.mm(global_embeds, additional_feats.t())
        else:
            random_drop=False
            if random_drop:
                keep_mask = torch.ones(size=(1,additional_feats.size(1)),dtype=additional_feats.dtype,device=additional_feats.device)
                keep_mask = F.dropout(keep_mask,p=0.2)
                additional_feats = additional_feats * keep_mask
            additional_sims, additional_sims_1 = sim_func(additional_feats, additional_feats)#.detach()
            # additional_sims = epoch * additional_sims 
            # print(additional_sims)
        
        if noisy_filter:
            additional_sims = additional_sims + torch.normal(mean=0.0,std=0.1,size=additional_sims.size(),device=additional_sims.device,dtype=additional_sims.dtype)
        if sim_func is None:
            # mark the similar text
            # sims_mask = (additional_sims >= filter_threshold) | (additional_sims < 0.4)
            # sims_mask = additional_sims < 0.4
            sims_mask = additional_sims >= filter_threshold

            # do not mark self
            sims_mask.fill_diagonal_(fill_value=False)
            filter_values = torch.zeros_like(additional_sims)

            # dropout-like scaling base on the number of removed values
            scale_values = (sims_mask.size(1)-sims_mask.sum(1,keepdim=True))/sims_mask.size(1)
            scale_values = scale_values
            

            
            # print(sims_mask.sum(),additional_sims.min())
            # create num_texts x num_vids of value 0 or inf
            filter_values = filter_values.masked_fill(sims_mask, 1e4)
        else:
            # use the sim value to reweight
            filter_values = additional_sims
            # print(filter_values)
            # filter_values = filter_values / filter_threshold
            # filter_values = (1 - filter_values) / filter_threshold
            scale_values = 1.
        # print(filter_values[0])

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        # remove false negatives
        if additional_sims is not None: 
            if scaling:
                sims = sims * scale_values
            sims = sims - filter_values
            # sims = sims * filter_values
            
    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        
        text_embeds = text_embeds.unsqueeze(1)
        
        # num_texts x 1 x embed_dim
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)
        # remove false negatives
        if additional_sims is not None:
            if scaling:
                sims = sims * scale_values
                global_sims = global_sims * scale_values
            # sims = sims - filter_values
            # global_sims = global_sims - filter_values
    
    return sims, global_sims, additional_sims_1


def sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type):
    """
    Computes the similarity matrix using pooled video frames using all texts per video

    Output
        sims: num_vids x max_text_per_vid x num_vids
    """
    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x embed_dim

        sims = text_embeds_per_video_id @ vid_embeds_pooled_per_video_id.t()

    else:
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x num_vids x max_text_per_vid x embed_dim
        num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        # num_vids x max_text_per_vid x embed_dim x num_vids
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1,2,3,0)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.view(num_vids*max_text_per_vid, embed_dim, num_vids)
        # num_vids x max_text_per_vid x 1 x embed_dim
        text_embeds_per_video_id = text_embeds_per_video_id.unsqueeze(2)
        text_embeds_per_video_id = text_embeds_per_video_id.view(num_vids*max_text_per_vid, 1, embed_dim)

        sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)
        sims = sims.view(num_vids, max_text_per_vid, 1, num_vids).squeeze(2)
        
    return sims


def generate_embeds_per_video_id(text_embeds, vid_embeds_pooled, all_vid_ids, pooling_type):
    # Construct dictionary of text embeds per unique video id
    text_embeds_per_video_id = {}

    for idx, v_id in enumerate(all_vid_ids):
        if v_id in text_embeds_per_video_id:
            text_embeds_per_video_id[v_id].append(text_embeds[idx])
        else:
            text_embeds_per_video_id[v_id] = [text_embeds[idx]]

    for v_id in text_embeds_per_video_id:
        text_embeds_per_video_id[v_id] = torch.stack(text_embeds_per_video_id[v_id])

    # num_vids x max_text_per_vid x embed_dim
    text_embeds_per_video_id = pad_and_stack_dict_to_tensor(text_embeds_per_video_id,
        text_embeds_per_video_id.keys(), text_embeds.shape[-1])

    if pooling_type == 'avg':
        # num_vids x embed_dim
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        # Construct dictionary of video embeds for each text per video_id
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            # num_vids x max_text_per_vid x embed_dim
            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                    vid_embeds_pooled_per_video_id[i].keys(), vid_embeds_pooled.shape[-1])

        # num_vids x num_vids x max_text_per_vid x embed_dim
        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id


def t2v_metrics(sims):
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sims = sims.permute(1,0,2)
    
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))
    
    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.numpy())


def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims!=sims] = float('-inf')
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy() # diagonal

    return compute_metrics(ranks)


def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])
    
    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), 
                                                        float("-inf"), device = input[k].device)]) for k in input}
    
    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input


if __name__ == '__main__':
    value = torch.rand(8,128).cuda() * 100
    value2 = torch.zeros_like(value)
    print(log_cosh(value,value))