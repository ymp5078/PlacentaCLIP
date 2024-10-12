from __future__ import annotations
import logging
from contextlib import suppress
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import abc
from typing import Callable, Optional
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, RidgeClassifierCV
from sklearn.metrics import top_k_accuracy_score, average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

# from ..models.tokenizer import tokenize
# from .utils import unwrap_model


# TODO: support distributed evaluation
class EvaluatorBase(abc.ABC):
    @abc.abstractmethod
    def eval(self) -> dict:
        raise NotImplementedError


# class ZeroShotClassifier(EvaluatorBase):
#     def __init__(
#         self,
#         model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel,
#         dataloader: torch.utils.data.DataLoader,
#         class_names: list[str],
#         templates: Iterable[Callable[[str], str]] | Callable[[str], str] = lambda x: x,
#         input_type: str = 'video',
#         precision: str = 'amp',
#         device: str | torch.device = 'cuda',
#         debug_mode: bool = False,
#         args: Optional[argparse.Namespace] = None
#     ):
#         assert precision in ['amp', 'fp32'], f'precision must be one of [amp, fp32], got {precision=}'
#         assert input_type in ['video', 'image'], f'`input_type` must be one of ["video", "image"] but got {input_type=}'

#         self.model = unwrap_model(model)
#         self.dataloader = dataloader
#         self.class_names = class_names
#         self.templates = templates if isinstance(templates, Iterable) else [templates]
#         self.input_type = input_type
#         # override by args if provided
#         if args is not None:
#             self.precision = args.precision
#             self.device = args.device
#             self.debug_mode = args.debug_mode
#         else:
#             self.precision = precision
#             self.device = device
#             self.debug_mode = debug_mode

#         with torch.no_grad():
#             zeroshot_weights = []
#             for class_name in class_names:
#                 texts = [template(class_name) for template in templates]  # format with class
#                 texts = tokenize(texts).to(self.device, non_blocking=True)  # tokenize
#                 class_embeddings = self.model.encode_text(texts)
#                 class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
#                 class_embedding /= class_embedding.norm()
#                 zeroshot_weights.append(class_embedding)
#             self.classifier = torch.stack(zeroshot_weights, dim=1).to(self.device, non_blocking=True)


#     def get_logits(self, dataloader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
#         all_logits = []
#         all_targets = []
#         autocast = torch.cuda.amp.autocast if self.precision == 'amp' else suppress
#         tdataloader = tqdm(dataloader, desc=f'zero-shot', unit_scale=dataloader.batch_size, leave=False)
#         feature_extractor = self.model.encode_image if self.input_type == 'image' else self.model.encode_video
#         with torch.no_grad():
#             for i, (visual_inputs, targets) in enumerate(tdataloader):
#                 visual_inputs = visual_inputs.to(self.device, non_blocking=True)
#                 targets = targets.to(self.device, non_blocking=True)

#                 with autocast():
#                     features = feature_extractor(visual_inputs)
#                     features = F.normalize(features, dim=-1)
#                     logits = 100. * features @ self.classifier

#                 all_logits.append(logits)
#                 all_targets.append(targets)

#                 # debug mode
#                 if self.debug_mode and i == 1:
#                     break

#             all_logits = torch.cat(all_logits).cpu().numpy()
#             all_targets = torch.cat(all_targets).cpu().numpy()
#         return all_logits, all_targets

#     def eval(self) -> dict[str, float]:
#         self.model.eval()
#         all_logits, all_targets = self.get_logits(self.dataloader)
#         if self.debug_mode:
#             return {'top1': -1, 'top5': -1}
#         top1 = top_k_accuracy_score(all_targets, all_logits, k=1) * 100.
#         top5 = top_k_accuracy_score(all_targets, all_logits, k=5) * 100.
#         return {'top1': top1, 'top5': top5}



class LinearProbClassifier(EvaluatorBase):
    def __init__(
        self,
        model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel,
        dataloaders: list[torch.utils.data.DataLoader],
        input_type: str = 'video',
        precision: str = 'amp',
        device: str | torch.device = 'cuda',
        debug_mode: bool = False,
        args: Optional[argparse.Namespace] = None
    ):
        assert len(dataloaders) == 2, f'2 dataloaders are required for training and validation but got {len(dataloaders)=}'
        assert precision in ['amp', 'fp32'], f'precision must be one of [amp, fp32], got {precision=}'
        assert input_type in ['video', 'image'], f'`input_type` must be one of ["video", "image"] but got {input_type=}'

        self.model = model
        self.input_type = input_type
        # override by args if provided
        if args is not None:
            self.precision = args.precision
            self.device = args.device
            self.debug_mode = args.debug_mode
        else:
            self.precision = precision
            self.device = device
            self.debug_mode = debug_mode

        self.train_dataloader, self.val_dataloader = dataloaders
        self.clf = LogisticRegression(
            random_state=1,
            max_iter=1000,
            C=3.16,
            solver='sag'
        )
        self.args = args


    def get_features(self, dataloader: torch.utils.data.DataLoader, split: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        tdataloader = tqdm(dataloader, desc=f'linear-eval ({split})', unit_scale=dataloader.batch_size, leave=False)
        autocast = torch.cuda.amp.autocast if self.precision == 'amp' else suppress
        feature_extractor = self.model.clip.encode_image 
        all_features = []
        all_targets = []
        with torch.no_grad():
            for i, (visual_inputs, targets) in enumerate(tdataloader):
                visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast():
                    features = feature_extractor(visual_inputs)
                    features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_targets.append(targets)

                # debug mode
                if self.debug_mode and i == 1:
                    break

            all_features = torch.cat(all_features).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()
        return all_features, all_targets


    def eval(self) -> dict[str, float]:
        self.model.eval()
        train_features, train_targets = self.get_features(self.train_dataloader, split='train')
        val_features, val_targets = self.get_features(self.val_dataloader, split='val')
        if self.debug_mode:
            return {'top1': -1, 'top5': -1}
        self.clf.fit(train_features, train_targets)
        probs = self.clf.predict_proba(val_features)
        top1 = top_k_accuracy_score(val_targets, probs, k=1) * 100.
        top5 = top_k_accuracy_score(val_targets, probs, k=5) * 100.
        return {'top1': top1, 'top5': top5}


class MulticlassLinearRegressor(EvaluatorBase):
    def __init__(
        self,
        model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel,
        dataloaders: list[torch.utils.data.DataLoader],
        input_type: str = 'video',
        precision: str = 'amp',
        device: str | torch.device = 'cuda',
        debug_mode: bool = False,
        args: Optional[argparse.Namespace] = None
    ):
        assert len(dataloaders) == 2, f'2 dataloaders are required for training and validation but got {len(dataloaders)=}'
        assert precision in ['amp', 'fp32'], f'precision must be one of [amp, fp32], got {precision=}'
        assert input_type in ['video', 'image'], f'`input_type` must be one of ["video", "image"] but got {input_type=}'

        self.model = model
        self.train_dataloader, self.val_dataloader = dataloaders
        self.input_type = input_type
        # override by args if provided
        if args is not None:
            self.precision = args.precision
            self.device = args.device
            self.debug_mode = args.debug_mode
        else:
            self.precision = precision
            self.device = device
            self.debug_mode = debug_mode

        self.base_clf = LogisticRegression(
            random_state=1,
            max_iter=2000,
            C=3.16,
            solver='sag',
            class_weight=None
        )
        self.clf = OneVsRestClassifier(self.base_clf, n_jobs=8)


    def get_features(self, dataloader: torch.utils.data.DataLoader, split: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        tdataloader = tqdm(dataloader, desc=f'linear-eval ({split})', unit_scale=dataloader.batch_size, leave=False)
        autocast = torch.cuda.amp.autocast if self.precision == 'amp' else suppress
        feature_extractor = self.model.clip.encode_image
        all_features = []
        all_targets = []
        with torch.no_grad():
            for i, batch in enumerate(tdataloader):
                # if len(batch) == 2:
                visual_inputs, targets = batch['image'],batch['label']
                visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast():
                    features = feature_extractor(visual_inputs)
                    features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_targets.append(targets)
                    
                # elif len(batch) == 3:
                #     visual_inputs, visual_masks, targets = batch
                #     visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                #     targets = targets.to(self.device, non_blocking=True)
                #     visual_masks = visual_masks.to(self.device, non_blocking=True)
                #     with autocast():
                #         features = feature_extractor(visual_inputs, visual_masks)
                #         if isinstance(features,tuple):
                #             features = features[0] 
                #         features = F.normalize(features, dim=-1)
                #     all_features.append(features)
                #     all_targets.append(targets)
                # elif len(batch) == 4:
                #     visual_inputs, visual_masks, targets, target_idx = batch
                #     visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                #     targets = targets.to(self.device, non_blocking=True)
                #     visual_masks = visual_masks.to(self.device, non_blocking=True)
                #     with autocast():
                #         features = feature_extractor(visual_inputs, visual_masks, mean_pool=False)
                #         if isinstance(features,tuple):
                #             features = features[0] 
                #         print(features.shape)
                #         features = F.normalize(features, dim=-1)
                #     all_features.append(features)
                #     all_targets.append(targets)
                    

                # debug mode
                if self.debug_mode and i == 1:
                    break

            all_features = torch.cat(all_features).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()
        return all_features, all_targets


    def eval(self) -> dict[str, float]:
        self.model.eval()
        train_features, train_targets = self.get_features(self.train_dataloader, split='train')
        val_features, val_targets = self.get_features(self.val_dataloader, split='val')
        if self.debug_mode:
            return {'mAP': -1, 'auc': -1}
        self.clf.fit(train_features, train_targets)
        probs = self.clf.predict_proba(val_features)
        mAP = average_precision_score(val_targets, probs) * 100.
        auc = roc_auc_score(val_targets, probs) * 100.
        mAP, auc = np.round(mAP, 2), np.round(auc, 2)
        return {'mAP': mAP, 'auc': auc}

class MulticlassLinearClassifier(EvaluatorBase):
    def __init__(
        self,
        model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel,
        dataloaders: list[torch.utils.data.DataLoader],
        input_type: str = 'video',
        precision: str = 'amp',
        device: str | torch.device = 'cuda',
        debug_mode: bool = False,
        tokenizer = None,
        args: Optional[argparse.Namespace] = None
    ):
        assert len(dataloaders) == 2, f'2 dataloaders are required for training and validation but got {len(dataloaders)=}'
        assert precision in ['amp', 'fp32'], f'precision must be one of [amp, fp32], got {precision=}'
        assert input_type in ['video', 'image'], f'`input_type` must be one of ["video", "image"] but got {input_type=}'

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader, self.val_dataloader = dataloaders
        self.input_type = input_type
        # override by args if provided
        if args is not None:
            self.precision = args.precision
            self.device = args.device
            self.debug_mode = args.debug_mode
        else:
            self.precision = precision
            self.device = device
            self.debug_mode = debug_mode

        self.base_clf = LogisticRegression(
            random_state=1,
            max_iter=2000,
            C=10.0,
            solver='sag',
            class_weight=None
        )
        self.clf = OneVsRestClassifier(self.base_clf, n_jobs=8)


    def get_features(self, dataloader: torch.utils.data.DataLoader, split: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        tdataloader = tqdm(dataloader, desc=f'linear-eval ({split})', unit_scale=dataloader.batch_size, leave=False)
        autocast = torch.cuda.amp.autocast if self.precision == 'amp' else suppress
        feature_extractor = self.model.get_feature
        all_features = []
        all_targets = []
        with torch.no_grad():
            for i, batch in enumerate(tdataloader):
                # if len(batch) == 2:
                visual_inputs, targets = batch['image'],batch['label']
                visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                text = self.tokenizer(batch['text'], return_tensors='pt', padding=True,
                                              truncation=True).to(self.device, non_blocking=True)
                batch['image'],batch['label'], batch['text'] = visual_inputs, targets, text

                with autocast():
                    features = feature_extractor(batch)
                    features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_targets.append(targets)
                    
                # elif len(batch) == 3:
                #     visual_inputs, visual_masks, targets = batch
                #     visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                #     targets = targets.to(self.device, non_blocking=True)
                #     visual_masks = visual_masks.to(self.device, non_blocking=True)
                #     with autocast():
                #         features = feature_extractor(visual_inputs, visual_masks)
                #         if isinstance(features,tuple):
                #             features = features[0] 
                #         features = F.normalize(features, dim=-1)
                #     all_features.append(features)
                #     all_targets.append(targets)
                    

                # debug mode
                if self.debug_mode and i == 1:
                    break

            all_features = torch.cat(all_features).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()
        return all_features, all_targets


    def eval(self) -> dict[str, float]:
        self.model.eval()
        train_features, train_targets = self.get_features(self.train_dataloader, split='train')
        val_features, val_targets = self.get_features(self.val_dataloader, split='val')
        if self.debug_mode:
            return {'mAP': -1, 'auc': -1}
        self.clf.fit(train_features, train_targets)
        probs = self.clf.predict_proba(val_features)
        mAP = average_precision_score(val_targets, probs) * 100.
        auc = roc_auc_score(val_targets, probs) * 100.
        mAP, auc = np.round(mAP, 2), np.round(auc, 2)
        return {'mAP': mAP, 'auc': auc}

class LinearClassifier(EvaluatorBase):
    def __init__(
        self,
        model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel,
        dataloaders: list[torch.utils.data.DataLoader],
        input_type: str = 'video',
        precision: str = 'amp',
        device: str | torch.device = 'cuda',
        debug_mode: bool = False,
        tokenizer = None,
        text_train = False,
        args: Optional[argparse.Namespace] = None
    ):
        assert len(dataloaders) in [2,3], f'2 or 3 dataloaders are required for training and validation but got {len(dataloaders)=}'
        assert precision in ['amp', 'fp32'], f'precision must be one of [amp, fp32], got {precision=}'
        assert input_type in ['video', 'image'], f'`input_type` must be one of ["video", "image"] but got {input_type=}'

        self.model = model
        self.tokenizer = tokenizer
        if len(dataloaders) == 2:
            self.train_dataloader, self.val_dataloader = dataloaders
        else:
            self.train_dataloader, self.val_dataloader, self.pretrain_dataloader = dataloaders

        self.input_type = input_type
        self.text_train = text_train
        # override by args if provided
        if args is not None:
            self.precision = args.precision
            self.device = args.device
            self.debug_mode = args.debug_mode
        else:
            self.precision = precision
            self.device = device
            self.debug_mode = debug_mode

        # self.clf = LogisticRegressionCV(
        #     cv=5,
        #     random_state=1,
        #     max_iter=2000,
        #     Cs=np.linspace(0.0001,100.0,30),
        #     solver='sag',
        #     class_weight=None
        # )
        self.clf = LogisticRegression(
            random_state=1,
            max_iter=1000,
            C=3.16,
            solver='sag'
        )

    def get_pretrain_features(self, dataloader: torch.utils.data.DataLoader, split: str = 'train', iteration: int = 1, return_attn=False) -> tuple[np.ndarray, np.ndarray]:
        tdataloader = tqdm(dataloader, desc=f'linear-eval ({split})', unit_scale=dataloader.batch_size, leave=False)
        autocast = torch.cuda.amp.autocast if self.precision == 'amp' else suppress
        feature_extractor = self.model.get_text_features if self.text_train else self.model.get_feature
        all_features = []
        all_targets = []
        all_files = []
        all_attn_weights = []
        all_images = []
        with torch.no_grad():
            for niter in range(iteration):
                for i, batch in enumerate(tdataloader):
                    # if len(batch) == 2:
                    if self.text_train:
                        visual_inputs, targets = None,batch['label']
                    else:
                        visual_inputs = batch['image']
                        visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                    text = self.tokenizer(batch['text'], return_tensors='pt', padding=True,
                                                truncation=True).to(self.device, non_blocking=True)
                    
                    batch['image'], batch['text'] = visual_inputs, text
                    if batch.get('text_feat',None) is not None:
                        batch['text_feat'] = batch['text_feat'].to(self.device, non_blocking=True)
                    if return_attn:
                        batch['text'] = self.tokenizer(batch['topic_text'], return_tensors='pt', padding=True,
                                                truncation=True).to(self.device, non_blocking=True)
                        with autocast():
                            features, attn_weights = feature_extractor(batch,condition=True,return_attn=True)
                            # print(attn_weights.shape)
                            all_attn_weights.append(attn_weights[:,:,:,0])
                            all_images.append(batch['image'])
                            features = F.normalize(features, dim=-1)
                    else:
                        with autocast():
                            features = feature_extractor(batch)
                            features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_files+=batch['studyid']
                    # elif len(batch) == 3:
                    #     visual_inputs, visual_masks, targets = batch
                    #     visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                    #     targets = targets.to(self.device, non_blocking=True)
                    #     visual_masks = visual_masks.to(self.device, non_blocking=True)
                    #     with autocast():
                    #         features = feature_extractor(visual_inputs, visual_masks)
                    #         if isinstance(features,tuple):
                    #             features = features[0] 
                    #         features = F.normalize(features, dim=-1)
                    #     all_features.append(features)
                    #     all_targets.append(targets)
                        

                    # debug mode
                    if self.debug_mode and i == 1:
                        break

            all_features = torch.cat(all_features).cpu().numpy()
            if len(all_attn_weights)>0:
                all_attn_weights = torch.cat(all_attn_weights).cpu().numpy()
                all_images = torch.cat(all_images).cpu().numpy()
            
            # print(all_files)
            # all_files = sum(all_files)
        if return_attn:
            return all_features, all_targets, all_files, all_attn_weights, all_images
        print(len(all_features))
        return all_features, all_targets, all_files
    

    def get_features(self, dataloader: torch.utils.data.DataLoader, split: str = 'train', iteration: int = 1, return_attn=False) -> tuple[np.ndarray, np.ndarray]:
        tdataloader = tqdm(dataloader, desc=f'linear-eval ({split})', unit_scale=dataloader.batch_size, leave=False)
        autocast = torch.cuda.amp.autocast if self.precision == 'amp' else suppress
        feature_extractor = self.model.get_text_features if self.text_train else self.model.get_feature
        all_features = []
        all_targets = []
        all_files = []
        all_attn_weights = []
        all_images = []
        with torch.no_grad():
            for niter in range(iteration):
                for i, batch in enumerate(tdataloader):
                    # if len(batch) == 2:
                    if self.text_train:
                        visual_inputs, targets = None,batch['label']
                    else:
                        visual_inputs, targets = batch['image'],batch['label']
                        visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    text = self.tokenizer(batch['text'], return_tensors='pt', padding=True,
                                                truncation=True).to(self.device, non_blocking=True)
                    
                    batch['image'],batch['label'], batch['text'] = visual_inputs, targets, text
                    if batch.get('text_feat',None) is not None:
                        batch['text_feat'] = batch['text_feat'].to(self.device, non_blocking=True)
                    if return_attn:
                        batch['text'] = self.tokenizer(batch['topic_text'], return_tensors='pt', padding=True,
                                                truncation=True).to(self.device, non_blocking=True)
                        with autocast():
                            features, attn_weights = feature_extractor(batch,condition=True,return_attn=True)
                            # print(attn_weights.shape)
                            all_attn_weights.append(attn_weights[:,:,:,0])
                            all_images.append(batch['image'])
                            features = F.normalize(features, dim=-1)
                    else:
                        with autocast():
                            features = feature_extractor(batch)
                            features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_targets.append(targets)
                    all_files+=batch['studyid']
                    # elif len(batch) == 3:
                    #     visual_inputs, visual_masks, targets = batch
                    #     visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                    #     targets = targets.to(self.device, non_blocking=True)
                    #     visual_masks = visual_masks.to(self.device, non_blocking=True)
                    #     with autocast():
                    #         features = feature_extractor(visual_inputs, visual_masks)
                    #         if isinstance(features,tuple):
                    #             features = features[0] 
                    #         features = F.normalize(features, dim=-1)
                    #     all_features.append(features)
                    #     all_targets.append(targets)
                        

                    # debug mode
                    if self.debug_mode and i == 1:
                        break

            all_features = torch.cat(all_features).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()
            if len(all_attn_weights)>0:
                all_attn_weights = torch.cat(all_attn_weights).cpu().numpy()
                all_images = torch.cat(all_images).cpu().numpy()
            
            # print(all_files)
            # all_files = sum(all_files)
        if return_attn:
            return all_features, all_targets, all_files, all_attn_weights, all_images
        print(len(all_features))
        return all_features, all_targets, all_files


    def eval(self) -> dict[str, float]:
        self.model.eval()
        train_features, train_targets, _ = self.get_features(self.train_dataloader, split='train')
        val_features, val_targets, val_files = self.get_features(self.val_dataloader, split='val')
        # print(val_files)
        if self.debug_mode:
            return {'mAP': -1, 'auc': -1}
        self.clf.fit(train_features, train_targets)
        # print('best C:',self.clf.C_)
        probs = self.clf.predict_proba(val_features)[:,1]
        # print(probs.shape)
        mAP = average_precision_score(val_targets, probs) * 100.
        auc = roc_auc_score(val_targets, probs) * 100.
        mAP, auc = np.round(mAP, 2), np.round(auc, 2)
        return {'mAP': mAP, 'auc': auc, 'targets': val_targets, 'probs': probs, 'studyid': val_files}
    
    def eval_with_attn_map(self):
        self.model.eval()
        train_features, train_targets, train_files, train_attn_weights, train_images = self.get_features(self.train_dataloader, split='train',return_attn=True)
        val_features, val_targets, val_files, val_attn_weights, val_images = self.get_features(self.val_dataloader, split='val',return_attn=True)
        attn_weights = np.concatenate([train_attn_weights,val_attn_weights])
        images = np.concatenate([train_images,val_images])
        print(train_attn_weights.shape,train_images.shape)
        # print(val_files)
        if self.debug_mode:
            return {'mAP': -1, 'auc': -1}
        self.clf.fit(train_features, train_targets)
        # print('best C:',self.clf.C_)
        probs = self.clf.predict_proba(val_features)[:,1]
        # print(probs.shape)
        mAP = average_precision_score(val_targets, probs) * 100.
        auc = roc_auc_score(val_targets, probs) * 100.
        mAP, auc = np.round(mAP, 2), np.round(auc, 2)

        
        val_files = [f"{val_file}_{probs[i]}_{val_targets[i]}" for (i,val_file) in enumerate(val_files)]
        train_files = [f"{train_file}_nan_{train_targets[i]}" for (i,train_file) in enumerate(train_files)]
        names = train_files+val_files

        return {'mAP': mAP, 'auc': auc, 'targets': val_targets, 'probs': probs, 'studyid': val_files,'attn_weights':attn_weights,"images":images,"names":names}
    
    def eval_pretrain_with_attn_map(self):
        self.model.eval()
        train_features, train_targets, names, attn_weights, images = self.get_pretrain_features(self.pretrain_dataloader, split='train',return_attn=True)
        # attn_weights = np.concatenate([train_attn_weights,val_attn_weights])
        # images = np.concatenate([train_images,val_images])
        # print(train_attn_weights.shape,train_images.shape)
        # names = train_files+val_files

        return {'mAP': 0, 'auc': 0, 'targets': 0, 'probs': 0, 'studyid': names,'attn_weights':attn_weights,"images":images,"names":names}



class MulticlassSlidingWindowLinearClassifier(EvaluatorBase):
    def __init__(
        self,
        model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel,
        dataloaders: list[torch.utils.data.DataLoader],
        input_type: str = 'video',
        precision: str = 'amp',
        device: str | torch.device = 'cuda',
        debug_mode: bool = False,
        args: Optional[argparse.Namespace] = None
    ):
        assert len(dataloaders) == 2, f'2 dataloaders are required for training and validation but got {len(dataloaders)=}'
        assert precision in ['amp', 'fp32'], f'precision must be one of [amp, fp32], got {precision=}'
        assert input_type in ['video', 'image'], f'`input_type` must be one of ["video", "image"] but got {input_type=}'

        self.model = unwrap_model(model)
        self.train_dataloader, self.val_dataloader = dataloaders
        self.input_type = input_type
        # override by args if provided
        if args is not None:
            self.precision = args.precision
            self.device = args.device
            self.debug_mode = args.debug_mode
        else:
            self.precision = precision
            self.device = device
            self.debug_mode = debug_mode

        self.base_clf = LogisticRegression(
            random_state=1,
            max_iter=2000,
            C=3.16,
            solver='sag',
            class_weight=None
        )
        self.clf = OneVsRestClassifier(self.base_clf, n_jobs=8)


    def get_features(self, dataloader: torch.utils.data.DataLoader, split: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        tdataloader = tqdm(dataloader, desc=f'linear-eval ({split})', unit_scale=dataloader.batch_size, leave=False)
        autocast = torch.cuda.amp.autocast if self.precision == 'amp' else suppress
        feature_extractor = self.model.encode_image if self.input_type == 'image' else self.model.encode_video
        all_features = []
        all_targets = []
        with torch.no_grad():
            for i, batch in enumerate(tdataloader):
                if len(batch) == 2:
                    visual_inputs, targets = batch
                    visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    with autocast():
                        features = feature_extractor(visual_inputs)
                        features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_targets.append(targets)
                    
                elif len(batch) == 3:
                    visual_inputs, visual_masks, targets = batch
                    visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    visual_masks = visual_masks.to(self.device, non_blocking=True)
                    with autocast():
                        features = feature_extractor(visual_inputs, visual_masks)
                        features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_targets.append(targets)
                    

                # debug mode
                if self.debug_mode and i == 1:
                    break
            annotations = dataloader.dataset.annotations
            unique_video_id = (annotations['video_id']+annotations['person_id'].astype(str)).values

            # sorted the idx to make grouping O(n)
            sorted_idx = np.argsort(unique_video_id)
            unique_video_id = unique_video_id[sorted_idx]

            all_features = torch.cat(all_features).cpu().numpy()[sorted_idx]
            all_targets = torch.cat(all_targets).cpu().numpy()[sorted_idx]

            # group by unique_video_id
            all_features = np.split(all_features , np.unique(unique_video_id, return_index=True)[1][1:])
            all_targets = np.split(all_targets , np.unique(unique_video_id, return_index=True)[1][1:])

            # get mean for each feature and any value for each target since they are all the same
            all_features = np.stack([np.mean(feat,axis=0) for feat in all_features],axis=0)
            all_targets = np.stack([targ[0] for targ in all_targets],axis=0)
            print(all_targets.shape,all_features.shape)

        return all_features, all_targets


    def eval(self) -> dict[str, float]:
        self.model.eval()
        train_features, train_targets = self.get_features(self.train_dataloader, split='train')
        val_features, val_targets = self.get_features(self.val_dataloader, split='val')
        if self.debug_mode:
            return {'mAP': -1, 'auc': -1}
        self.clf.fit(train_features, train_targets)
        probs = self.clf.predict_proba(val_features)
        mAP = average_precision_score(val_targets, probs) * 100.
        auc = roc_auc_score(val_targets, probs) * 100.
        mAP, auc = np.round(mAP, 2), np.round(auc, 2)
        return {'mAP': mAP, 'auc': auc}


