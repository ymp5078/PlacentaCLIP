from config.base_config import Config
from datasets.model_transforms import init_transform_dict, NU_transform_dict, must_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from datasets.msvd_dataset import MSVDDataset
from datasets.lsmdc_dataset import LSMDCDataset
from datasets.nu_dataset import NUPreTrainDataset, NUFinetuneDataset, NUFinetuneSplitsDataset, NUTextDataset, NUPreTrainDatasetForVis
from torch.utils.data import DataLoader

# dev
from datasets.nu_dataset_dev import NUFinetuneSplitsDataset as NUFinetuneRobustSplitsDataset
from datasets.nu_dataset_dev import NUPreTrainDataset as NUPreTrainRobustDataset
from datasets.nu_dataset_dev import NUIpadDataset
from datasets.must_dataset import MustDataset
from datasets.robustness_transforms import get_transform, get_train_transforms

class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = NU_transform_dict(config.input_res)
        
        seg_trans = img_transforms['seg_trans']

        if config.dataset_name == "NU_Pretrain":
            img_trans = img_transforms['train_img_trans']
            img_seg_trans = img_transforms['train_img_seg_trans']
            # TODO: merge NUPreTrainRobustDataset and NUPreTrainDataset
            if config.additional_data:
                dataset = NUPreTrainRobustDataset(config, split_type=split_type,transforms = img_seg_trans, img_transforms=img_trans,seg_transforms=seg_trans, text_sample_method=config.text_sample_method, additional_data=config.additional_data)
                print(len(dataset))
            else:
                dataset = NUPreTrainDataset(config, split_type=split_type,transforms = img_seg_trans, img_transforms=img_trans,seg_transforms=seg_trans, text_sample_method=config.text_sample_method)
            return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)

        elif config.dataset_name == "NU_Finetune":
            img_trans = img_transforms['val_img_trans']
            if split_type == 'train':
                dataset = NUFinetuneDataset(config,split_type=split_type, img_transforms=img_trans,seg_transforms=seg_trans)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
            else:
                dataset = NUFinetuneDataset(config,split_type=split_type, img_transforms=img_trans,seg_transforms=seg_trans)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)

        else:
            raise NotImplementedError
    
    @staticmethod
    def get_robustness_data_loader(config: Config, split_type='train', severity=1):
        img_transforms = NU_transform_dict(config.input_res)
        train_img_trans = get_train_transforms(severity=severity,normalize=False,to_tensor=False)
        seg_trans = img_transforms['seg_trans']

        if config.dataset_name == "NU_Pretrain":
            img_trans = train_img_trans
            img_seg_trans = img_transforms['train_img_seg_trans']
            dataset = NUPreTrainRobustDataset(config, split_type=split_type,
                                              transforms = img_seg_trans, 
                                              img_transforms=img_trans,
                                              seg_transforms=seg_trans, 
                                              text_sample_method=config.text_sample_method,
                                              wb_aug_prob=0.2 if severity > 0 else 0)
            return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)

        elif config.dataset_name == "NU_Finetune":
            img_trans = img_transforms['val_img_trans']
            if split_type == 'train':
                dataset = NUFinetuneDataset(config,split_type=split_type, img_transforms=img_trans,seg_transforms=seg_trans)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
            else:
                dataset = NUFinetuneDataset(config,split_type=split_type, img_transforms=img_trans,seg_transforms=seg_trans)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)

        else:
            raise NotImplementedError
    
    @staticmethod
    def get_split_data_loader(config: Config, topic, random_seed = 100,  split_type='train', use_text_feats=False):
        img_transforms = NU_transform_dict(config.input_res)
        img_seg_trans = img_transforms['train_img_seg_trans']
        train_img_trans = img_transforms['train_img_trans']
        img_trans = img_transforms['val_img_trans']
        seg_trans = img_transforms['seg_trans']
        val_img_trans_ipad = img_transforms['val_img_trans_ipad']

        if config.dataset_name == "NU_Finetune":
            if 'train' in split_type:
                if use_text_feats and topic!='sepsis':
                    dataset = NUTextDataset(config,topic, random_seed, split_type=split_type,img_transforms=img_trans,seg_transforms=seg_trans)
                else:
                    dataset = NUFinetuneSplitsDataset(config,topic, random_seed, split_type=split_type, transforms=None, img_transforms=img_trans,seg_transforms=seg_trans)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
            elif 'test' in split_type:
                dataset = NUFinetuneSplitsDataset(config,topic, random_seed, split_type=split_type, img_transforms=img_trans,seg_transforms=seg_trans)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)
            elif 'ipad' in split_type:
                dataset = NUIpadDataset(config,topic, random_seed, split_type=split_type, img_transforms=val_img_trans_ipad,seg_transforms=seg_trans)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)

        else:
            raise NotImplementedError
        
    @staticmethod
    def get_vis_data_loader(config: Config, topic, random_seed = 100,  split_type='train', use_text_feats=False):
        img_transforms = NU_transform_dict(config.input_res)
        img_seg_trans = img_transforms['train_img_seg_trans']
        train_img_trans = img_transforms['train_img_trans']
        img_trans = img_transforms['val_img_trans']
        seg_trans = img_transforms['seg_trans']
        val_img_trans_ipad = img_transforms['val_img_trans_ipad']
        dataset = NUPreTrainDatasetForVis(config,topic=topic, split_type=split_type, img_transforms=img_trans,seg_transforms=seg_trans)
        return DataLoader(dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=config.num_workers)
    
    @staticmethod
    def get_robust_split_data_loader(config: Config, topic, random_seed = 100,  split_type='train', use_text_feats=False, transform_type='none', severity=1):
        img_transforms = NU_transform_dict(config.input_res)
        base_trans = get_transform(transform_type='none',severity=1)
        if transform_type ==' none' or transform_type == 'wb':
            img_trans = get_transform(transform_type='none',severity=1)
        else:
            img_trans = get_transform(transform_type=transform_type,severity=severity)
        seg_trans = img_transforms['seg_trans']

        if config.dataset_name == "NU_Finetune":
            if 'train' in split_type:
                # if use_text_feats and topic!='sepsis':
                #     dataset = NUTextDataset(config,topic, random_seed, split_type=split_type, img_transforms=img_trans,seg_transforms=seg_trans)
                # else:
                dataset = NUFinetuneRobustSplitsDataset(config,topic, random_seed, split_type=split_type, img_transforms=base_trans,seg_transforms=seg_trans)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
            else:
                dataset = NUFinetuneRobustSplitsDataset(config,topic, random_seed, split_type=split_type, img_transforms=img_trans,seg_transforms=seg_trans, wb_aug_prob = 1.0 if transform_type == 'wb' else 0.0, wb_setting=severity-1 if transform_type=='wb' else None )
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)

        else:
            raise NotImplementedError
        

    @staticmethod
    def get_must_data_loader(config: Config, topic):
        img_transforms = must_transform_dict(config.input_res)
        img_trans = img_transforms['val_img_trans']
        seg_trans = img_transforms['seg_trans']

        dataset = MustDataset(config,topic, img_transforms=img_trans,seg_transforms=seg_trans)
        return DataLoader(dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=config.num_workers)

    
    
