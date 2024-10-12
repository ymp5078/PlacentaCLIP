import os
import cv2
import sys
import sys
sys.path.append("/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta_torch")
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from config.vlp_config import AllConfig
from datasets.video_capture import VideoCapture
from preprocess.WBAugmenter import WBEmulator as wbAug

TOPIC_TEXT = {
    'fir':'fetal inflammatory response',
    'mir':'maternal inflammatory response',
    'meconium':'meconium',
    'chorioamnionitis':'chorioamnionitis',
    'sepsis':'fetal inflammatory response, maternal inflammatory response, chorioamnionitis'
}

def load_mir_fir_file(filename):
    with open(filename,'r') as f:
        lines = [line.strip().split(',') for line in f.readlines()]
    # print(lines)
    mir_fir_dict = {line[1]: int(line[2]) for line in lines}
    return mir_fir_dict

def load_test_file(filename):
    test_df = pd.read_csv(filename)
    test_df.insertion = test_df.insertion.map(lambda x: ['normal','furcate','velamentous','circummarginate','marginal','circumvallate'].index(x))
    test_df = test_df.fillna(value=0)
    return dict(zip(test_df['studyid'].values,test_df[['meconium','fir','mir','chorioamnionitis']].values))


def load_clinical_info(filename):
    clinical_df = pd.read_csv(filename)

    # fillna with mean
    clinical_df = clinical_df.fillna(value=clinical_df[['Maternal age','Gestational age','delivery']].mean())
    return dict(zip(clinical_df['studyid'].values,clinical_df[['Maternal age','Gestational age','delivery']].values))


class NUPreTrainDataset(Dataset):
    """
        data_dir: directory where all data and annotations are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, 
                 split_type = 'train', 
                 transforms = None, 
                 img_transforms=None, 
                 seg_transforms=None, 
                 mask_img=True, 
                 text_sample_method='group', 
                 wb_aug_prob=0.0,
                 wb_setting = None,
                 additional_data = False,
                 use_2017_as_test=True):
        self.config = config
        self.data_dir = config.data_dir
        self.transforms = transforms
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
        self.split_type = split_type
        self.mask_img = mask_img
        self.wb_aug_prob = wb_aug_prob
        self.text_sample_method = text_sample_method
        self.additional_images_files = []
        if use_2017_as_test:
            test_set_dict = load_test_file(os.path.join(self.data_dir,'placenta_test_labels.csv'))
        else:
            test_set_dict = load_mir_fir_file(os.path.join(self.data_dir,'fir_labels.txt'))
        # clinical_dict = None
        # if use_clinical:
        clinical_dict = load_clinical_info(os.path.join(self.data_dir,'clinical_info.csv'))
        #print(np.unique(list(test_set_dict.values()),return_counts=True))
        test_studyids = list(test_set_dict.keys())
        annotation_file = os.path.join(self.data_dir,'annotation_diagnosis_new.json')
        annotation_feat_file = os.path.join(self.data_dir,'annotation_diagnosis_new.npy')
        image_dir = os.path.join(self.data_dir,'fetal_side')

        images_files = os.listdir(image_dir)
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.annotation_feats = np.load(annotation_feat_file, allow_pickle='TRUE').item()

        image_path_to_caption = defaultdict(list)
        image_label = defaultdict(int)

        clinical_data = defaultdict(np.ndarray)
        self.image_files = list()
        for val in images_files:
            studyid = val.split('.')[0].split('_')[0]
            if studyid not in self.annotations.keys():
                continue
            if split_type=='test':
                if studyid in test_studyids:
                    self.image_files.append(val)
            else:
                if studyid not in test_studyids:
                    self.image_files.append(val)
        
        if wb_aug_prob > 0:
            self.wb_color_aug = wbAug.WBEmulator()
            target_files = []
            for image_file in self.image_files:
                img_file = os.path.join(self.data_dir,'fetal_side',image_file)
                target_files.append(img_file)
            target_dir = os.path.join(self.data_dir,f'mfs_pretrain')
            if not os.path.exists(target_dir):
                self.target_dir = self.wb_color_aug.precompute_mfs(target_files,target_dir=target_dir)
            else:
                self.target_dir = target_dir
        else:
            self.target_dir = None

        # print(additional_data)
        if additional_data:
            additional_annotation_file = os.path.join(self.data_dir,'additional_annotation_diagnosis_new.json')
            additional_annotation_feat_file = os.path.join(self.data_dir,'additional_annotation_diagnosis_new.npy')
            additional_image_dir = os.path.join(self.data_dir,'additional_fetal_side')
            additional_images_files = os.listdir(additional_image_dir)
            with open(additional_annotation_file, 'r') as f:
                self.additional_annotations = json.load(f)
            self.additional_annotation_feats = np.load(additional_annotation_feat_file, allow_pickle='TRUE').item()

            clinical_data = defaultdict(np.ndarray)
            self.additional_images_files = list()
            for val in additional_images_files:
                studyid = val.split('.')[0].split('_')[0]
                if studyid not in self.additional_annotations.keys():
                    continue
                if split_type=='test':
                    if studyid in test_studyids:
                        self.additional_images_files.append(val)
                else:
                    if studyid not in test_studyids:
                        self.additional_images_files.append(val)
            
            if wb_aug_prob > 0:
                self.wb_color_aug = wbAug.WBEmulator()
                target_files = []
                for image_file in self.additional_images_files:
                    img_file = os.path.join(self.data_dir,'additional_fetal_side',image_file)
                    target_files.append(img_file)
                additional_target_dir = os.path.join(self.data_dir,f'additional_mfs_pretrain')
                if not os.path.exists(additional_target_dir):
                    self.additional_target_dir = self.wb_color_aug.precompute_mfs(target_files,target_dir=additional_target_dir)
                else:
                    self.additional_target_dir = additional_target_dir
            else:
                self.additional_target_dir = None
            # print(self.additional_images_files)

    def filter_topic(self,annotation,topics=['membranes','acute','funisitis','chorion','abnormal','inflam','meconium','focal','arteri']):
        new_annotation = []
        for anno in annotation:
            for topic in topics:
                if topic in anno:
                    new_annotation.append(anno)
                    break

        if len(new_annotation) == 0:
            new_annotation = np.random.choice(annotation,1,replace=False)
        return new_annotation
        
            
    def __getitem__(self, index):
        if index < len(self.image_files):
            image_file = self.image_files[index]
            studyid = image_file.split('.')[0].split('_')[0]
            img_file = os.path.join(self.data_dir,'fetal_side',image_file)
            img_seg = os.path.join(self.data_dir,'fetal_seg',image_file.replace('.jpg','.png'))
            annotation = self.annotations[studyid]
            annotation_feats = self.annotation_feats[studyid]
            # filter out some topics
            annotation = self.filter_topic(annotation)
            if self.target_dir is not None and random.random() <= self.wb_aug_prob:
                img = self.wb_color_aug.open_with_wb_aug(img_file, self.target_dir)
            else:
                img = Image.open(img_file)
        else: # addtional NU data
            index = index - len(self.image_files)
            image_file = self.additional_images_files[index]
            studyid = image_file.split('.')[0].split('_')[0]
            img_file = os.path.join(self.data_dir,'additional_fetal_side',image_file)
            img_seg = os.path.join(self.data_dir,'additional_fetal_seg',image_file.replace('.jpg','.png'))
            annotation = self.additional_annotations[studyid]
            annotation_feats = self.additional_annotation_feats[studyid]
            # filter out some topics
            annotation = self.filter_topic(annotation)
            if self.additional_target_dir is not None and random.random() <= self.wb_aug_prob:
                img = self.wb_color_aug.open_with_wb_aug(img_file, self.additional_target_dir)
            else:
                img = Image.open(img_file)
        # img = Image.open(img_file)
        seg = Image.open(img_seg)

        # process images 
        if self.img_transforms is not None:
            if isinstance(self.img_transforms,transforms.Compose):
                img = self.img_transforms(img)
            else:
                # TODO: better way to resize
                img = self.img_transforms(image=np.array(img.resize((512,384), resample=Image.BICUBIC)))['image'] # ['image'] is required for albumentations, resize is a work around to ensure consistent result with pil
                # convert to PIL
                img = Image.fromarray(img)
        if self.seg_transforms is not None:
            seg = self.seg_transforms(seg) > 0
        if self.transforms is not None:
            img, seg = self.transforms(img,seg)
        if self.mask_img:
            img = img * seg
        one_annotation = np.random.choice(annotation,1,replace=False)[0]
        if self.text_sample_method == 'group':
            choice_indices = np.random.choice(len(annotation),len(annotation),replace=False)
            annotation = " ".join(np.array(annotation)[choice_indices])
            annotation_feats = np.array(annotation_feats)[choice_indices].mean(0)
        elif self.text_sample_method == 'boot_group':
            choice_indices = np.random.choice(len(annotation),len(annotation),replace=True)
            annotation = " ".join(np.array(annotation)[choice_indices])
            annotation_feats = np.array(annotation_feats).mean(0)
            # annotation_feats = np.array(annotation_feats)[choice_indices].mean(0)
        elif self.text_sample_method == 'fix_group':
            annotation = " ".join(annotation)
            annotation_feats = np.array(annotation_feats).mean(0)
        
        # print(img.size)
        # print(seg.size)

        return {
            'image': img,
            'seg':seg,
            'text': annotation,
            'one_text': one_annotation,
            'text_feat': annotation_feats
        }

    
    def __len__(self):
        return len(self.image_files)+len(self.additional_images_files)

class NUFinetuneDataset(Dataset):
    """
        data_dir: directory where all data and annotations are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None, seg_transforms=None, mask_img=True, text_sample_method='group', use_2017_as_test=True):
        self.config = config
        self.data_dir = config.data_dir
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
        self.split_type = split_type
        self.mask_img = mask_img
        self.text_sample_method = text_sample_method
        self.label_dict = load_test_file(os.path.join(self.data_dir,'placenta_test_labels.csv'))
            
        sepsis_label_dict = load_test_file(os.path.join(self.data_dir,'sepsis_labels.csv'))
        # clinical_dict = None
        # if use_clinical:
        clinical_dict = load_clinical_info(os.path.join(self.data_dir,'clinical_info.csv'))
        sepsis_clinical_dict = load_clinical_info(os.path.join(self.data_dir,'sepsis_clinical_info.csv'))
        studyids = list(self.label_dict.keys())
        image_dir = os.path.join(self.data_dir,'fetal_side')
        images_files = os.listdir(image_dir)
        self.image_files = list()
        topics = ['meconium','insertion','hypercoiled','fir','mir','chorioamnionitis','sepsis']
        drop_percentage = [0.65,0.85,0.8,0.8,0.6,0.8,0.9] # percentage
        #drop_percentage = [0.65,0.85,0.8,0.85,0.65,0.8,0.9] # percentage if keep stage 1 for fir and mir
        num_classes_list = [1,1,1,1,1,1,1]#[2,5,2,3,3,2]
        for val in images_files:
            studyid = val.split('.')[0].split('_')[0]
            if studyid in studyids:
                self.image_files.append(val)
        # print(self.image_files)
        np.random.seed(100)
        np.random.shuffle(self.image_files)
        num_train = int(len(self.image_files)*0.8)
        if split_type == 'train':
            self.image_files = self.image_files[:num_train]
        else:
            self.image_files = self.image_files[num_train:]
        # if use_2017_as_test:
        #     test_set_dict = load_test_file(os.path.join(self.data_dir,'placenta_test_labels.csv'))
        # else:
        #     test_set_dict = load_mir_fir_file(os.path.join(self.data_dir,'fir_labels.txt'))
        # # clinical_dict = None
        # # if use_clinical:
        # clinical_dict = load_clinical_info(os.path.join(self.data_dir,'clinical_info.csv'))
        # #print(np.unique(list(test_set_dict.values()),return_counts=True))
        # test_studyids = list(test_set_dict.keys())
        # annotation_file = os.path.join(self.data_dir,'annotation_diagnosis_new.json')
        # image_dir = os.path.join(self.data_dir,'fetal_side')
        # images_files = os.listdir(image_dir)
        # with open(annotation_file, 'r') as f:
        #     self.annotations = json.load(f)
        # image_path_to_caption = defaultdict(list)
        # image_label = defaultdict(int)

        # clinical_data = defaultdict(np.ndarray)
        # self.image_files = list()
        # for val in images_files:
        #     studyid = val.split('.')[0].split('_')[0]
        #     if studyid not in self.annotations.keys():
        #         continue
        #     if split_type=='test':
        #         if studyid in test_studyids:
        #             self.image_files.append(val)
        #     else:
        #         if studyid not in test_studyids:
        #             self.image_files.append(val)

            
    def __getitem__(self, index):
        image_file = self.image_files[index]
        studyid = image_file.split('.')[0].split('_')[0]
        img_file = os.path.join(self.data_dir,'fetal_side',image_file)
        img_seg = os.path.join(self.data_dir,'fetal_seg',image_file.replace('.jpg','.png'))
        label = (np.array(self.label_dict[studyid]) > np.array([0,1,1,0])).astype(int)
        img = Image.open(img_file)
        seg = Image.open(img_seg)

        # process images 
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        if self.seg_transforms is not None:
            seg = self.seg_transforms(seg) > 0
        if self.mask_img:
            img = img * seg
        annotation = ['meconium-laden macrophages in the amnion and chorion.','fetal inflammatory response, stage 2.','maternal inflammatory response, stage 2.','membranes with acute necrotizing chorioamnionitis.']
        annotation = " ".join(annotation)
        # if self.text_sample_method == 'group':
        #     annotation = " ".join(np.random.choice(annotation,len(annotation),replace=False))
        # elif self.text_sample_method == 'sample':
        #     annotation = np.random.choice(annotation,1,replace=False)
        # print(img.size)
        # print(seg.size)

        return {
            'studyid': studyid,
            'image': img,
            'seg':seg,
            'label': label,
            'text': annotation
        }

    
    def __len__(self):
        return len(self.image_files)


class NUFinetuneSplitsDataset(Dataset):
    """
        data_dir: directory where all data and annotations are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, 
                config: Config, topic, 
                random_seed = 100, 
                split_type = 'train', 
                transforms = None, 
                img_transforms=None, 
                seg_transforms=None, 
                mask_img=True, 
                text_sample_method='group', 
                wb_aug_prob=0.0,
                wb_setting = None,
                use_2017_as_test=True):
        self.config = config
        self.topic = topic
        self.data_dir = config.data_dir
        self.transforms = transforms
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
        self.split_type = split_type
        self.mask_img = mask_img
        self.text_sample_method = text_sample_method
        self.wb_aug_prob = wb_aug_prob
        self.wb_setting = wb_setting
        self.label_dict = load_test_file(os.path.join(self.data_dir,'placenta_test_labels.csv'))
        random.seed(random_seed) # deterministic transform if applicable
        sepsis_label_dict = load_test_file(os.path.join(self.data_dir,'sepsis_labels.csv'))
        # clinical_dict = None
        # if use_clinical:
        clinical_dict = load_clinical_info(os.path.join(self.data_dir,'clinical_info.csv'))
        sepsis_clinical_dict = load_clinical_info(os.path.join(self.data_dir,'sepsis_clinical_info.csv'))
        studyids = list(self.label_dict.keys())
        image_dir = os.path.join(self.data_dir,'fetal_side')
        images_files = os.listdir(image_dir)
        self.image_files = list()
        topics = ['meconium','insertion','hypercoiled','fir','mir','chorioamnionitis','sepsis']
        drop_percentage = [0.65,0.85,0.8,0.8,0.6,0.8,0.9] # percentage
        #drop_percentage = [0.65,0.85,0.8,0.85,0.65,0.8,0.9] # percentage if keep stage 1 for fir and mir
        num_classes_list = [1,1,1,1,1,1,1]#[2,5,2,3,3,2]
        for val in images_files:
            studyid = val.split('.')[0].split('_')[0]
            if studyid in studyids:
                self.image_files.append(val)
        # print(self.image_files)
        num_train = int(len(self.image_files)*0.8)
        if split_type == 'train+val':
            self.image_files = None
            for split_type in ['train', 'val']:
                split_path = os.path.join(self.data_dir,"evaluation_splits",f'splits_{topic}_{random_seed}.json')
                with open(split_path, 'r') as f:
                    split_json = json.load(f)
                if self.image_files is None:
                    self.image_files = np.array(split_json[split_type]['images'])
                    self.labels = np.array(split_json[split_type]['labels'])
                else:
                    self.image_files = np.concatenate([self.image_files,np.array(split_json[split_type]['images'])],axis=0)
                    self.labels = np.concatenate([self.labels,np.array(split_json[split_type]['labels'])],axis=0)
            print(len(self.image_files))   
        else:
            split_path = os.path.join(self.data_dir,"evaluation_splits",f'splits_{topic}_{random_seed}.json')
            with open(split_path, 'r') as f:
                split_json = json.load(f)
            self.image_files = np.array(split_json[split_type]['images'])
            print(len(self.image_files))
            self.labels = np.array(split_json[split_type]['labels'])
        if wb_aug_prob > 0:
            self.wb_color_aug = wbAug.WBEmulator()
            target_files = []
            for image_file in self.image_files:
                if self.topic == 'sepsis':
                    img_file = os.path.join(self.data_dir,'sepsis_fetal_side',image_file)
                    if not os.path.exists(img_file):
                        img_file = os.path.join(self.data_dir,'fetal_side',image_file)
                else:
                    img_file = os.path.join(self.data_dir,'fetal_side',image_file)
                target_files.append(img_file)
            target_dir = os.path.join(self.data_dir,f'mfs_{topic}_{random_seed}')
            if not os.path.exists(target_dir):
                self.target_dir = self.wb_color_aug.precompute_mfs(target_files,target_dir=target_dir)
            else:
                self.target_dir = target_dir
        else:
            self.target_dir = None
        # if use_2017_as_test:
        #     test_set_dict = load_test_file(os.path.join(self.data_dir,'placenta_test_labels.csv'))
        # else:
        #     test_set_dict = load_mir_fir_file(os.path.join(self.data_dir,'fir_labels.txt'))
        # # clinical_dict = None
        # # if use_clinical:
        # clinical_dict = load_clinical_info(os.path.join(self.data_dir,'clinical_info.csv'))
        # #print(np.unique(list(test_set_dict.values()),return_counts=True))
        # test_studyids = list(test_set_dict.keys())
        # annotation_file = os.path.join(self.data_dir,'annotation_diagnosis_new.json')
        # image_dir = os.path.join(self.data_dir,'fetal_side')
        # images_files = os.listdir(image_dir)
        # with open(annotation_file, 'r') as f:
        #     self.annotations = json.load(f)
        # image_path_to_caption = defaultdict(list)
        # image_label = defaultdict(int)

        # clinical_data = defaultdict(np.ndarray)
        # self.image_files = list()
        # for val in images_files:
        #     studyid = val.split('.')[0].split('_')[0]
        #     if studyid not in self.annotations.keys():
        #         continue
        #     if split_type=='test':
        #         if studyid in test_studyids:
        #             self.image_files.append(val)
        #     else:
        #         if studyid not in test_studyids:
        #             self.image_files.append(val)

            
    def __getitem__(self, index):
        image_file = self.image_files[index]
        studyid = image_file.split('.')[0].split('_')[0]
        if self.topic == 'sepsis':
            
            img_file = os.path.join(self.data_dir,'sepsis_fetal_side',image_file)
            img_seg = os.path.join(self.data_dir,'sepsis_fetal_seg',image_file.replace('.jpg','.png'))
            if not os.path.exists(img_file):
                img_file = os.path.join(self.data_dir,'fetal_side',image_file)
                img_seg = os.path.join(self.data_dir,'fetal_seg',image_file.replace('.jpg','.png'))
        else:
            img_file = os.path.join(self.data_dir,'fetal_side',image_file)
            img_seg = os.path.join(self.data_dir,'fetal_seg',image_file.replace('.jpg','.png'))
        label = self.labels[index]
        # topic_idx = ['meconium','fir','mir','chorioamnionitis','sepsis'].index(self.topic)
        # label = (self.label_dict[studyid] > [0,1,1,0]).astype(int)[topic_idx]
        # label = np.zeros(2)
        # label[label_idx] = 1

        # label = (np.array(self.label_dict[studyid]) > np.array([0,1,1,0])).astype(int)
        if self.target_dir is not None and random.random() <= self.wb_aug_prob:
            img = self.wb_color_aug.open_with_wb_aug(img_file, self.target_dir,self.wb_setting)
        else:
            img = Image.open(img_file)
        seg = Image.open(img_seg)

        # process images 
        if self.img_transforms is not None:
            # print(self.img_transforms) 
            # TODO: better way to resize
            img = self.img_transforms(image=np.array(img.resize((512,384), resample=Image.BICUBIC)))['image'] # ['image'] is required for albumentations, resize is a work around to ensure consistent result with pil
        if self.seg_transforms is not None:
            seg = self.seg_transforms(seg) > 0
        if self.transforms is not None:
            img, seg = self.transforms(img,seg)
        if self.mask_img:
            img = img * seg
        annotation = ['meconium-laden macrophages in the amnion and chorion.','fetal inflammatory response.','maternal inflammatory response.','membranes with acute necrotizing chorioamnionitis.']
        annotation = " ".join(annotation)
        # if self.text_sample_method == 'group':
        #     annotation = " ".join(np.random.choice(annotation,len(annotation),replace=False))
        # elif self.text_sample_method == 'sample':
        #     annotation = np.random.choice(annotation,1,replace=False)
        # print(img.size)
        # print(seg.size)

        return {
            'studyid':studyid,
            'image': img,
            'seg':seg,
            'label': label,
            'text': annotation,
            'topic_text': TOPIC_TEXT[self.topic]
        }

    
    def __len__(self):
        return len(self.image_files)


class NUIpadDataset(Dataset):
    """
        data_dir: directory where all data and annotations are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config, topic, random_seed = 100, split_type = 'train', img_transforms=None, seg_transforms=None, mask_img=True, text_sample_method='group', use_2017_as_test=True):
        self.config = config
        self.topic = topic
        self.data_dir = config.data_dir
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
        self.split_type = split_type
        self.mask_img = mask_img
        self.text_sample_method = text_sample_method
        image_dir = os.path.join(self.data_dir,'NU_ipad_anno_fetal_side')
        self.images_files = os.listdir(image_dir)

        with open(os.path.join(self.data_dir,'ipad_label.json'),'r') as f:
            image_label_file = json.load(f)[topic.split('_')[0]]
        self.image_paths = [val for val in self.images_files if val.split('_')[1].split('F')[0] in image_label_file.keys()]
        # print(self.image_paths)
        image_labels = [image_label_file[val.split('/')[-1].split('_')[1].split('F')[0]] for val in self.image_paths]
        # num_examples = len(self.image_paths)
        self.labels = np.array(image_labels)
        # sample_weights = np.zeros(len(self.labels))
        # new_image_paths = np.array(self.image_paths)
        # # fake clinicals
        # clinicals = [np.array([32.013719,35.648668,0.361212],dtype=np.float32) for _ in range(len(new_image_paths))]
        # clinicals = np.stack(clinicals,0)
        # num of all topics
        
        # the last element will be the filename
        
        # if use_2017_as_test:
        #     test_set_dict = load_test_file(os.path.join(self.data_dir,'placenta_test_labels.csv'))
        # else:
        #     test_set_dict = load_mir_fir_file(os.path.join(self.data_dir,'fir_labels.txt'))
        # # clinical_dict = None
        # # if use_clinical:
        # clinical_dict = load_clinical_info(os.path.join(self.data_dir,'clinical_info.csv'))
        # #print(np.unique(list(test_set_dict.values()),return_counts=True))
        # test_studyids = list(test_set_dict.keys())
        # annotation_file = os.path.join(self.data_dir,'annotation_diagnosis_new.json')
        # image_dir = os.path.join(self.data_dir,'fetal_side')
        # images_files = os.listdir(image_dir)
        # with open(annotation_file, 'r') as f:
        #     self.annotations = json.load(f)
        # image_path_to_caption = defaultdict(list)
        # image_label = defaultdict(int)

        # clinical_data = defaultdict(np.ndarray)
        # self.image_files = list()
        # for val in images_files:
        #     studyid = val.split('.')[0].split('_')[0]
        #     if studyid not in self.annotations.keys():
        #         continue
        #     if split_type=='test':
        #         if studyid in test_studyids:
        #             self.image_files.append(val)
        #     else:
        #         if studyid not in test_studyids:
        #             self.image_files.append(val)

            
    def __getitem__(self, index):
        image_file = self.image_paths[index]
        studyid = image_file.split('/')[-1].split('_')[1].split('F')[0]
        if 'placentanet' in self.topic:
            
            img_file = os.path.join(self.data_dir,'NU_ipad_fetal_side',image_file)
            img_seg = os.path.join(self.data_dir,'NU_ipad_fetal_seg',image_file.replace('.jpg','.png'))
            # if not os.path.exists(img_file):
            #     img_file = os.path.join(self.data_dir,'fetal_side',image_file)
            #     img_seg = os.path.join(self.data_dir,'fetal_seg',image_file.replace('.jpg','.png'))
        else:
            img_file = os.path.join(self.data_dir,'NU_ipad_anno_fetal_side',image_file)
            img_seg = os.path.join(self.data_dir,'NU_ipad_anno_fetal_seg',image_file.replace('.jpg','.png'))
        label = self.labels[index]
        # topic_idx = ['meconium','fir','mir','chorioamnionitis','sepsis'].index(self.topic)
        # label = (self.label_dict[studyid] > [0,1,1,0]).astype(int)[topic_idx]
        # label = np.zeros(2)
        # label[label_idx] = 1

        # label = (np.array(self.label_dict[studyid]) > np.array([0,1,1,0])).astype(int)
        img = Image.open(img_file)
        seg = Image.open(img_seg)

        # process images 
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        if self.seg_transforms is not None:
            seg = self.seg_transforms(seg) > 0
        if self.mask_img:
            img = img * seg
        annotation = ['meconium-laden macrophages in the amnion and chorion.','fetal inflammatory response.','maternal inflammatory response.','membranes with acute necrotizing chorioamnionitis.']
        annotation = " ".join(annotation)
        # if self.text_sample_method == 'group':
        #     annotation = " ".join(np.random.choice(annotation,len(annotation),replace=False))
        # elif self.text_sample_method == 'sample':
        #     annotation = np.random.choice(annotation,1,replace=False)
        # print(img.size)
        # print(seg.size)

        return {
            'studyid':studyid,
            'image': img,
            'seg':seg,
            'label': label,
            'text': annotation,
            'topic_text': TOPIC_TEXT[self.topic]
        }

    
    def __len__(self):
        return len(self.image_paths)

class NUTextDataset(Dataset):
    """
        data_dir: directory where all data and annotations are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, topic, random_seed = 100, split_type = 'train', transforms = None, img_transforms=None, seg_transforms=None, mask_img=True, text_sample_method='group', use_2017_as_test=True):
        self.config = config
        self.data_dir = config.data_dir
        self.transforms = transforms
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
        self.split_type = split_type
        self.mask_img = mask_img
        self.text_sample_method = text_sample_method
        # clinical_dict = None
        # if use_clinical:
        # clinical_dict = load_clinical_info(os.path.join(self.data_dir,'clinical_info.csv'))
        # #print(np.unique(list(test_set_dict.values()),return_counts=True))
        # test_studyids = list(test_set_dict.keys())
        annotation_file = os.path.join(self.data_dir,'annotation_diagnosis_new.json')
        annotation_feat_file = os.path.join(self.data_dir,'annotation_diagnosis_new.npy')
        image_dir = os.path.join(self.data_dir,'fetal_side')
        images_files = os.listdir(image_dir)
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        # print(len(annotation_set))
        self.annotation_feats = np.load(annotation_feat_file, allow_pickle='TRUE').item()
        annotation_set = list()
        annotation_feats_set = list()
        for key in annotations.keys():
            annotation_set += annotations[key]
            annotation_feats_set.append(self.annotation_feats[key])
        annotation_set, indices = np.unique(annotation_set,return_index=True)
        annotation_feats_set = np.concatenate(annotation_feats_set,0)[indices]
        topic_list = ['meconium','fir','mir','chorioamnionitis']
        assert topic in topic_list
        topic_id = topic_list.index(topic)
        topic_cap = ['meconium','fetal inflammatory response','maternal inflammatory response','chorioamnion']  
        topic_cap_mask = np.vectorize(lambda x: topic_cap[topic_id] in x)(annotation_set)
        if topic in ['mir','fir']:
            topic_set_mask = np.vectorize(lambda x: 'stage 2' in x or 'stage 3' in x)(annotation_set)
            topic_set = annotation_set[topic_set_mask]
            topic_feat_set = annotation_feats_set[topic_set_mask]
        else:    
            topic_set = annotation_set[topic_cap_mask]
            topic_feat_set = annotation_feats_set[topic_cap_mask]
        self.topic_support_set = annotation_set[~topic_cap_mask]
        self.topic_feat_support_set = annotation_feats_set[~topic_cap_mask]
        self.topic_set = topic_set
        self.topic_feat_set = topic_feat_set
        self.num_samples = 2000
        self.labels = np.random.randint(0, high=2, size=self.num_samples, dtype=int)
        
            
    def __getitem__(self, index):
        label = self.labels[index]
        if label == 1:
            idx = np.random.choice(len(self.topic_set),1,replace=True)
            cap = self.topic_set[idx]
            cap = ' '.join(cap)
            cap_feats = self.topic_feat_set[idx].mean(0)
        else:
            num_samples = 10 #np.random.choice(np.arange(1,11),1)[0]
            idx = np.random.choice(len(self.topic_set),num_samples)
            cap = self.topic_support_set[idx]
            cap = ' '.join(cap)
            cap_feats = self.topic_feat_support_set[idx].mean(0)
        
        # print(img.size)
        # print(seg.size)

        return {
            'label': label,
            'text': cap,
            'text_feat': cap_feats
        }

    
    def __len__(self):
        return self.num_samples

if __name__=="__main__":
    config = AllConfig().parse_args()
    img_trans = transforms.Compose([
            transforms.Resize((384,512), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.CenterCrop(input_res),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    seg_trans = transforms.Compose([
            transforms.Resize((384,512), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            # transforms.CenterCrop(input_res),
        ])
    dataset = NUIpadDataset(config,topic='mir',img_transforms=img_trans,seg_transforms=seg_trans)
    img = (dataset[0]['image'].permute(1,2,0).numpy()*255).astype(np.uint8)
    print(img[244,244])
    img = Image.fromarray(img)
    img.save(f'/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta_torch/dumps/ipad_example.jpg')
    # dataset = NUPreTrainDataset(config, img_transforms=img_trans,seg_transforms=seg_trans, wb_aug_prob=1.0)
    # print(dataset[0])
    # dataset = NUFinetuneDataset(config, img_transforms=img_trans,seg_transforms=seg_trans)
    # print(len(dataset))
    # dataset = NUFinetuneDataset(config,split_type='val', img_transforms=img_trans,seg_transforms=seg_trans)
    # print(len(dataset))
    # dataset = NUFinetuneSplitsDataset(config,topic='sepsis',random_seed=100, split_type='val', img_transforms=None,seg_transforms=seg_trans,mask_img=False, wb_aug_prob=1.0)
    # sum_dt = None
    # image_file = 'ADWVIF_2.jpg'
    # for i in range(10):
    #     img_file = os.path.join(dataset.data_dir,'fetal_side',image_file)
    #     img = dataset.wb_color_aug.open_with_wb_aug(img_file, dataset.target_dir,i)
    #     img.save(f'/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta_torch/dumps/wb_example_{i}.jpg')
    # dataset = NUTextDataset(config,topic='mir',random_seed=100, split_type='val', img_transforms=None,seg_transforms=seg_trans,mask_img=False)
    # print(dataset.labels.sum())
    # for dt in range(100):
    #     if sum_dt is None:
    #         print(np.array(dataset[dt]['image']).shape)
    #         sum_dt = np.array(dataset[dt]['image']).mean((0,1))
    #     else:
    #         sum_dt += np.array(dataset[dt]['image']).mean((0,1))
    # print(sum_dt/100)