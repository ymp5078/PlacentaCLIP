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


class NUPreTrainDatasetForVis(Dataset):
    """
        data_dir: directory where all data and annotations are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', topic = None, transforms = None, img_transforms=None, seg_transforms=None, mask_img=True, text_sample_method='group', use_2017_as_test=True):
        self.config = config
        self.data_dir = config.data_dir
        self.topic = topic
        self.transforms = transforms
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
        self.split_type = split_type
        self.mask_img = mask_img
        self.text_sample_method = text_sample_method
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
        image_file = self.image_files[index]
        studyid = image_file.split('.')[0].split('_')[0]
        img_file = os.path.join(self.data_dir,'fetal_side',image_file)
        img_seg = os.path.join(self.data_dir,'fetal_seg',image_file.replace('.jpg','.png'))
        annotation = self.annotations[studyid]
        annotation_feats = self.annotation_feats[studyid]
        # filter out some topics
        annotation = self.filter_topic(annotation)
        img = Image.open(img_file)
        seg = Image.open(img_seg)

        # process images 
        if self.img_transforms is not None:
            img = self.img_transforms(img)
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
            'studyid':studyid,
            'image': img,
            'seg':seg,
            'text': annotation,
            'one_text': one_annotation,
            'text_feat': annotation_feats,
            'topic_text': TOPIC_TEXT[self.topic]
        }
    
    
    def __len__(self):
        return len(self.image_files) // 10

class NUPreTrainDataset(Dataset):
    """
        data_dir: directory where all data and annotations are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', transforms = None, img_transforms=None, seg_transforms=None, mask_img=True, text_sample_method='group', use_2017_as_test=True):
        self.config = config
        self.data_dir = config.data_dir
        self.transforms = transforms
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
        self.split_type = split_type
        self.mask_img = mask_img
        self.text_sample_method = text_sample_method
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
        image_file = self.image_files[index]
        studyid = image_file.split('.')[0].split('_')[0]
        img_file = os.path.join(self.data_dir,'fetal_side',image_file)
        img_seg = os.path.join(self.data_dir,'fetal_seg',image_file.replace('.jpg','.png'))
        annotation = self.annotations[studyid]
        annotation_feats = self.annotation_feats[studyid]
        # filter out some topics
        annotation = self.filter_topic(annotation)
        img = Image.open(img_file)
        seg = Image.open(img_seg)

        # process images 
        if self.img_transforms is not None:
            img = self.img_transforms(img)
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
        return len(self.image_files)

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
    def __init__(self, config: Config, topic, random_seed = 100, split_type = 'train', transforms=None, img_transforms=None, seg_transforms=None, mask_img=True, text_sample_method='group', use_2017_as_test=True):
        self.config = config
        self.topic = topic
        self.data_dir = config.data_dir
        self.transforms = transforms
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
        img = Image.open(img_file)
        seg = Image.open(img_seg)

        # process images 
        if self.img_transforms is not None:
            img = self.img_transforms(img)
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
            'text_feat': cap_feats,
            'topic_text': TOPIC_TEXT[self.topic]
        }

    
    def __len__(self):
        return self.num_samples

if __name__=="__main__":
    config = AllConfig().parse_args()
    img_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512,384), interpolation=Image.BICUBIC),
            # transforms.CenterCrop(input_res),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    seg_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512,384), interpolation=Image.NEAREST),
            # transforms.CenterCrop(input_res),
        ])
    dataset = NUPreTrainDataset(config, img_transforms=img_trans,seg_transforms=seg_trans)
    print(dataset[0])
    # dataset = NUFinetuneDataset(config, img_transforms=img_trans,seg_transforms=seg_trans)
    # print(len(dataset))
    # dataset = NUFinetuneDataset(config,split_type='val', img_transforms=img_trans,seg_transforms=seg_trans)
    # print(len(dataset))
    dataset = NUFinetuneSplitsDataset(config,topic='mir',random_seed=100, split_type='val', img_transforms=None,seg_transforms=seg_trans,mask_img=False)
    sum_dt = None
    dataset = NUTextDataset(config,topic='mir',random_seed=100, split_type='val', img_transforms=None,seg_transforms=seg_trans,mask_img=False)
    print(dataset.labels.sum())
    # for dt in range(100):
    #     if sum_dt is None:
    #         print(np.array(dataset[dt]['image']).shape)
    #         sum_dt = np.array(dataset[dt]['image']).mean((0,1))
    #     else:
    #         sum_dt += np.array(dataset[dt]['image']).mean((0,1))
    # print(sum_dt/100)