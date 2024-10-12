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

def load_test_file(filename,task_name):
    test_df = pd.read_csv(filename)
    if task_name == 'meconium':
        sample_label = test_df.meconium.map(lambda finding: ['1. Absent','2. Present','3. Not recorded'].index(finding) if finding is not np.nan else -1)
        sample_label_mask = np.logical_and(sample_label >=0, sample_label <2)
        sample_label = sample_label[sample_label_mask]
        sample_id = test_df.study_id[sample_label_mask]
    elif task_name == 'mir':
        sample_label = test_df.chorio_mat_stage.map(lambda finding: ['1. Absent','2. Stage 1','3. Stage 2', '4. Stage 3'].index(finding) if finding is not np.nan else -1)
        sample_label_mask = np.logical_and(sample_label >=0, sample_label !=1)
        sample_label = (sample_label[sample_label_mask] > 1).astype(int)
        sample_id = test_df.study_id[sample_label_mask]
    elif task_name == 'fir':
        sample_label = test_df.chorio_fet_stage.map(lambda finding: ['1. Absent','2. Stage 1','3. Stage 2', '4. Stage 3'].index(finding) if finding is not np.nan else -1)
        sample_label_mask = np.logical_and(sample_label >=0, sample_label !=1)
        sample_label = (sample_label[sample_label_mask] > 1).astype(int)
        sample_id = test_df.study_id[sample_label_mask]
    return dict(zip(sample_id.values,sample_label.values))


def load_clinical_info(filename):
    clinical_df = pd.read_csv(filename)

    # fillna with mean
    clinical_df = clinical_df.fillna(value=clinical_df[['Maternal age','Gestational age','delivery']].mean())
    return dict(zip(clinical_df['studyid'].values,clinical_df[['Maternal age','Gestational age','delivery']].values))



class MustDataset(Dataset):
    """
        data_dir: directory where all data and annotations are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config, topic, random_seed = 100, split_type = 'test', img_transforms=None, seg_transforms=None, mask_img=True, text_sample_method='group', use_2017_as_test=True):
        self.config = config
        self.topic = topic
        self.data_dir = '/ocean/projects/iri180005p/ymp5078/VLPlacenta/data/MUST_data'
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
        self.split_type = split_type
        self.mask_img = mask_img
        self.text_sample_method = text_sample_method
        image_dir = os.path.join(self.data_dir,'named_data')

        self.labels_dict = load_test_file(os.path.join(self.data_dir,'PACO_data_for_Kalynn_23_Mar_23.csv'),task_name=topic)
        self.image_paths = self.process_image_path(image_dir=image_dir,case_ids=list(self.labels_dict.keys()))

    def process_image_path(self,image_dir,case_ids):
        image_paths = []
        for case in os.listdir(image_dir):
            if int(case) in case_ids and os.path.exists(os.path.join(image_dir,case,'fetal')):
                image_paths += [os.path.join(image_dir,case,'fetal',image_path) for image_path in os.listdir(os.path.join(image_dir,case,'fetal')) if ' 2.' not in image_path and os.path.exists(os.path.join(image_dir,case,'fetal',image_path).replace('.JPG','.png').replace('named_data','MUST_seg'))]
        return image_paths
            
    def __getitem__(self, index):
        image_file = self.image_paths[index]
        studyid = int(image_file.split('/')[-3])
        label = self.labels_dict[studyid]
        image_seg = image_file.replace('.JPG','.png').replace('named_data','MUST_seg')

        img = Image.open(image_file)
        seg = Image.open(image_seg)

        # process images 
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        if self.seg_transforms is not None:
            # print(self.seg_transforms(seg).unique())
            seg = self.seg_transforms(seg) > 0
        if self.mask_img:
            img = img * seg
        annotation = ['meconium-laden macrophages in the amnion and chorion.','fetal inflammatory response.','maternal inflammatory response.','membranes with acute necrotizing chorioamnionitis.']
        annotation = " ".join(annotation)

        return {
            'studyid': str(studyid)+'_'+image_file.split('/')[-1].replace('.JPG',''),
            'image': img,
            'seg':seg,
            'label': label,
            'text': annotation,
            'topic_text': TOPIC_TEXT[self.topic]
        }

    
    def __len__(self):
        return len(self.image_paths)

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