from torchvision import transforms
from PIL import Image
import random

def init_transform_dict(input_res=224):
    tsfm_dict = {
        'clip_test': transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]),
        'clip_train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0, saturation=0, hue=0),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    }

    return tsfm_dict


class RandomRotate(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, degrees, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def __call__(self, img, seg):
        angle = random.randint(self.degrees[0],self.degrees[1])
        img = transforms.functional.rotate(img=img, angle=angle, interpolation = self.interpolation, expand = self.expand, center = self.center, fill = self.fill) 
        seg = transforms.functional.rotate(img=seg, angle=angle, interpolation = transforms.InterpolationMode.NEAREST, expand = self.expand, center = self.center, fill = 0) 
        img = self.to_tensor(img)
        img = self.normalize(img)
        # seg = self.to_tensor(seg)

        return img, seg


def NU_transform_dict(input_res=(384,512)):
    
    train_img_trans = transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.05,hue=0.05),
            # transforms.ToTensor(),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    train_img_seg_trans = RandomRotate(degrees=[-180,180],interpolation=transforms.InterpolationMode.BICUBIC)


    val_img_trans = transforms.Compose([
            # transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    val_img_trans_ipad = transforms.Compose([
            # transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.05,hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # transforms.Normalize((0.48145466+0.35, 0.4578275-0.15, 0.40821073-0.2), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    seg_trans = transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])
    
    return {
        'train_img_trans': train_img_trans,
        'val_img_trans_ipad':val_img_trans_ipad,
        'train_img_seg_trans':train_img_seg_trans,
        'val_img_trans':val_img_trans,
        'seg_trans':seg_trans,
    }

def must_transform_dict(input_res=(384,512)):
    
    # train_img_trans = transforms.Compose([
    #         transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC),
    #         transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.05,hue=0.05),
    #         # transforms.ToTensor(),
    #         # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    #     ])
    # train_img_seg_trans = RandomRotate(degrees=[-180,180],interpolation=transforms.InterpolationMode.BICUBIC)


    val_img_trans = transforms.Compose([
            # transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
            transforms.Resize(input_res[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    
    seg_trans = transforms.Compose([
            transforms.Resize(input_res[0], interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(input_res),
            transforms.PILToTensor(),
        ])
    
    return {
        'val_img_trans':val_img_trans,
        'seg_trans':seg_trans,
    }