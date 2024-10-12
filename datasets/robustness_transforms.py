import numpy as np
import albumentations as A
import random
from PIL import Image
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter
from albumentations import random_utils
from albumentations.augmentations.blur.functional import blur
from albumentations.augmentations import functional as F
from albumentations.augmentations.utils import (
    MAX_VALUES_BY_DTYPE,
    _maybe_process_in_chunks,
    clip,
    clipped,
    ensure_contiguous,
    is_grayscale_image,
    is_rgb_image,
    non_rgb_warning,
    preserve_channel_dim,
    preserve_shape,
)
import cv2

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@ensure_contiguous
@preserve_shape
def add_shadow(img, vertices_list, random_subset=False):
    """Add shadows to the image.
    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    Args:
        img (numpy.ndarray):
        vertices_list (list):
    Returns:
        numpy.ndarray:
    """
    non_rgb_warning(img)
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = F.from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomShadow augmentation".format(input_dtype))

    # image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    width, height = img.shape[:2]
    # print(width,height)
    mask = np.zeros(img.shape[:2])

    # adding all shadow polygons on empty mask, single 255 denotes only red channel
    if random_subset:
        vertices_list = vertices_list[:random.randint(0, len(vertices_list))]
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255)
    # Draw ovals on the image
    num_ovals = len(vertices_list)
    for i in range(num_ovals):
        # Generate random oval parameters
        center_x = random.randint(0, height)
        center_y = random.randint(0, width)
        axis_x = random.randint(50, 200)
        axis_y = random.randint(50, 200)
        angle = random.randint(0, 360)
        
        # Draw the oval on the image
        cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, 255, thickness=-1)


    # Create a random Gaussian blur kernel size and sigma value to simulate the soft edges of the shadow
    kernel_size = random.randint(int(max(img.shape[:2]) / 80), int(max(img.shape[:2]) / 30))
    sigma = random.uniform(0, 5)
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel * kernel.T

    # Apply the blur to the shadow mask
    mask = cv2.filter2D(mask, -1, kernel) / 255.
    
    # apply shadow

    shadow_darkness = random.uniform(0.2+0.08*len(vertices_list),0.2+0.08*len(vertices_list)+0.1)
    image_rgb = img * (1 - shadow_darkness * np.expand_dims(mask,-1))
    # if red channel is hot, image's "Lightness" channel's brightness is lowered
    # red_max_value_ind = mask[:, :, 0] == 255
    # image_hls[:, :, 1][red_max_value_ind] = image_hls[:, :, 1][red_max_value_ind] * 0.5

    # image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    if needs_float:
        image_rgb = F.to_float(image_rgb, max_value=255)

    return image_rgb.astype(np.uint8)

class Shadow(A.augmentations.transforms.RandomShadow):
    def apply(self, image, vertices_list=(), **params):
        return add_shadow(image, vertices_list)

class MotionBlur(A.augmentations.MotionBlur):
    """Apply motion blur to the input image using a random-sized kernel.
    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        allow_shifted (bool): if set to true creates non shifted kernels only,
            otherwise creates randomly shifted kernels. Default: True.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: int = 7,
        allow_shifted: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)
        self.allow_shifted = allow_shifted
        self.blur_limit = (blur_limit,blur_limit)

        if not allow_shifted and self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError(f"Blur limit must be odd when centered=True. Got: {self.blur_limit}")



class Glare(A.augmentations.transforms.Spatter):
    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        h, w = params["image"].shape[:2]

        mean = random.uniform(self.mean[0], self.mean[1])
        std = random.uniform(self.std[0], self.std[1])
        cutout_threshold = random.uniform(self.cutout_threshold[0], self.cutout_threshold[1])
        sigma = random.uniform(self.gauss_sigma[0], self.gauss_sigma[1])
        mode = random.choice(self.mode)
        intensity = random.uniform(self.intensity[0], self.intensity[1])
        color = np.array(self.color[mode]) / 255.0

        liquid_layer = random_utils.normal(size=(h, w), loc=mean, scale=std)
        liquid_layer = gaussian_filter(liquid_layer, sigma=sigma, mode="nearest")
        liquid_layer[liquid_layer < cutout_threshold] = 0

        if mode == "rain":
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = blur(dist, 3).astype(np.uint8)
            dist = F.equalize(dist)

            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = F.convolve(dist, ker)
            dist = blur(dist, 3).astype(np.float32)

            m = liquid_layer * dist
            m *= 1 / np.max(m, axis=(0, 1))

            drops = m[:, :, None] * color * intensity
            mud = None
            non_mud = None
        else:
            m = np.where(liquid_layer > cutout_threshold, 1, 0)
            m = gaussian_filter(m.astype(np.float32), sigma=sigma, mode="nearest")
            m[m < 1.2 * cutout_threshold] = 0
            m = m[..., np.newaxis]
            m = gaussian_filter(m.astype(np.float32), sigma=0.5, mode="nearest")
            mud = m * color
            non_mud = 1 - m
            drops = None

        return {
            "non_mud": non_mud,
            "mud": mud,
            "drops": drops,
            "mode": mode,
        }

def transform_wrapper(transforms=None,input_res=(384,512),normalize=False,to_tensor=False):
    # def to_numpy(img, dtype=None, copy=True, order='K', subok=False, ndmin=0, like=None, **kwargs):
    #     img = np.array(img,dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like)
    #     return img

    # TODO: FIX resize, using cv2 resize generate different result for PIL.
    albumentations_pil_transform = [
            # A.Lambda(image=to_numpy,always_apply=True),
            # A.Resize(input_res[0],input_res[1],interpolation=2), 
        ]
    if transforms is not None:
        albumentations_pil_transform += transforms
    if normalize:
        albumentations_pil_transform.append(A.Normalize(
                                                mean=(0.48145466, 0.4578275, 0.40821073),
                                                std=(0.26862954, 0.26130258, 0.27577711),
                                            ))
    if to_tensor:
        albumentations_pil_transform.append(ToTensorV2())
    albumentations_pil_transform = A.Compose(albumentations_pil_transform)
    return albumentations_pil_transform

def blood(mode='blood',severity=1,always_apply=True,normalize=False,to_tensor=False):
    c = [(0.65, 0.3, 4, 0.68, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 1.0, 0),
         (0.65, 0.3, 2, 0.68, 1.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    color = [255,255,255] if mode=='glare' else [102, 0, 0]
    trans = Glare if mode == 'glare' else A.augmentations.transforms.Spatter
    aug_mode = 'mud'
    transforms = transform_wrapper([trans(mean=c[0], std=c[1], gauss_sigma=c[2], cutout_threshold=c[3], intensity=c[4], mode=aug_mode, color=color, always_apply=always_apply, p=1.0)],normalize=normalize,to_tensor=to_tensor)
    return transforms

def glare(mode='glare',severity=1,always_apply=True,normalize=False,to_tensor=False):
    c = [(0.61, 0.3, 1, 0.68, 0.6, 0),
         (0.62, 0.3, 1, 0.68, 1.0, 0),
         (0.62, 0.3, 1, 0.68, 1.5, 0),
         (0.62, 0.4, 1, 0.65, 1.5, 1),
         (0.63, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    color = [255,255,255] if mode=='glare' else [102, 0, 0]
    trans = Glare if mode == 'glare' else A.augmentations.transforms.Spatter
    aug_mode = 'mud'
    transforms = transform_wrapper([trans(mean=c[0], std=c[1], gauss_sigma=c[2], cutout_threshold=c[3], intensity=c[4], mode=aug_mode, color=color, always_apply=always_apply, p=1.0)],normalize=normalize,to_tensor=to_tensor)
    return transforms

# def shot_noise(x, severity=1):
#     c = [60, 25, 12, 5, 3][severity - 1]
#     transforms = transform_wrapper([A.augmentations.transforms.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=True, p=1)])
#     return transforms

def defocus_blur(severity=1,always_apply=True,normalize=False,to_tensor=False):
    c = [(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.3), (5, 0.4), (6, 0.5)][severity - 1]
    transforms = transform_wrapper([A.augmentations.Defocus(radius=(c[0],c[0]), alias_blur=(c[1],c[1]), always_apply=always_apply, p=1)],normalize=normalize,to_tensor=to_tensor)
    return transforms

def motion_blur(severity=1,always_apply=True,normalize=False,to_tensor=False):
    c = [(10, 3), (15, 5), (15, 7), (15, 13), (20, 15)][severity - 1]
    transforms = transform_wrapper([MotionBlur(blur_limit=c[1], allow_shifted=True, always_apply=always_apply, p=1)],normalize=normalize,to_tensor=to_tensor)
    return transforms

def zoom_blur(severity=1,always_apply=True,normalize=False,to_tensor=False):
    c = [(1.01, 0.005),
         (1.03, 0.005),
         (1.05, 0.01),
         (1.07, 0.01),
         (1.09, 0.02)][severity - 1]
    transforms = transform_wrapper([A.augmentations.ZoomBlur(max_factor=(c[0],c[0]), step_factor=(c[1],c[1]), always_apply=always_apply, p=1)],normalize=normalize,to_tensor=to_tensor)
    return transforms

def contrast(severity=1,variant=1,always_apply=True,normalize=False,to_tensor=False):
    c = [.1, .3, .5, .6, .7][severity - 1]
    transforms = transform_wrapper([A.augmentations.transforms.RandomContrast(limit=(c,c) if variant==1 else (-c,-c), always_apply=always_apply, p=1)],normalize=normalize,to_tensor=to_tensor)
    return transforms

def brightness(severity=1,variant=1,always_apply=True,normalize=False,to_tensor=False):
    c = [.1, .2, .3, .4, .5][severity - 1]
    transforms = transform_wrapper([A.augmentations.transforms.RandomBrightness(limit=(c,c) if variant==1 else (-c,-c), always_apply=always_apply, p=1)],normalize=normalize,to_tensor=to_tensor)
    return transforms

def saturate(severity=1,variant=1,always_apply=True,normalize=False,to_tensor=False):
    c = [10, 20, 40, 60, 80][severity - 1]
    transforms = transform_wrapper([A.augmentations.transforms.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(c,c) if variant==1 else (-c,-c), val_shift_limit=0, always_apply=always_apply, p=1)],normalize=normalize,to_tensor=to_tensor)
    return transforms

def jpeg_compression(severity=1,always_apply=True,normalize=False,to_tensor=False):
    # c = [25, 18, 15, 10, 7][severity - 1]
    c = [80, 60, 40, 20, 10][severity - 1]
    transforms = transform_wrapper([A.augmentations.transforms.JpegCompression(quality_lower=c, quality_upper=c, always_apply=always_apply, p=1)],normalize=normalize,to_tensor=to_tensor)
    return transforms

def shadow(severity=1,always_apply=True,normalize=False,to_tensor=False):
    c = [
            [(0, 0, 1, 1),1,1,5],
            [(0, 0, 1, 1),2,2,5],
            [(0, 0, 1, 1),3,3,8],
            [(0, 0, 1, 1),4,4,8],
            [(0, 0, 1, 1),5,5,8]
        ][severity - 1]
    transforms = transform_wrapper([Shadow(shadow_roi=c[0], num_shadows_lower=c[1], num_shadows_upper=c[2], shadow_dimension=c[3], always_apply=always_apply, p=1)],normalize=normalize,to_tensor=to_tensor)
    return transforms


def get_train_transforms(severity,normalize=False,to_tensor=False):
    p=0.2
    transforms = []
    # blood
    blood_c = [(0.65, 0.3, 4, 0.68, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 1.0, 0),
         (0.65, 0.3, 2, 0.68, 1.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    color = [102, 0, 0]
    aug_mode = 'mud'
    transforms.append(A.augmentations.transforms.Spatter(mean=blood_c[0], std=blood_c[1], gauss_sigma=blood_c[2], cutout_threshold=blood_c[3], intensity=blood_c[4], mode=aug_mode, color=color, always_apply=False, p=p))
    # glare
    glare_c = [(0.61, 0.3, 1, 0.68, 0.6, 0),
         (0.62, 0.3, 1, 0.68, 1.0, 0),
         (0.62, 0.3, 1, 0.68, 1.5, 0),
         (0.62, 0.4, 1, 0.65, 1.5, 1),
         (0.63, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    color = [255,255,255]
    aug_mode = 'mud'
    transforms.append(Glare(mean=glare_c[0], std=glare_c[1], gauss_sigma=glare_c[2], cutout_threshold=glare_c[3], intensity=glare_c[4], mode=aug_mode, color=color, always_apply=False, p=p))
    # shadow
    shadow_c = [
            [(0, 0, 1, 1),1,1,5],
            [(0, 0, 1, 1),2,2,5],
            [(0, 0, 1, 1),3,3,8],
            [(0, 0, 1, 1),4,4,8],
            [(0, 0, 1, 1),5,5,8]
        ][severity - 1]
    transforms.append(Shadow(shadow_roi=shadow_c[0], num_shadows_lower=shadow_c[1], num_shadows_upper=shadow_c[2], shadow_dimension=shadow_c[3], always_apply=False, p=p))
    # defocus_blur
    defocus_blur_c = [(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.3), (5, 0.4), (6, 0.5)][severity - 1]
    transforms.append(A.augmentations.Defocus(radius=defocus_blur_c[0], alias_blur=defocus_blur_c[1], always_apply=False, p=p))
    # motion_blur
    motion_blur_c = [(10, 3), (15, 5), (15, 7), (15, 13), (20, 15)][severity - 1]
    transforms.append(A.augmentations.MotionBlur(blur_limit=motion_blur_c[1], allow_shifted=True, always_apply=False, p=p))
    # zoom_blur
    zoom_blur_c = [(1.01, 0.005),
         (1.03, 0.005),
         (1.05, 0.01),
         (1.07, 0.01),
         (1.09, 0.02)][severity - 1]
    transforms.append(A.augmentations.ZoomBlur(max_factor=zoom_blur_c[0], step_factor=zoom_blur_c[1], always_apply=False, p=p))
    # contrast
    contrast_c = [.1, .3, .5, .6, .7][severity - 1]
    transforms.append(A.augmentations.transforms.RandomContrast(limit=contrast_c, always_apply=False, p=p))
    # brightness
    brightness_c = [.1, .2, .3, .4, .5][severity - 1]
    transforms.append(A.augmentations.transforms.RandomBrightness(limit=brightness_c, always_apply=False, p=p))
    # saturate
    saturate_c = [10, 20, 40, 60, 80][severity - 1]
    transforms.append(A.augmentations.transforms.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=saturate_c, val_shift_limit=0, always_apply=False, p=p))
    # jpeg_compression
    jpeg_compression_c = [80, 60, 40, 20, 10][severity - 1]
    transforms.append(A.augmentations.transforms.JpegCompression(quality_lower=jpeg_compression_c, always_apply=False, p=p))

    # # colorjitter
    # transforms.append(A.transforms.ColorJitter (brightness=0.2,contrast=0.2,saturation=0.05,hue=0.05, always_apply=False, p=p))
    transforms = transform_wrapper(transforms=transforms,normalize=normalize,to_tensor=to_tensor)
    return transforms

# TODO: https://github.com/mahmoudnafifi/WB_color_augmenter

def get_transform(transform_type,severity,normalize=True,to_tensor=True):
    transform_types = ['blood','glare','shadow','defocus_blur','motion_blur','zoom_blur','contrast+','brightness+','saturate+','contrast-','brightness-','saturate-','jpeg_compression','none']
    assert transform_type in transform_types, f'{transform_type} should be in {transform_types}'
    assert severity > 0 and severity < 6, f'severity {severity} should be 1<=severity<=5'
    if transform_type == 'blood': transforms = blood(mode='blood',severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'glare': transforms = glare(mode='glare',severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'shadow': transforms = shadow(severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'defocus_blur': transforms = defocus_blur(severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'motion_blur': transforms = motion_blur(severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'zoom_blur': transforms = zoom_blur(severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'contrast+': transforms = contrast(severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'brightness+': transforms = brightness(severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'saturate+': transforms = saturate(severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'contrast-': transforms = contrast(severity=severity,variant=2,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'brightness-': transforms = brightness(severity=severity,variant=2,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'saturate-': transforms = saturate(severity=severity,variant=2,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'jpeg_compression': transforms = jpeg_compression(severity=severity,always_apply=True,normalize=normalize,to_tensor=to_tensor)
    elif transform_type == 'none': transforms = transform_wrapper(transforms=None,input_res=(384,512),normalize=normalize,to_tensor=to_tensor)

    return transforms


if __name__=='__main__':
    import os
    base_dir = '/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta_torch/dumps/ADWVIF'
    # img_file = '/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta/data/NU_nlp_data/fetal_side/AABWIC_2.jpg'
    img_file = '/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta/data/NU_nlp_data/fetal_side/ADWVIF_2.jpg'
    img = Image.open(img_file)
    img  = np.array(img)
    severity = 2
    transform = get_train_transforms(severity=severity)
    transform_type = 'all'
    trans_img = transform(image=img)
    Image.fromarray(trans_img['image']).save(os.path.join(base_dir,f"{transform_type}_{severity}.png"))

    # for transform_type in ['blood','glare','shadow','defocus_blur','motion_blur','zoom_blur','contrast+','brightness+','saturate+','contrast-','brightness-','saturate-','jpeg_compression','none']:
    #     for severity in range(1,6):
    #         transform = get_transform(transform_type,severity=severity,normalize=False,to_tensor=False)
    #         trans_img = transform(image=img)
    #         Image.fromarray(trans_img['image']).save(os.path.join(base_dir,f"{transform_type}_{severity}.png"))
    
    # add_blood = spatter(mode='blood',severity=4)
    # blood_img = add_blood(image=np.array(img))
    # Image.fromarray(blood_img['image']).save('/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta_torch/dumps/blood_img.png')
    # add_glare = spatter(mode='glare',severity=2)
    # glare_img = add_glare(image=np.array(img))
    # Image.fromarray(glare_img['image']).save('/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta_torch/dumps/glare_img.png')
    # add_saturate = saturate(severity=5)
    # saturate_img = add_saturate(image=np.array(img))
    # Image.fromarray(saturate_img['image']).save('/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta_torch/dumps/saturate_img.png')