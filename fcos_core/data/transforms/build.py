# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import mask2polygons,SegmentationMask
from . import transforms as T

import albumentations as A
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import numpy as np
import torch

def build_transforms(cfg, is_train=True):
    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            ))
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        other_prob = 0.4
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        other_prob = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    Talbu = A.Compose([
        #A.Resize(int(H / 2), int(W / 2)),
        A.Blur(p=other_prob),
        A.GaussianBlur(p=other_prob),
        A.OneOf(
            [
        A.RandomFog(fog_coef_upper=0.5),
        A.RandomRain(),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.3), num_flare_circles_lower=1, num_flare_circles_upper=5, src_radius=300,)
        ],p=other_prob),
        A.Cutout(num_holes=16,max_h_size=16,max_w_size=16,p=other_prob),
        A.ShiftScaleRotate(shift_limit=0, rotate_limit=0, scale_limit=0.6, border_mode=cv2.BORDER_CONSTANT,p=other_prob)
    ], bbox_params=A.BboxParams("pascal_voc",), )#mode:xyxy

    Ttorch = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return MulTransform(Talbu,Ttorch)

class MulTransform(object):
    def __init__(self,Talbu=None,Ttorch=None):
        self.Talbu=Talbu
        self.Ttorch=Ttorch
        self.mode="xyxy"
        self.albuformat="pascal_voc"
    def __call__(self, image, target,Talbu_force=True):
        if self.Talbu and Talbu_force:
            #albuformat = "pascal_voc"
            if target.mode!=self.mode:
                target.convert(self.mode)
            extra_fields=target.extra_fields
            fields_temp = extra_fields.copy()

            immasks=None
            if 'masks' in extra_fields:
                immasks=[]
                masks = extra_fields.pop('masks')
                for p in masks:
                    immasks.append(p.convert("mask").numpy())

            original_bbox_p = self.Talbu.processors['bboxes'].params._to_dict()
            original_bbox_p.update({"format":self.albuformat,"label_fields": list(extra_fields.keys())})
            self.Talbu.processors["bboxes"] = A.BboxProcessor(A.BboxParams(**original_bbox_p))

            albuDict={}
            imagearr=None
            if isinstance(image,Image.Image):
                imagearr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            elif isinstance(image,np.ndarray):
                imagearr = image
            elif isinstance(image,torch.Tensor):
                input_tensor=image.to(torch.device('cpu')).numpy()
                in_arr = np.transpose(input_tensor, (1, 2, 0))#(c,h,w)->(h,w,c)
                imagearr = cv2.cvtColor(np.uint8(in_arr), cv2.COLOR_RGB2BGR)

            albuDict["image"] = imagearr
            albuDict["bboxes"]=target.bbox
            if immasks is not None:
                albuDict["masks"] =immasks
            albuDict.update(extra_fields)
            #albuDict['masks'] = immask

            augmented=self.Talbu(**albuDict)
            bboxes=augmented["bboxes"]

            if len(bboxes)==0:# if bboxes is zero then using the original data
                #augmented=albuDict
                target.extra_fields.update(fields_temp)
                #bboxes=np.empty((0,4),dtype=np.float32)
            else:
                image = augmented["image"]
                h, w, c = image.shape
                target = BoxList(augmented["bboxes"], (w, h), mode=target.mode)
                for k in extra_fields.keys():
                    target.add_field(k,torch.tensor(augmented[k]))
                if immasks is not None:
                    amasks=augmented['masks']
                    segs=mask2polygons(amasks)
                    smasks = SegmentationMask(segs, (w,h))
                    target.add_field("masks",smasks)
        if self.Ttorch:
            if isinstance(image,np.ndarray):
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image=Image.fromarray(image)
            image, target = self.Ttorch(image, target)
        #_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
        return image,target

def detection_vis(im, dets,label=None,score=None,thiness=5, color=(0,0,255)):
    '''
    im:cv2.im
    dets:[xmin,ymin,xmax,ymax]
    label:class_ind(optional)
    score:(optional)

    '''
    H,W,C=im.shape
    for i in range(len(dets)):
        rectangle_tmp = im.copy()
        bbox = dets[i, :4].astype(np.int32)
        class_ind = int(label[i])

        # score = dets[i, -1]
        # if GT_color:
        #     color=GT_color
        # else:
        #     color=colors[class_ind]

        string = str(class_ind)#CLASS_NAMES[class_ind]
        if score:
            string+='%.4f'%(score[i])

        # string = '%s' % (CLASSES[class_ind])
        fontFace = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 1.5

        text_size, baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
        text_origin = (bbox[0]-1, bbox[1])  # - text_size[1]
    ###########################################putText
        p1=[text_origin[0] - 1, text_origin[1] + 1]
        p2=[text_origin[0] + text_size[0] + 1,text_origin[1] - text_size[1] - 2]
        if p2[0]>W:
            dw=p2[0]-W
            p2[0]-=dw
            p1[0]-=dw

        rectangle_tmp=cv2.rectangle(rectangle_tmp, (p1[0], p1[1]),
                           (p2[0], p2[1]),
                           color, cv2.FILLED)
        cv2.addWeighted(im, 0.7, rectangle_tmp, 0.3, 0, im)
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thiness)
        im = cv2.putText(im, string, (p1[0]+1,p1[1]-1),
                         fontFace, fontScale, (0, 0, 0), thiness,lineType=-1)
    return im