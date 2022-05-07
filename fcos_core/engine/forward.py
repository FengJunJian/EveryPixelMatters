from fcos_core.structures.image_list import to_image_list
from fcos_core.modeling.poolers import Pooler
from fcos_core.config import cfg
import cv2
import numpy as np
import os
def foward_detector(model, images, targets=None, return_maps=False):
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()

    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    images = to_image_list(images)
    features = model_backbone(images.tensors)

    f = {
        layer: features[map_layer_to_index[layer]]
        for layer in feature_layers
    }
    losses = {}

    if model_fcos.training and targets is None:
        # train G on target domain
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=None, return_maps=return_maps)
        assert len(proposal_losses) == 1 and proposal_losses["zero"] == 0 ,"proposal losses:{}".format(proposal_losses["zero"])# loss_dict should be empty dict
    else:
        # train G on source domain / inference
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=targets, return_maps=return_maps)

    if model_fcos.training:
        # training
        m = {
            layer: {
                map_type:
                score_maps[map_type][map_layer_to_index[layer]]
                for map_type in score_maps
            }
            for layer in feature_layers
        }
        losses.update(proposal_losses)
        return losses, f, m
    else:
        # inference
        result = proposals
        return result


def foward_detector_roifeature(model, images, targets=None, return_maps=False,saveFolder='',imgnnames=None):
    oh,ow=images.tensors.shape[2:]
    rh,rw=images.image_sizes[0]
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()

    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    # images = to_image_list(images)
    features = model_backbone(images.tensors)
    ######################################################
    # pooled 特征
    # scales=[]
    # for fea in features:
    #     ch,cw=fea.shape[-2:]
    #     scales.append(cw/ow)
    # pooler = Pooler(
    #     output_size=(7, 7),
    #     scales=scales,
    #     sampling_ratio=0,
    # )
    # pooledfeature=pooler(features,targets)
    #########################################################
    # f = {
    #     layer: features[map_layer_to_index[layer]]
    #     for layer in feature_layers
    # }
    #losses = {}

    proposals, score_maps = model_fcos.featureROI(
        images, features, targets=targets, return_maps=return_maps)

    #imgind=4
    for imgind in range(len(images.image_sizes)):
        imgname=os.path.splitext(imgnnames[imgind])[0]

        img=images.tensors[imgind].cpu().detach().numpy()
        img=np.transpose(img,(1,2,0))
        mean = cfg.INPUT.PIXEL_MEAN
        std = cfg.INPUT.PIXEL_STD
        img=(img*std+mean).astype(np.uint8)
        #size=(0,0)
        totalImg=[]

        for i in range(len(score_maps['centerness'])):
            cenfeature=score_maps['centerness'][i][imgind,0]#
            cenfeature=cenfeature.cpu().detach().numpy()
            fea=cv2.normalize(cenfeature,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            #fea= ((cenfeature-cenfeature.min())).astype(np.uint8)
            #if i==0:
            #fea = cv2.resize(fea,(0,0),fx=3.0,fy=3.0, interpolation=cv2.INTER_NEAREST)  # 改变特征呢图尺寸
            fea = cv2.resize(fea, (rw, rh), fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)  # 改变特征呢图尺寸
            totalImg.append(fea)

            fea = cv2.applyColorMap(fea, cv2.COLORMAP_JET)  # 变成伪彩图
            #cv2.imshow('b'+str(i),fea)
            cv2.imwrite(os.path.join(saveFolder,"%s_F%02d.jpg"%(imgname,i)),fea)
        MImg = np.mean(totalImg, axis=0).astype(np.uint8)
        MImg = cv2.applyColorMap(MImg, cv2.COLORMAP_JET)  # 变成伪彩图
        # TImg = np.sum(totalImg, axis=0).astype(np.uint8)
        Weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        TImg = np.sum(np.multiply(totalImg, Weights.reshape(len(totalImg), 1,  1)), axis=0).astype(np.uint8)
        TImg = cv2.applyColorMap(TImg, cv2.COLORMAP_JET)  # 变成伪彩图
        cv2.imwrite(os.path.join(saveFolder, "%s_T%02d.jpg" % (imgname, 5)),TImg)
        Weights = np.array([0.8, 0.2, 0.0, 0.0, 0.0])
        TImg = np.sum(np.multiply(totalImg, Weights.reshape(len(totalImg), 1, 1)), axis=0).astype(np.uint8)
        TImg = cv2.applyColorMap(TImg, cv2.COLORMAP_JET)  # 变成伪彩图
        cv2.imwrite(os.path.join(saveFolder, "%s_T%02d.jpg" % (imgname, 2)), TImg)
        # cv2.imshow('T',TImg)
        # cv2.imshow('M', MImg)
        cv2.imwrite(os.path.join(saveFolder,"%s.jpg" % (imgname)),img)
        cv2.imwrite(os.path.join(saveFolder, "%s_M.jpg" % (imgname)),MImg)
        #cv2.imshow('src', img)
        cv2.waitKey(1)

    # if model_fcos.training:
    #     # training
    #     m = {
    #         layer: {
    #             map_type:
    #             score_maps[map_type][map_layer_to_index[layer]]
    #             for map_type in score_maps
    #         }
    #         for layer in feature_layers
    #     }
    #     losses.update(proposal_losses)
    #     return losses, f, m
    # else:
        # inference
    result = proposals
    return result