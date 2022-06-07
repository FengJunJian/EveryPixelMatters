from fcos_core.structures.image_list import to_image_list
from fcos_core.modeling.poolers import Pooler
from fcos_core.config import cfg
from fcos_core.utils.miscellaneous import mkdir
from torchvision.ops import MultiScaleRoIAlign,box_iou
from fcos_core.structures.boxlist_ops import boxlist_nms

import torch
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

    if model_fcos.training and targets is None:#target domain
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

def foward_detector_roifeature_fb(model, images, targets=None, return_maps=False,saveFolder='',imgnnames=None):
    # assert len(imgnnames)==1
    # imgname=imgnnames[0]
    oh,ow=images.tensors.shape[2:]
    #rh,rw=images.image_sizes[0]
    image_sizes = images.image_sizes
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()

    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    # images = to_image_list(images)
    features = model_backbone(images.tensors)
    boxes, score_maps = model_fcos.featureROI(images, features, targets=targets, return_maps=return_maps)
    bc = score_maps['box_cls']
    cen = score_maps['centerness']

    f = {
        layer: features[map_layer_to_index[layer]]
        for layer in feature_layers
    }
    fea_list=[]
    for i,imgname in enumerate(imgnnames):

        # bcn = bc[0][i, 0].cpu().detach().numpy()
        # bcnN = cv2.normalize(bcn, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # bcnNM = cv2.applyColorMap(bcnN, cv2.COLORMAP_JET)  # 变成伪彩图
        # #cv2.imshow('bcnNM', bcnNM)  # 变成伪彩图
        # cv2.imwrite(os.path.join(saveFolder, 'b' + imgname), bcnNM)  # 变成伪彩图
        #
        # cenn = cen[0][i, 0].cpu().detach().numpy()
        # cennN = cv2.normalize(cenn, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # cennNM = cv2.applyColorMap(cennN, cv2.COLORMAP_JET)  # 变成伪彩图
        # #cv2.imshow('cennNM', cennNM)  # 变成伪彩图
        # cv2.imwrite(os.path.join(saveFolder, 'c' + imgname), cennNM)  # 变成伪彩图
        # # bce = bc[0] * cen[0]
        # # bcen = bce[0, 0].cpu().detach().numpy()
        # bcen = bcnN.astype(np.float32) * cennN.astype(np.float32)
        # bcenN = cv2.normalize(bcen, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # bcenNM = cv2.applyColorMap(bcenN, cv2.COLORMAP_JET)  # 变成伪彩图
        # #cv2.imshow('bcenNM2', bcenNM)  # 变成伪彩图
        # cv2.imwrite(os.path.join(saveFolder,'bc'+imgname), bcenNM)  # 变成伪彩图
        #
        # fn=torch.mean(features[0],dim=1)[i].cpu().detach().numpy()#cen[0]
        # fnN = cv2.normalize(fn, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # fnNM = cv2.applyColorMap(fnN, cv2.COLORMAP_JET)  # 变成伪彩图
        # #cv2.imshow('fnNM', fnNM)  # 变成伪彩图
        # cv2.imwrite(os.path.join(saveFolder, 'f' + imgname), fnNM)  # 变成伪彩图
        #
        # fcn = fnN.astype(np.float32) * cennN.astype(np.float32)
        # fcnN = cv2.normalize(fcn, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # fcnNM = cv2.applyColorMap(fcnN, cv2.COLORMAP_JET)  # 变成伪彩
        # #cv2.imshow('fcnNM', fcnNM)  # 变成伪彩图
        # cv2.imwrite(os.path.join(saveFolder, 'fc' + imgname), fcnNM)  # 变成伪彩图
        #
        # cv2.waitKey(1)

        ######################################################
        output_size = [8, 16]
        RoIpooler = MultiScaleRoIAlign(featmap_names=feature_layers, output_size=output_size, sampling_ratio=0)
        bboxes = [b.bbox for b in targets]
        # fg_num_boxes=0
        fg_num_boxes =sum([len(b) for b in bboxes])
        bg_num_boxes=0
        if fg_num_boxes>0:
            iou=box_iou(bboxes[i],boxes[i].bbox)
            _,boxeskeeps=torch.where(iou<=0.01)#无iou交集为bg
            inds=torch.randperm(len(boxeskeeps))
            boxes_bg=[]
            bgboxes=boxes[i][boxeskeeps[inds[:fg_num_boxes * 2]]]
            boxes_bg.append(bgboxes)
            # for b in boxes:
            #     bg_inds=torch.where(b.get_field('scores') < 0.1)[0]
            #     boxes_bg.append(b[bg_inds])
            bg_num_boxes+=len(bgboxes)
            bboxes+=[b.bbox for b in boxes_bg]
        pooledfeatures = RoIpooler(f, bboxes, image_sizes)

        if len(pooledfeatures)==0:
            #return fea_list
            fea_list.append(None)
        else:
            pi = pooledfeatures.cpu().detach().numpy()  # for p0
            # if len(pooledfeatures)>2:
            #     print(pooledfeatures.shape)
            #print('fg:',fg_num_boxes,'bg:',bg_num_boxes,'fea shape:',pi.shape)

            fea_dict={}
            fea_dict['fg']=pi[:fg_num_boxes].mean(1)
            fea_dict['bg']=pi[fg_num_boxes:].mean(1)
            fea_list.append(fea_dict)

    return fea_list

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
    scales=[]
    #output_size = []

    for fea in features:
        ch,cw=fea.shape[-2:]
        scales.append(cw/ow)
    pooler = Pooler(
        output_size=(7, 7),
        scales=scales,
        sampling_ratio=0,
    )
    pooledfeatures, rois = pooler.extract(features, targets)
    ###pooled feature
    indflag = rois[:, 0]
    for i, btarget in enumerate(targets):
        inds_each_target = torch.nonzero(indflag == i).squeeze(1)
        feas_each_target = []
        for j,pooledf in enumerate(pooledfeatures):
            if j >1:
                break
            feas_each_target.append(pooledf[inds_each_target].flatten(1))  # (batch,-1)

        cfeas_each_target = torch.cat(feas_each_target, dim=1).cpu()  # (batch,fea_num)
        btarget.add_field("featureROI", cfeas_each_target)
    ###

    proposals, score_maps = model_fcos.featureROI(
        images, features, targets=targets, return_maps=return_maps)

    for imgind in range(len(images.image_sizes)):
        imgname=os.path.splitext(imgnnames[imgind])[0]
        img=images.tensors[imgind].cpu().detach().numpy()
        img=np.transpose(img,(1,2,0))
        mean = cfg.INPUT.PIXEL_MEAN
        std = cfg.INPUT.PIXEL_STD
        img=(img*std+mean).astype(np.uint8)
        totalImg=[]

        for i in range(len(score_maps['centerness'])):
            cenfeature=score_maps['centerness'][i][imgind,0]#
            cenfeature=cenfeature.cpu().detach().numpy()
            fea=cv2.normalize(cenfeature,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            #fea= ((cenfeature-cenfeature.min())).astype(np.uint8)
            fea = cv2.resize(fea, (rw, rh),  interpolation=cv2.INTER_NEAREST)  # 改变特征呢图尺寸
            totalImg.append(fea)
            fea = cv2.applyColorMap(fea, cv2.COLORMAP_JET)  # 变成伪彩图
            #cv2.imshow('b'+str(i),fea)
            cv2.imwrite(os.path.join(saveFolder,"%s_F%02d.jpg"%(imgname,i)),fea)
        # MImg = np.mean(totalImg, axis=0).astype(np.uint8)
        # MImg = cv2.applyColorMap(MImg, cv2.COLORMAP_JET)  # 变成伪彩图
        # # TImg = np.sum(totalImg, axis=0).astype(np.uint8)
        # Weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        # TImg = np.sum(np.multiply(totalImg, Weights.reshape(len(totalImg), 1,  1)), axis=0).astype(np.uint8)
        # TImg = cv2.applyColorMap(TImg, cv2.COLORMAP_JET)  # 变成伪彩图
        # cv2.imwrite(os.path.join(saveFolder, "%s_T%02d.jpg" % (imgname, 5)),TImg)
        # Weights = np.array([0.8, 0.2, 0.0, 0.0, 0.0])
        # TImg = np.sum(np.multiply(totalImg, Weights.reshape(len(totalImg), 1, 1)), axis=0).astype(np.uint8)
        # TImg = cv2.applyColorMap(TImg, cv2.COLORMAP_JET)  # 变成伪彩图
        # cv2.imwrite(os.path.join(saveFolder, "%s_T%02d.jpg" % (imgname, 2)), TImg)
        # # cv2.imshow('T',TImg)
        # # cv2.imshow('M', MImg)
        cv2.imwrite(os.path.join(saveFolder,"%s.jpg" % (imgname)),img)
        # cv2.imwrite(os.path.join(saveFolder, "%s_M.jpg" % (imgname)),MImg)
        #cv2.imshow('src', img)
        #cv2.waitKey(1)

    result = proposals
    return result


def foward_detector_roifeature_one(model, images, targets=None, return_maps=False,saveFolder='',imgnnames=None):
    oh,ow=images.tensors.shape[2:]
    rh,rw=images.image_sizes[0]
    image_sizes=images.image_sizes
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()

    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    # images = to_image_list(images)
    features = model_backbone(images.tensors)
    f = {
        layer: features[map_layer_to_index[layer]]
        for layer in feature_layers
    }
    ######################################################
    # pooled 特征
    scales=[]
    output_size = []
    output_size=[128,256]
    # compute fixed output_size
    # for target in targets:
    #     target=target.convert("xyxy")
    #     area=target.area()
    #     ti=area.argmax()
    #     a=area[ti].cpu().detach().numpy().item()
    #     xmin,ymin,xmax,ymax=target.bbox[ti].cpu().detach().numpy()
    #     w,h=xmax-xmin,ymax-ymin
    #     output_size.append((a,h,w))
    # output_size=np.array(output_size).round().astype(np.int64)
    # mi=np.argmax(output_size[:,0])
    # output_size=output_size[mi,1:].tolist()
    # output_size.append(t)
    for fea in features:
        ch,cw=fea.shape[-2:]
        scales.append(cw/ow)
    # pooler = Pooler(
    #     output_size=output_size,
    #     scales=scales,
    #     sampling_ratio=0,
    # )
    RoIpooler=MultiScaleRoIAlign(featmap_names=feature_layers,output_size=output_size,sampling_ratio=0)
    bboxes=[b.bbox for b in targets]
    num_boxes=len(bboxes[0])

    a=RoIpooler(f,bboxes,image_sizes)
    #b= pooler(features, targets)
    #a=a.mean(1)

    #for i in range(len(pooledfeatures)):
    if len(a)==0:
        return None,None
    pi = a.cpu().detach().numpy()  # for p0
    if len(a)>2:
        print(a.shape)
    print(a.shape)
    diffp = np.abs((pi[:num_boxes] - pi[num_boxes:]).mean(1))

    diffpN = cv2.normalize(diffp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    diffpM = cv2.applyColorMap(diffpN[0], cv2.COLORMAP_JET)  # 变成伪彩图
    # cv2.imwrite(os.path.join(savePDiff, "%s_diffpooled%d.jpg" % (imgname,i)),diffp)  # 变成伪彩图
    cv2.imshow('d', diffpM)
    cv2.waitKey(1)
    return diffp,diffpN

    if False:
        pooledfeatures,rois=pooler.extract(features,targets)
        ###pooled feature
        indflag=rois[:,0]
        for i ,btarget in enumerate(targets):
            inds_each_target=torch.nonzero(indflag==i).squeeze(1)
            feas_each_target=[]
            for pooledf in pooledfeatures:
                feas_each_target.append(pooledf[inds_each_target].flatten(1))#(batch,-1)
            cfeas_each_target=torch.cat(feas_each_target,dim=1).cpu()#(batch,fea_num)
            btarget.add_field("featureROI",cfeas_each_target)
        ###

        imgname = os.path.splitext(imgnnames[0])[0]
        dirname = os.path.dirname(saveFolder)
        saveP0 = os.path.join(dirname, 'feaMap0')
        savePm = os.path.join(dirname, 'feaMapm')
        savePDiff = os.path.join(dirname, 'feaMapDiff')
        mkdir(saveP0)
        mkdir(savePm)
        mkdir(savePDiff)

        #visual pooledfeature
        p0 = pooledfeatures[0].cpu().detach().numpy()  # for p0
        for i in range(len(pooledfeatures)):
            pi=pooledfeatures[i].cpu().detach().numpy()#for p0
            diffp=np.abs((pi[0]-pi[1]).mean(0))
            diffp = cv2.normalize(diffp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            diffp = cv2.applyColorMap(diffp, cv2.COLORMAP_JET)  # 变成伪彩图
            # cv2.imwrite(os.path.join(savePDiff, "%s_diffpooled%d.jpg" % (imgname,i)),diffp)  # 变成伪彩图
            cv2.imshow('d',diffp)
            cv2.waitKey(1)

        for pci in range(p0.shape[1]):
            pc=p0[:,pci,:,:]
            pc = cv2.normalize(pc, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(os.path.join(saveP0,"%s_%d.jpg"%(imgname,pci)),pc[0])
            cv2.imwrite(os.path.join(saveP0, "%s_%df.jpg" % (imgname, pci)),
                        cv2.applyColorMap(pc[0], cv2.COLORMAP_JET))  # 变成伪彩图
            cv2.imwrite(os.path.join(savePm, "%s_%d.jpg" % (imgname, pci)), pc[1])
            cv2.imwrite(os.path.join(savePm, "%s_%df.jpg" % (imgname, pci)),
                        cv2.applyColorMap(pc[1], cv2.COLORMAP_JET))  # 变成伪彩图
            # cv2.imshow('o',pc[0])
            # cv2.imshow('m',pc[1])
            # cv2.moveWindow('o', 1000, 100)
            # cv2.moveWindow('m', 1000, 300)
            # cv2.waitKey(1)
        #########################################################
        proposals, score_maps = model_fcos.featureROI(
            images, features, targets=targets, return_maps=return_maps)

        #imgind=4
        cfeamap,rois=pooler.extract(score_maps['centerness'], targets)
        for i in range(len(cfeamap)):
            fea=cfeamap[i].cpu().detach().numpy()
            fea=cv2.normalize(fea,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            cv2.imshow('a'+str(i),cv2.applyColorMap(fea[0,0], cv2.COLORMAP_JET))
            cv2.imshow('b'+str(i),cv2.applyColorMap(fea[1,0], cv2.COLORMAP_JET))
            diffp = np.abs((fea[0,0] - fea[1,0]))
            diffp = cv2.normalize(diffp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            diffp = cv2.applyColorMap(diffp, cv2.COLORMAP_JET)  # 变成伪彩图
            cv2.imshow('diffp',diffp)#差异图
            cv2.imwrite(os.path.join(savePDiff, "%s_diffcenter%d.jpg" % (imgname,i)),
                        diffp)  # 变成伪彩图

            cv2.moveWindow('a'+str(i), 1000, 100)
            cv2.moveWindow('b'+str(i), 1000, 300)

            feas=score_maps['centerness'][i].cpu().detach().numpy()
            feas = cv2.normalize(feas, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imshow('as' + str(i), cv2.resize(cv2.applyColorMap(feas[0, 0], cv2.COLORMAP_JET),None,fx=3.0,fy=3.0))
            cv2.imshow('bs' + str(i), cv2.resize(cv2.applyColorMap(feas[1, 0], cv2.COLORMAP_JET),None,fx=3.0,fy=3.0))
            cv2.waitKey(1)

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
                fea = cv2.resize(fea, (rw, rh),  interpolation=cv2.INTER_NEAREST)  # 改变特征呢图尺寸
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


        result = proposals
    return result