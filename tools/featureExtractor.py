import argparse
import torch
#from torchsummary import summary
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.miscellaneous import mkdir
from fcos_core.modeling.detector import build_detection_model
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn
from fcos_core.engine.forward import foward_detector,foward_detector_roifeature,foward_detector_roifeature_one,foward_detector_roifeature_fb
from fcos_core.data import datasets as D
from fcos_core.data.transforms import build_transforms
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.image_list import to_image_list,ImageList
import logging
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.tensorboard import SummaryWriter

import time
import re
def annotation_onefile(xmlpath):
    """
    加载一张图片的GT
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    return (xmin,ymin,xmax,ymax)
    """
    filename = xmlpath#os.path.join(self._data_path, 'Annotations', index + '.xml')
    # print(filename)
    tree = ET.parse(filename)
    objs = tree.findall('object')

    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.int16)
    gt_classes = []

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)-1
        y1 = float(bbox.find('ymin').text)-1
        x2 = float(bbox.find('xmax').text)-1
        y2 = float(bbox.find('ymax').text)-1
        cls = obj.find('name').text.strip()
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes.append(cls)
    return boxes,gt_classes
def dataBlob(imgfile,xmlfile,maskfile=None):
    # imgfile='E:/SeaShips_SMD/JPEGImages/000001.jpg'
    # xmlfile='E:/SeaShips_SMD/Annotations/000001.xml'
    # maskfile='E:/SeaShips_SMD/Segmentations1/SegmentationObjectPNG/000001.png'
    transforms = build_transforms(cfg, False)
    # dataset=D.COCODataset(ann_file, root, remove_images_without_annotations)
    #img = Image.open(imgfile).convert('RGB')
    img=cv2.imread(imgfile,cv2.IMREAD_COLOR)

    H,W,C=img.shape
    boxes, gt_classes = annotation_onefile(xmlfile)
    # cv2.imshow('src',img)
    # cv2.imshow('mo',imgmask)
    # cv2.waitKey()
    imgmasklist=None
    if maskfile:
        mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
        imgmask=cv2.bitwise_and(img,img,mask=mask)
        imgmask=Image.fromarray(cv2.cvtColor(imgmask, cv2.COLOR_BGR2RGB))

    img=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    target = BoxList(boxes, (W,H), mode="xyxy")
    target = target.clip_to_image(remove_empty=True)

    targetmask = target.copy_with_fields(target.fields(), True)
    img, target = transforms(img, target)
    if maskfile:
        imgmask, targetmask = transforms(imgmask, targetmask)
        imglist=to_image_list([img,imgmask],cfg.DATALOADER.SIZE_DIVISIBILITY)
        targets=(target,targetmask)
    else:
        imglist=to_image_list(img,cfg.DATALOADER.SIZE_DIVISIBILITY)
        targets=(target)

    return imglist,targets

def dataBlobAug(imgfiles,xmlfile):
    # imgfile='E:/SeaShips_SMD/JPEGImages/000001.jpg'
    # xmlfile='E:/SeaShips_SMD/Annotations/000001.xml'
    # maskfile='E:/SeaShips_SMD/Segmentations1/SegmentationObjectPNG/000001.png'
    transforms = build_transforms(cfg, False)
    boxes, gt_classes = annotation_onefile(xmlfile)

    imglist=[]
    targets=[]
    for imgfile in imgfiles:
        img=cv2.imread(imgfile,cv2.IMREAD_COLOR)
        H,W,C=img.shape
        # cv2.imshow('src',img)
        # cv2.imshow('mo',imgmask)
        # cv2.waitKey()
        #imgmasklist=None
        # if maskfile:
        #     mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
        #     imgmask=cv2.bitwise_and(img,img,mask=mask)
        #     imgmask=Image.fromarray(cv2.cvtColor(imgmask, cv2.COLOR_BGR2RGB))

        img=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (W,H), mode="xyxy")
        target = target.clip_to_image(remove_empty=True)
        img, target = transforms(img, target)
        # if maskfile:
        #     imgmask, targetmask = transforms(imgmask, targetmask)
        #     imglist=to_image_list([img,imgmask],cfg.DATALOADER.SIZE_DIVISIBILITY)
        #     targets=(target,targetmask)
        # else:
        imglist.append(img)
        targets.append(target)

    imglist=to_image_list(imglist,cfg.DATALOADER.SIZE_DIVISIBILITY)

    return imglist,targets

def extractor(model,
            data_loader,
            dataset_name,
            device="cuda",
            output_folder=None,flag_fg=False):
    '''
    flag_fg:False, only extract featuresof fgs;True,extract features of both fgs and bgs

    '''
    # convert to a torch.device for efficiency
    device = torch.device(device)
    cpu_device = torch.device("cpu")
    logger = logging.getLogger("tools.featureExtractor.extractor")
    dataset = data_loader.dataset
    logger.info("Start extract the target features on {} dataset({} images).".format(dataset_name, len(dataset)))
    #coco = dataset.coco

    # img_id = dataset.ids[0]
    # ann_ids = coco.getAnnIds(imgIds=img_id)
    # target = coco.loadAnns(ann_ids)
    # path = coco.loadImgs(img_id)[0]['file_name']
    # img = Image.open(os.path.join(dataset.root, path)).convert('RGB')
    # boxes = [obj["bbox"] for obj in target]
    # boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    # target= BoxList(boxes, img.size, mode="xywh").convert("xyxy")
    # img, target = dataset._transforms(img, target)
    #start_time = time.time()
    for k in model:
        model[k].eval()
    boxfeature_dict = {}
    if flag_fg:
        FeaMapFolder=os.path.join(output_folder,'feaMapfg')
    else:
        FeaMapFolder = os.path.join(output_folder, 'feaMap')
    if not os.path.exists(FeaMapFolder):
        os.mkdir(FeaMapFolder)

    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch#images:(1,3,608,1088), targets:(600,1066)
        images = images.to(device)
        targets=[target.to(device) for target in targets]
        file_names=[]
        for image_id in image_ids:
            file_names.append(dataset.coco.loadImgs(dataset.ids[image_id])[0]['file_name'])
        with torch.no_grad():  # no compute the gradient
            if flag_fg:
                roifeatures=foward_detector_roifeature_fb(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
                try:
                    #outputFeatures=[target.get_field('featureROI').to(cpu_device) for target in targets]
                    boxfeature_dict.update({img_id: result for img_id, result in zip(image_ids, roifeatures)})
                except KeyError as e:
                    print('Error ',image_ids,)

        # torch.save(boxfeature_dict, os.path.join(output_folder, 'targetFeaturesFG.pth'))
            else:
                roifeatures = foward_detector_roifeature(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
            #roifeatures = model.featureTarget(images,targets)
            #outputFeatures=[(p,r) for p,r in zip(proposals,rois)]
                try:
                    outputFeatures=[target.get_field('featureROI').to(cpu_device) for target in targets]
                    boxfeature_dict.update({img_id: result for img_id, result in zip(image_ids, outputFeatures)})
                except KeyError as e:
                    print('Error ',image_ids,)
    if flag_fg:
        torch.save(boxfeature_dict,os.path.join(output_folder,'targetFeaturesFG.pth'))
    else:
        torch.save(boxfeature_dict, os.path.join(output_folder, 'targetFeatures.pth'))
    return boxfeature_dict

def extractoroneMask(model,
            device="cuda",
            output_folder=None):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    # cpu_device = torch.device("cpu")
    logger = logging.getLogger("tools.featureExtractor.extractor")
    #dataset = data_loader.dataset
    #logger.info("Start extract the target features on {} dataset({} images).".format(dataset_name, len(dataset)))


    for k in model:
        model[k].eval()
    boxfeature_dict = {}
    FeaMapFolder=os.path.join(output_folder,'feaMap')
    if not os.path.exists(FeaMapFolder):
        os.mkdir(FeaMapFolder)

    imgfile = 'E:/SeaShips_SMD/JPEGImages/000001.jpg'  #000001 MVI_1624_VIS_00489
    imgfiles = ['E:/SeaShips_SMD/JPEGImages/000001.jpg','E:/SeaShips_SMD/JPEGImagesAug/F000001.jpg','E:/SeaShips_SMD/JPEGImagesAug/R000001.jpg']
    imgfilea = 'E:/SeaShips_SMD/JPEGImagesAug/000001.jpg'
    xmlfile = 'E:/SeaShips_SMD/Annotations/000001.xml'
    maskfile = 'E:/SeaShips_SMD/Segmentations1/SegmentationObjectPNG/000001.png'
    batch=dataBlobAug(imgfiles,xmlfile)

    #batch = dataBlob([imgfile,imgfilea], xmlfile, maskfile=maskfile)
    filename=os.path.splitext(os.path.basename(imgfiles[0]))
#for i, batch in enumerate(tqdm(data_loader)):
    images, targets = batch#images:(1,3,608,1088), targets:(600,1066)
    images = images.to(device)
    targets=[target.to(device) for target in targets]

    file_names=[]
    file_names.append('{}{}'.format(filename[0], filename[1]))
    file_names.append('F{}{}'.format(filename[0], filename[1]))
    file_names.append('R{}{}'.format(filename[0], filename[1]))
    # file_names.append('{}mask{}'.format(filename[0],filename[1]))
    with torch.no_grad():  # no compute the gradient
        #roifeatures = foward_detector_roifeature(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
        roifeatures = foward_detector_roifeature_one(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
        #roifeatures = model.featureTarget(images,targets)
        #outputFeatures=[(p,r) for p,r in zip(proposals,rois)]

        # try:
        #     outputFeatures=[roifeature.get_field('featureROI').to(cpu_device) for roifeature in roifeatures]
        #     boxfeature_dict.update({img_id: result for img_id, result in zip(image_ids, outputFeatures)})
        # except KeyError as e:
        #     print('Error ',image_ids,)

    #torch.save(boxfeature_dict,os.path.join(output_folder,'targetFeatures.pth'))
    return boxfeature_dict

def extractorone(model,
            device="cuda",
            output_folder=None):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    # cpu_device = torch.device("cpu")
    logger = logging.getLogger("tools.featureExtractor.extractor")

    for k in model:
        model[k].eval()
    boxfeature_dict = {}
    FeaMapFolder=os.path.join(output_folder,'feaMap')
    if not os.path.exists(FeaMapFolder):
        os.mkdir(FeaMapFolder)

    imgfile = 'E:/SeaShips_SMD/JPEGImages/000001.jpg'  #000001 MVI_1624_VIS_00489
    imgfiles = ['E:/SeaShips_SMD/JPEGImages/000001.jpg','E:/SeaShips_SMD/JPEGImagesAug/F000001.jpg','E:/SeaShips_SMD/JPEGImagesAug/R000001.jpg']
    imgfilea = 'E:/SeaShips_SMD/JPEGImagesAug/000001.jpg'
    xmlfile = 'E:/SeaShips_SMD/Annotations/000001.xml'
    #maskfile = 'E:/SeaShips_SMD/Segmentations1/SegmentationObjectPNG/000001.png'
    batch=dataBlobAug(imgfiles,xmlfile)

    #batch = dataBlob([imgfile,imgfilea], xmlfile, maskfile=maskfile)
    filename=os.path.splitext(os.path.basename(imgfiles[0]))
#for i, batch in enumerate(tqdm(data_loader)):
    images, targets = batch#images:(1,3,608,1088), targets:(600,1066)
    images = images.to(device)
    targets=[target.to(device) for target in targets]

    file_names=[]
    file_names.append('{}{}'.format(filename[0], filename[1]))
    file_names.append('F{}{}'.format(filename[0], filename[1]))
    file_names.append('R{}{}'.format(filename[0], filename[1]))
    # file_names.append('{}mask{}'.format(filename[0],filename[1]))
    with torch.no_grad():  # no compute the gradient
        #roifeatures = foward_detector_roifeature(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
        roifeatures = foward_detector_roifeature_one(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
        #roifeatures = model.featureTarget(images,targets)
        #outputFeatures=[(p,r) for p,r in zip(proposals,rois)]

        # try:
        #     outputFeatures=[roifeature.get_field('featureROI').to(cpu_device) for roifeature in roifeatures]
        #     boxfeature_dict.update({img_id: result for img_id, result in zip(image_ids, outputFeatures)})
        # except KeyError as e:
        #     print('Error ',image_ids,)

    # torch.save(boxfeature_dict,os.path.join(output_folder,'targetFeatures.pth'))
    return boxfeature_dict

def featureExtractor(cfg, model, comment='',useOneFile=False):
    regxint = re.findall(r'\d+', cfg.MODEL.WEIGHT)
    numstr = ''
    if len(regxint) > 0:
        numstr = str(int(regxint[-1]))

    torch.cuda.empty_cache()  #
    # iou_types = ("bbox",)
    # if cfg.MODEL.MASK_ON:
    #     iou_types = iou_types + ("segm",)
    # if cfg.MODEL.KEYPOINT_ON:
    #     iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "extract"+comment+numstr, dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, is_train=False,shuffle=False)
    results=[]
    if useOneFile==0:
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            result=extractor(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                device=cfg.MODEL.DEVICE,
                output_folder=output_folder,
                flag_fg=True
            )
            #results.append(result)
    elif useOneFile==1: #使用一个样本
        output_folder= os.path.join(cfg.OUTPUT_DIR, "extract"+comment, "customData")
        mkdir(output_folder)
        result=extractorone(model,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder)
    elif useOneFile==2:#使用mask
        output_folder = os.path.join(cfg.OUTPUT_DIR, "extract" + comment, "customData")
        mkdir(output_folder)
        result = extractoroneMask(model,
                              device=cfg.MODEL.DEVICE,
                              output_folder=output_folder)


    return results


def write_graph(model,output_dir):
    for k in model:
        model[k].eval()
    #output_folder=''
    sumwriter = SummaryWriter(output_dir)
    model_input = torch.rand((1, 3, 608, 1088), device="cuda")
    sumwriter.add_graph(model['backbone'], model_input, False)  ##(['backbone', 'fcos'])
    features = model['backbone'](model_input)
    # model_input = to_image_list(model_input)
    flagF=torch.tensor(0).bool()
    flagT=torch.tensor(1).bool()
    sumwriter.add_graph(model['fcos'], [model_input, features,flagF,flagF,flagT], False)  ##erro

    #RuntimeError: Tracer cannot infer type of ([BoxList(num_boxes=0, image_width=1088, image_height=608, mode=xyxy)], {}, None) :Could not infer type of list element: Only tensors and (possibly nested) tuples of tensors, lists, or dictsare supported as inputs or outputs of traced functions, but instead got value of type BoxList.



if __name__=="__main__":
    #--config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC DATASETS.TEST ('ship_test_SMD_cocostyle',) MODEL.WEIGHT model_0020000.pth
    #--config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSshipC DATASETS.TEST ('ship_test_SeaShips_cocostyle',) MODEL.WEIGHT model_0020000.pth
    #--config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) MODEL.WEIGHT model_0020000.pth

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="../configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD00.yaml",
        # "../configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD03.yaml",#
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--comment',default='feature',type=str,help='comment of the name of the folder')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=['OUTPUT_DIR', 'logFRCNN0'
                 ],
        nargs=argparse.REMAINDER, type=str,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    args.opts+=['TEST.IMS_PER_BATCH', '1']#batchsize=1
    cfg.merge_from_list(args.opts)
    #cfg.freeze()

    model = {}
    model["backbone"] = build_backbone(cfg).to(cfg.MODEL.DEVICE)#model+fpn
    model["fcos"] = build_rpn(cfg, model["backbone"].out_channels).to(cfg.MODEL.DEVICE)#feature map -> cls+reg
    # with open('FCOS.txt','w') as f:
    #     f.write('Backbone:\n')
    #     f.write(model["backbone"].__str__()+'\n')
    #     f.write('FCOS:\n')
    #     f.write(model["fcos"].__str__()+'\n')

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model,)

    _ = checkpointer.load(f=os.path.join(output_dir,cfg.MODEL.WEIGHT), load_dis=False, load_opt_sch=False)


    #test inference time
    imgfiles = ['E:/SeaShips_SMD/JPEGImages/000001.jpg', 'E:/SeaShips_SMD/JPEGImagesAug/F000001.jpg',
                'E:/SeaShips_SMD/JPEGImagesAug/R000001.jpg']
    imgfilea = 'E:/SeaShips_SMD/JPEGImagesAug/000001.jpg'
    xmlfile = 'E:/SeaShips_SMD/Annotations/000001.xml'
    #cfg.INPUT.MIN_SIZE_TRAIN
    # cfg.INPUT.MIN_SIZE_TEST=608
    # cfg.INPUT.MAX_SIZE_TEST=608

    batch = dataBlobAug(imgfiles, xmlfile)
    images, targets = batch  # images:(1,3,608,1088), targets:(600,1066)
    images = images.to("cuda")
    targets = [target.to("cuda") for target in targets]
    # model_input = torch.rand((1, 3, 608, 1088), device="cuda")
    timeList=[]
    for i in range(20):
        begin=time.time()
        features = model["backbone"](images.tensors)
        out = model["fcos"](images, features)
        end=time.time()
        #del features,out
        timeList.append(images.tensors.shape[0]/(end-begin))
        print(end-begin,images.tensors.shape)
    print(timeList)

    if False:
        featureExtractor(cfg, model, comment=args.comment,useOneFile=0)
        '''
        useOneFile:
        0:使用测试集
        1:#使用一个样本
        2：#使用mask
        '''
        # useOneFile:0,1,2
    # write_graph(model, output_dir)
    # for k in model:
    #     model[k].eval()
    # torch.save(model['backbone'],'backbone.pt')
    # torch.save(model['fcos'], 'fcos.pt')


