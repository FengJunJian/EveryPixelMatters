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
from fcos_core.engine.forward import foward_detector,foward_detector_roifeature,foward_detector_roifeature_one
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from fcos_core.modeling.detector import build_detection_model
import time

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
    '''
    two imgs: one for original image and the other for image with mask targets
    xml
    '''

    transforms = build_transforms(cfg, False)
    img=cv2.imread(imgfile,cv2.IMREAD_COLOR)

    H,W,C=img.shape
    boxes, gt_classes = annotation_onefile(xmlfile)
    imgmasklist=None
    if maskfile:
        mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
        imgmask=cv2.bitwise_and(img,img,mask=mask)
        imgmask=Image.fromarray(cv2.cvtColor(imgmask, cv2.COLOR_BGR2RGB))

    img=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    target = BoxList(boxes, (W,H), mode="xyxy")
    target = target.clip_to_image(remove_empty=True)
    # classes = [obj["category_id"] for obj in anno]
    # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
    # classes = torch.tensor(classes)
    # target.add_field("labels", classes)

    # masks = [obj["segmentation"] for obj in anno]
    # masks = SegmentationMask(masks, img.size, mode='poly')
    # target.add_field("masks", masks)

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

def dataBlobDiffBackground(imgfile,xmlfile,transforms):
    boxes, gt_classes = annotation_onefile(xmlfile)

    #for imgfile in imgfiles:
    img=cv2.imread(imgfile,cv2.IMREAD_COLOR)

    imgmask=np.ones_like(img)*125
    boxes=boxes.astype(np.int64)
    for xmin,ymin,xmax,ymax in boxes:
        imgmask[ymin:ymax,xmin:xmax]=img[ymin:ymax,xmin:xmax].copy()
    H,W,C=img.shape

    img=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imgmask = Image.fromarray(cv2.cvtColor(imgmask, cv2.COLOR_BGR2RGB))

    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    target = BoxList(boxes, (W,H), mode="xyxy")
    target = target.clip_to_image(remove_empty=True)
    # classes = [obj["category_id"] for obj in anno]
    # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
    # classes = torch.tensor(classes)
    # target.add_field("labels", classes)

    # masks = [obj["segmentation"] for obj in anno]
    # masks = SegmentationMask(masks, img.size, mode='poly')
    # target.add_field("masks", masks)
    #targetmask = target.copy_with_fields(target.fields(), True)
    targetmask = target.copy_with_fields(list(target.extra_fields.keys()), skip_missing=True)
    img, target = transforms(img, target)
    imgmask, targetmask = transforms(imgmask, targetmask)
    imglist = [img,imgmask]

    targets=[target,targetmask]

    imglist=to_image_list(imglist,cfg.DATALOADER.SIZE_DIVISIBILITY)

    return imglist,targets

def extractor(model,
            data_loader,
            dataset_name,
            device="cuda",
            output_folder=None,):
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
    FeaMapFolder=os.path.join(output_folder,'feaMap')
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
            roifeatures = foward_detector_roifeature(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
            #roifeatures = model.featureTarget(images,targets)
            #outputFeatures=[(p,r) for p,r in zip(proposals,rois)]

            try:
                outputFeatures=[target.get_field('featureROI').to(cpu_device) for target in targets]
                boxfeature_dict.update({img_id: result for img_id, result in zip(image_ids, outputFeatures)})
            except KeyError as e:
                print('Error ',image_ids,)


    torch.save(boxfeature_dict,os.path.join(output_folder,'targetFeatures.pth'))
    return boxfeature_dict

def extractorBatch(model,
            device="cuda",
            output_folder=None):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    # cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, False)
    MainPath = 'E:/SeaShips_SMD/ImageSets\Main'
    SSset = 'test_SeaShips.txt'
    SMDset = 'test_SMD.txt'
    MP = os.path.join(MainPath, SSset)
    with open(MP, 'r') as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    imgPath = 'E:/SeaShips_SMD/JPEGImages'
    xmlPath = 'E:/SeaShips_SMD/Annotations'

    for k in model:
        model[k].eval()
    boxfeature_dict = {}
    FeaMapFolder=os.path.join(output_folder,'feaMapD')

    if not os.path.exists(FeaMapFolder):
        os.mkdir(FeaMapFolder)

    diffps=[]
    diffpNs=[]
    for name in names:
        #imagefile= imagefiles[0]

        imgp=os.path.join(imgPath,name+'.jpg')
        xmlp=os.path.join(xmlPath,name+'.xml')
        batch=dataBlobDiffBackground(imgp,xmlp,transforms)

        images, targets = batch#images:(1,3,608,1088), targets:(600,1066)
        images = images.to(device)
        targets=[target.to(device) for target in targets]

        file_names=[]
        file_names.append('{}'.format(name))
        file_names.append('Mas{}'.format(name))

        with torch.no_grad():  # no compute the gradient
            #roifeatures = foward_detector_roifeature(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
            diffp,diffpN = foward_detector_roifeature_one(model, images, targets=targets,saveFolder=FeaMapFolder,imgnnames=file_names)
            diffps.append(diffp)
            diffpNs.append(diffpN)
            #roifeatures = model.featureTarget(images,targets)
            #outputFeatures=[(p,r) for p,r in zip(proposals,rois)]

            # try:
            #     outputFeatures=[roifeature.get_field('featureROI').to(cpu_device) for roifeature in roifeatures]
            #     boxfeature_dict.update({img_id: result for img_id, result in zip(image_ids, outputFeatures)})
            # except KeyError as e:
            #     print('Error ',image_ids,)
    torch.save({'diffps':diffps,'diffpNs':diffpNs },os.path.join(FeaMapFolder,'SSdiffp.pth'))
    #torch.save(boxfeature_dict,os.path.join(output_folder,'targetFeatures.pth'))
    return boxfeature_dict


def featureExtractor(cfg, model, comment='',useBatchFile=False):
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
            output_folder = os.path.join(cfg.OUTPUT_DIR, "extract"+comment, dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, is_train=False,shuffle=False)
    results=[]
    if useBatchFile:
        output_folder= os.path.join(cfg.OUTPUT_DIR, "extract"+comment, "customData")
        mkdir(output_folder)
        result=extractorBatch(model,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder)
    else:
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            result=extractor(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                device=cfg.MODEL.DEVICE,
                output_folder=output_folder,
            )
            #results.append(result)


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

def visualDifferBackground(path):
    #计算特征差异热力图
    smdpath=os.path.join(path,'SMDdiffp.pth')
    sspath=os.path.join(path,'SSdiffp.pth')

    dataSMD=torch.load(smdpath)
    dataSS=torch.load(sspath)
    print(dataSMD.keys(),dataSS.keys())

    diffps=dataSMD['diffps']+dataSS['diffps']
    Num=len(diffps)
    rN=Num
    inds=np.random.choice(Num,rN,False)
    diffpArr=np.zeros((1,)+diffps[0].shape[1:],np.float32)
    sum=0
    for ind in inds:
        diffp=diffps[ind]
    #for diffp in tqdm(diffps[a]):
        if diffp is None:
            continue
            #print(diffp.shape)
        sum+=len(diffp)
        diffpArr += diffp.sum(0)
    diffpArr/=sum
        #diffpArr=np.concatenate([diffpArr,diffp],axis=0)
    #diffpArr=diffpArr.mean(0)
    diffpArr=diffpArr.mean(0)
    diffpN = cv2.normalize(diffpArr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite('diffpN.jpg',diffpN)
    diffpM = cv2.applyColorMap(diffpN, cv2.COLORMAP_JET)  # 变成伪彩图
    #plt.imshow(diffpM[:,:,::-1],cmap=cm.hot)
    pltim=plt.imshow(diffpN, cmap=cm.jet)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation='horizontal',pad=0.05,anchor=(0,0,0.5))
    #plt.colorbar(orientation='horizontal')

    plt.savefig('color.png', bbox_inches='tight')
    cv2.imshow('b',diffpM)
    cv2.imwrite('SS+SMD{}.jpg'.format(rN),diffpM)
    cv2.waitKey(1)
    #diffpNs=data['diffpNs']
if __name__=="__main__":
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
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #
    # #imagefiles = os.listdir(imgPath)
    #
    # for name in names:
    #     name= names[0]
    #     #name,ext=os.path.splitext(imagefile)
    #     imgp=os.path.join(imgPath,name+'.jpg')
    #     xmlp=os.path.join(xmlPath,name+'.xml')
    #     # imgfile=imgp
    #     # xmlfile=xmlp
    #     batch=dataBlobDiffBackground(imgp,xmlp,transforms)
    output_dir = cfg.OUTPUT_DIR
    path='E:/DA2/logSSToSMDshipC/extractfeature/customData/feaMapD'#背景差异热力图
    visualDifferBackground(path)
    # exit(2)
    model = {}
    model["backbone"] = build_backbone(cfg).to(cfg.MODEL.DEVICE)#model+fpn
    model["fcos"] = build_rpn(cfg, model["backbone"].out_channels).to(cfg.MODEL.DEVICE)#feature map -> cls+reg



    checkpointer = DetectronCheckpointer(cfg, model,)

    _ = checkpointer.load(f=os.path.join(output_dir,cfg.MODEL.WEIGHT), load_dis=False, load_opt_sch=False)


    featureExtractor(cfg, model, comment=args.comment,useBatchFile=True)#'feature'

    # write_graph(model, output_dir)
    # for k in model:
    #     model[k].eval()
    # torch.save(model['backbone'],'backbone.pt')
    # torch.save(model['fcos'], 'fcos.pt')


