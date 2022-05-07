# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse

import time
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.engine.inference import inference
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

from tensorboardX import SummaryWriter
from ForTest import testbbox
import re
from ast import literal_eval

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("fcos_core", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    # model = build_detection_model(cfg)
    # model.to(cfg.MODEL.DEVICE)
    from fcos_core.modeling.backbone import build_backbone
    from fcos_core.modeling.rpn.rpn import build_rpn
    model = {}
    model["backbone"] = build_backbone(cfg).to(cfg.MODEL.DEVICE)
    model["fcos"] = build_rpn(cfg, model["backbone"].out_channels).to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(f=cfg.MODEL.WEIGHT, load_dis=False, load_opt_sch=False)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

def main_testbbox():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--flagVisual",type=bool,default=False)
    parser.add_argument("--flagEven", type=bool, default=False)
    parser.add_argument("--checkpoint",type=str,default=None)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=['MODEL.DOMAIN_ADAPTATION_ON',False,
                 'OUTPUT_DIR','log12'],

        nargs=argparse.REMAINDER,type=str
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    print(args.opts)
    print(args.flagVisual)
    time.sleep(2)
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.freeze()
    model_dir = cfg.OUTPUT_DIR

    logger = setup_logger("fcos_core", model_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    # model = build_detection_model(cfg)
    # model.to(cfg.MODEL.DEVICE)
    from fcos_core.modeling.backbone import build_backbone
    from fcos_core.modeling.rpn.rpn import build_rpn
    model = {}
    model["backbone"] = build_backbone(cfg).to(cfg.MODEL.DEVICE)
    model["fcos"] = build_rpn(cfg, model["backbone"].out_channels).to(cfg.MODEL.DEVICE)

    #output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_dir)
    #_ = checkpointer.load(f=cfg.MODEL.WEIGHT, load_dis=False, load_opt_sch=False)

    if args.checkpoint:
        try:
            cps = literal_eval(args.checkpoint)
        except:
            cps = [args.checkpoint]

        cfg.MODEL.WEIGHT=[os.path.join(model_dir,cp) for cp in cps]
    else:
        last_checkpoint=os.path.join(model_dir,'last_checkpoint')
        if os.path.exists(last_checkpoint):
            with open(last_checkpoint,'r') as f:
                lines=f.readlines()
                cfg.MODEL.WEIGHT=os.path.join(model_dir,os.path.basename(lines[-1]))
        else:
            cfg.MODEL.WEIGHT=os.path.join(model_dir,'model_final.pth')

    if args.flagEven:
        writerT = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'even_'+cfg.DATASETS.TEST[0]))#+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()).replace(':','-')
    if not isinstance(cfg.MODEL.WEIGHT,list):
        cfg.MODEL.WEIGHT=[cfg.MODEL.WEIGHT]
    for cp in cfg.MODEL.WEIGHT:
        _ = checkpointer.load(cp,load_dis=False, load_opt_sch=False)
        regxint=re.findall(r'\d+', cp)
        numstr=''
        if len(regxint)>0:
            numstr=str(int(regxint[-1]))
        save_dir = os.path.join(model_dir, 'inference' + numstr)
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        logger.info(cfg)

        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + collect_env_info())
        logger.info("results will be saved in %s"%(save_dir))

        testResult=testbbox(cfg,model,numstr,flagVisual=args.flagVisual)# will call the model.eval()

        if args.flagEven:
            try:
                for k, v in testResult[0][0].results['bbox'].items():
                    writerT.add_scalar(tag=k, scalar_value=v, global_step=int(numstr))
                    writerT.flush()
            except:
                print('Error:testResult is empty!')


if __name__ == "__main__":
    main_testbbox()
