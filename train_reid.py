import os
import random
import torch
import numpy as np
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_stage
from config import cfg as cfg 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default=os.path.join(WORK_DIR,"configs/MURI/vcnet.yml"), help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = os.path.join(WORK_DIR,cfg.OUTPUT_DIR,cfg.TEST.EVAL_METHD)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train_loader, query_loader,gallery_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num,name="vcnet") 
    model_view = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num,name="vcnet_view")
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)


    model_view.load_param("/home/zhenjie/VCNet_re/logs/veri_vcnet_view/vcnetstage_view.pth") # 填写自己第一步训练得到的权重地址
    do_stage( # train id model 
        cfg,
        model_view,
        model,
        center_criterion,
        train_loader,
        query_loader,
        gallery_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, 
        output_dir
    )
