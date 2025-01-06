"""
Detection Testing Script.

"""

import json
import os
import time

import detectron2.utils.comm as comm
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from fsdet.evaluation import (
    DatasetEvaluators,
    verify_results,
    DefectEvaluator,
    NEUDefectEvaluator
)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "defect":
            return DefectEvaluator(dataset_name)
        if evaluator_type == "neudefect":
            return NEUDefectEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "No Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Trainer.build_model(cfg)
        self.check_pointer = DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR
        )

        self.best_res = None
        self.best_file = None
        self.all_res = {}

    def test(self, ckpt):
        self.check_pointer._load_model(self.check_pointer._load_file(ckpt))
        print("evaluating checkpoint {}".format(ckpt))
        res = Trainer.test(self.cfg, self.model)

        if comm.is_main_process():
            verify_results(self.cfg, res)
            print(res)
            if (self.best_res is None) or (
                self.best_res is not None
                and self.best_res["bbox"]["AP"] < res["bbox"]["AP"]
            ):
                self.best_res = res
                self.best_file = ckpt
            print("best results from checkpoint {}".format(self.best_file))
            print(self.best_res)
            self.all_res["best_file"] = self.best_file
            self.all_res["best_res"] = self.best_res
            self.all_res[ckpt] = res
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, "inference"), exist_ok=True
            )
            with open(
                os.path.join(self.cfg.OUTPUT_DIR, "inference", "all_res.json"),
                "w",
            ) as fp:
                json.dump(self.all_res, fp)

def setup_new_config(cfg):
    # Example of additional config modification
    cfg.MODEL.ETF.RESIDUAL = True
    cfg.MODEL.ETF.BACKGROUND = 1
    cfg.LOSS.TERM = "ce"
    cfg.LOSS.ADJUST_TAU = 1.0
    cfg.LOSS.ADJUST_BACK = 1000.0
    cfg.LOSS.ADJUST_MODE = "multiply"
    cfg.LOSS.ADJUST_STAGE = "fixed"
    cfg.RESETOUT = False
    return cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg  = setup_new_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if cfg.RESETOUT:
        dir_comp = cfg.OUTPUT_DIR.split('/')
        shot = cfg.DATASETS.TRAIN[0].split('_')[-1] # The first one is always the novel set
        data = cfg.DATASETS.TRAIN[0].split('_')[0]
        model_key = 'ETFRes' if cfg.MODEL.ETF.RESIDUAL else 'ETF'
        if data == 'voc':
            the_split = cfg.DATASETS.TRAIN[0].split('_')[-2][-1]
            lr = int(1000*cfg.SOLVER.BASE_LR)
            bk_ratio = cfg.LOSS.ADJUST_BACK/1000.0
            rfs = cfg.DATALOADER.REPEAT_THRESHOLD * 100 if cfg.DATALOADER.SAMPLER_TRAIN == 'RepeatFactorTrainingSampler' else 0
            if cfg.LOSS.ADJUST_STAGE == 'distill':
                assert cfg.MODEL.BACKBONE.FREEZE 
                print('Logging modification for finetuning')
                loss_suffix = dir_comp[-1]
                new_out_dir = '{}_distill{}_{}_lr{}_{}'.format(
                    model_key, the_split, shot, lr, loss_suffix
                )
                new_out_dir = '/'.join(dir_comp[:3]+[new_out_dir])
                dir_comp = cfg.OUTPUT_DIR.split('/')
                load_path = 'checkpoints/voc/prior/ETFRes_pre{}_{}_lr20_adj{}_rfs{}_t1/model_clean_student.pth'.format(
                    the_split, shot, cfg.LOSS.ADJUST_BACK/1000.0, 5.0 if '2' in shot else 1.0,
                )
            else:
                print('Logging modification for pre-training')
                new_out_dir = '/'.join(dir_comp[:3]+[
                    '{}_pre{}_{}_lr{}_adj{}_rfs{}_{}'.format(model_key,the_split,shot,lr,bk_ratio,rfs,dir_comp[-1])
                    if cfg.LOSS.TERM == 'adjustment' else
                    '{}_pre{}_{}_lr{}_{}'.format(model_key,the_split,shot,lr,dir_comp[-1])
                ])
                load_path = None
        elif data == 'coco':
            lr = int(1000*cfg.SOLVER.BASE_LR)
            bk_ratio = cfg.LOSS.ADJUST_BACK/1000.0
            if cfg.LOSS.ADJUST_STAGE == 'distill':
                assert cfg.MODEL.BACKBONE.FREEZE 
                print('Logging modification for finetuning')
                loss_suffix = dir_comp[-1]
                new_out_dir = '{}_distill_{}_lr{}_adj{}_{}'.format(
                    model_key, shot, lr,
                    bk_ratio, loss_suffix
                )
                new_out_dir = '/'.join(dir_comp[:3]+[new_out_dir])
                load_path = 'checkpoints/coco/prior/ETFRes_pre_{}_lr20_adj{}.0_rfs2.5_t1/model_clean_student.pth'.format(
                    shot, 20 if '30' in shot else 10,
                )
            else:
                print('Logging modification for pre-training')
                rfs = cfg.DATALOADER.REPEAT_THRESHOLD * 100 if cfg.DATALOADER.SAMPLER_TRAIN == 'RepeatFactorTrainingSampler' else 0
                new_out_dir = '/'.join(dir_comp[:3]+[
                    '{}_pre_{}_lr{}_adj{}_rfs{}_{}'.format(model_key,shot,lr,bk_ratio,rfs,dir_comp[-1])
                    if cfg.LOSS.TERM == 'adjustment' else
                    '{}_pre_{}_lr{}_{}'.format(model_key,shot,lr,dir_comp[-1])
                ])
                load_path = None
        elif data == 'defect':
            the_split = cfg.DATASETS.TRAIN[0].split('_')[-2][-1]
            lr = int(1000 * cfg.SOLVER.BASE_LR)
            bk_ratio = cfg.LOSS.ADJUST_BACK / 1000.0
            rfs = cfg.DATALOADER.REPEAT_THRESHOLD * 100 if cfg.DATALOADER.SAMPLER_TRAIN == 'RepeatFactorTrainingSampler' else 0
            if cfg.LOSS.ADJUST_STAGE == 'distill':
                assert cfg.MODEL.BACKBONE.FREEZE
                print('Logging modification for finetuning')
                loss_suffix = dir_comp[-1]
                new_out_dir = '{}_distill{}_{}_lr{}_{}'.format(
                    model_key, the_split, shot, lr, loss_suffix
                )
                new_out_dir = '/'.join(dir_comp[:3] + [new_out_dir])
                dir_comp = cfg.OUTPUT_DIR.split('/')
                load_path = 'checkpoints/defect/prior/ETFRes_pre{}_{}_lr20_adj{}_rfs{}_t1/model_clean_student.pth'.format(
                    the_split, shot, cfg.LOSS.ADJUST_BACK / 1000.0, 5.0 if '2' in shot else 1.0,
                )
            else:
                print('Logging modification for pre-training')
                new_out_dir = '/'.join(dir_comp[:3] + [
                    '{}_pre{}_{}_lr{}_adj{}_rfs{}_{}'.format(model_key, the_split, shot, lr, bk_ratio, rfs,
                                                             dir_comp[-1])
                    if cfg.LOSS.TERM == 'adjustment' else
                    '{}_pre{}_{}_lr{}_{}'.format(model_key, the_split, shot, lr, dir_comp[-1])
                ])
                load_path = None
        elif data == 'neudefect':
            the_split = cfg.DATASETS.TRAIN[0].split('_')[-2][-1]
            lr = int(1000 * cfg.SOLVER.BASE_LR)
            bk_ratio = cfg.LOSS.ADJUST_BACK / 1000.0
            rfs = cfg.DATALOADER.REPEAT_THRESHOLD * 100 if cfg.DATALOADER.SAMPLER_TRAIN == 'RepeatFactorTrainingSampler' else 0
            if cfg.LOSS.ADJUST_STAGE == 'distill':
                assert cfg.MODEL.BACKBONE.FREEZE
                print('Logging modification for finetuning')
                loss_suffix = dir_comp[-1]
                new_out_dir = '{}_distill{}_{}_lr{}_{}'.format(
                    model_key, the_split, shot, lr, loss_suffix
                )
                new_out_dir = '/'.join(dir_comp[:3] + [new_out_dir])
                dir_comp = cfg.OUTPUT_DIR.split('/')
                load_path = 'checkpoints/neudefect/prior/ETFRes_pre{}_{}_lr20_adj{}_rfs{}_t1/model_clean_student.pth'.format(
                    the_split, shot, cfg.LOSS.ADJUST_BACK / 1000.0, 5.0 if '2' in shot else 1.0,
                )
            else:
                print('Logging modification for pre-training')
                new_out_dir = '/'.join(dir_comp[:3] + [
                    '{}_pre{}_{}_lr{}_adj{}_rfs{}_{}'.format(model_key, the_split, shot, lr, bk_ratio, rfs,
                                                             dir_comp[-1])
                    if cfg.LOSS.TERM == 'adjustment' else
                    '{}_pre{}_{}_lr{}_{}'.format(model_key, the_split, shot, lr, dir_comp[-1])
                ])
                load_path = None
        new_cfg_list = ['OUTPUT_DIR',new_out_dir] if load_path is None else ['OUTPUT_DIR',new_out_dir,'MODEL.WEIGHTS',load_path]
        cfg.merge_from_list(new_cfg_list)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # Set the model weights to your specific path
    weight_path = "/home/AC0109/taohj/Liph/test1/DiGeo-master/tools/checkpoints/GC10-DET/GC10_DET_final_30-shot.pth"

    if args.eval_only:
        model = Trainer.build_model(cfg)

        # Load the specified checkpoint
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            weight_path, resume=False
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)

            # Save evaluation results in JSON
            os.makedirs(
                os.path.join(cfg.OUTPUT_DIR, "inference"), exist_ok=True
            )
            with open(
                os.path.join(cfg.OUTPUT_DIR, "inference", "res_final.json"),
                "w",
            ) as fp:
                json.dump(res, fp)
        return res
    else:
        if comm.is_main_process():
            print("Please specify --eval-only to run this script")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.eval_only = True  # Enable evaluation mode

    print("Command Line Args:", args)

    # Launch the evaluation
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="tcp://127.0.0.1:{:05d}".format(
            np.random.choice(np.arange(0, 65534))
        ),
        args=(args,),
    )
