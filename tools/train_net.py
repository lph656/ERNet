"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in FsDet.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import os,pdb
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import launch

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from fsdet.evaluation import (
    DatasetEvaluators,
    verify_results,
    DefectEvaluator,
    NEUDefectEvaluator,
    TCALEvaluator
)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    # @classmethod是一个装饰器，在Python中用于定义类方法。
    # 类方法是绑定到类而不是实例的方法，可以通过类名直接调用，而无需创建实例。
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "defect":
            return DefectEvaluator(dataset_name)
        if evaluator_type == "neudefect":
            return NEUDefectEvaluator(dataset_name)
        if evaluator_type == "tcal":
            return TCALEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

# 函数接受一个配置文件cfg作为参数，并对该配置文件进行一系列修改后返回
def setup_new_config(cfg):
    # 使用残差连接
    cfg.MODEL.ETF.RESIDUAL = True
    # 背景类别的标签为1
    cfg.MODEL.ETF.BACKGROUND = 1
    # 使用交叉熵损失
    cfg.LOSS.TERM = "ce"
    # 调整损失函数中的温度参数
    cfg.LOSS.ADJUST_TAU = 1.0
    # 用于调整损失函数中的背景类别权重
    cfg.LOSS.ADJUST_BACK =  1000.0
    # 表示通过乘法方式调整损失函数的权重
    cfg.LOSS.ADJUST_MODE = 'multiply'
    # 表示调整损失函数的阶段是固定的
    cfg.LOSS.ADJUST_STAGE = 'fixed'
    # 不重置输出
    cfg.RESETOUT = False
    return cfg

# 根据传入的参数设置模型训练过程中的各种配置信息，包括输出目录、加载路径等。
def setup(args):
    """
    Create configs and perform basic set``ups.
    """
    cfg = get_cfg()
    cfg = setup_new_config(cfg)
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
                # cfg.LOSS.DISTILL_MAR
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
                load_path = 'checkpoints/voc/prior/ETFRes_pre{}_{}_lr20_adj{}_rfs{}_t1/model_clean_student.pth'.format(
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
                load_path = 'checkpoints/voc/prior/ETFRes_pre{}_{}_lr20_adj{}_rfs{}_t1/model_clean_student.pth'.format(
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
        elif data == 'tcal':
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
                load_path = 'checkpoints/voc/prior/ETFRes_pre{}_{}_lr20_adj{}_rfs{}_t1/model_clean_student.pth'.format(
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
    # 将配置对象cfg冻结，配置对象不可再修改
    cfg.freeze()
    # 将冻结后的配置对象cfg设置为全局配置
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    # 参数配置
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # 参数加载
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # 多GPU分布式训练
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
