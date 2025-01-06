import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    #parser.add_argument(
    #    "--src", type=str, default="../checkpoints/voc/prior/", help="Path to the main checkpoint"
    #)
    parser.add_argument(
        "--src", type=str, default="./checkpoints/voc/prior", help="Path to the main checkpoint"
    )
    args = parser.parse_args()
    return args

def reset_ckpt(ckpt):
    if "scheduler" in ckpt:
        del ckpt["scheduler"]
    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "iteration" in ckpt:
        ckpt["iteration"] = 0


if __name__ == "__main__":
    args = parse_args()

    # 遍历指定路径下的所有文件
    for exp in os.listdir(args.src):
        # 创建源检测器权重文件（src_ckpt）的完整路径。
        # 这里假设'model_final.pth'是所使用的检测器权重文件的命名规则
        src_ckpt = os.path.join(args.src,exp,'model_final.pth')
        # 创建目标检测器权重文件dst_ckpt的完整路径
        # 此处将原始文件中的model_final替换为model_clean_student
        dst_ckpt = src_ckpt.replace('model_final','model_clean_student')
        # 检查目标检测器权重文件是否已存在。
        # 如果存在，则打印一条消息表明该实验的清理模型已经存在，并继续迭代下一个项目
        if os.path.isfile(dst_ckpt):
            print('The cleaned model for distillation stage already exists for Exp {}'.format(exp))
            continue
        # 如果目标检测器权重文件不存在，那么检查源检测器权重文件是否存在。
        # 如果不存在，则打印一条消息表明找不到最终模型，并继续迭代下一个项目。
        elif not os.path.isfile(src_ckpt):
            print('The final model is not found, please check {}'.format(exp))
            continue

        # 如果源检测器权重文件存在，加载源检测器权重文件，将其存储在ckpt变量中
        ckpt = torch.load(src_ckpt)
        # 对ckpt中的优化器、学习率调度器和迭代次数删除或重置为0
        reset_ckpt(ckpt)
        # 针对ckpt['model']中包含‘box_’的键名，进行以下操作
        for key in [k for k in ckpt['model'] if 'box_' in k]:
            # 赋值键名对应的权重值，并将其存储在new_weight变量中
            new_weight = ckpt['model'][key].clone()
            # 将键名按'.'进行拆分，并将结果存储在key_comp变量中
            key_comp = key.split('.')
            new_key = '.'.join(key_comp[:1]+['student_'+key_comp[1]]+key_comp[2:])
            ckpt['model'][new_key] = new_weight

        # 将清理后的检测器权重文件保存到目标路径中
        torch.save(ckpt,dst_ckpt)
        print('The final model has been cleaned for {}'.format(exp))

