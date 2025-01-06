这个仓库是2024 MS文章：**Enhanced Multiview Attention Network with Random  Interpolation Resize for Few-Shot Surface Defect  Detection** 的实现代码。

## 运行实验

### 数据集

我们使用了GC10-DET，NEU-DET和TCAL数据集来测试ERNet。

```
% GC10-DET dataset is proposed in:
@article{lv2020deep,
  title={Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network},
  author={Lv, Xiaoming and Duan, Feng and Jiang, Jianjun and Fu, Xiang and Gan, Liang},
  journal={Sensors},
  volume={20},
  number={6},
  pages={1562},
  year={2020},
  publisher={MDPI},
  doi={10.3390/s20061562}
}
% NEU-DET dataset is proposed in:
@article{song2013noise,
  title={A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects},
  author={Song, Kechen and Yan, Yunhui},
  journal={Applied Surface Science},
  volume={285},
  pages={858--864},
  year={2013},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.apsusc.2013.09.002}
}
% TCAL dataset is proposed in:
@misc{
        title={铝型材表面瑕疵识别数据集}, 
        url={https://tianchi.aliyun.com/dataset/dataDetail?dataId=140666},
        author={Tianchi},
        year={2016}, 
}
```

### 数据准备（以GC10-DET为例）

首先，并把原始文件的结构调整到下面这样：

```
datasets
└── GC10-DET
    ├── ANNOTATIONS
    │   ├── 1.xml
    │   ├── 2.xml
    │   └── ...
    ├── IMAGES
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── ...
└── gc10split
    ├── box_[x]shot_[defectname]_train.txt
    └── ...

```

可以通过以下链接[链接: https://pan.baidu.com/s/13ZRCwLaZMUr_RGS3ii5oQg?pwd=liph 提取码: liph]快速获取结构调整好的GC10-DET数据集，数据集下载后保存在路径./datasets/下

### 训练

要训练ERNet，这里是一个快速开始的命令：

```shell
cd ./tools/
python train_net.py
```

### 结果验证

可以通过以下链接[链接: https://pan.baidu.com/s/13ZRCwLaZMUr_RGS3ii5oQg?pwd=liph 提取码: liph]快速获取ERNet模型于GC10-DET在5-shot设置中的预训练参数，下载后，保存在路径：./tools/checkpoints，然后通过以下命令进行结果验证：

```shell
cd ./tools/
python test_net.py
```

### 注意

首先，项目下载后，首先将fsdet.zip解压到当前路径形成fsdet文件夹。
其次，项目中的 RIR 模块通过 `augmentation_impl.py` 和 `transform.py` 两个文件实现。在获取 Detectron2 后，可以将对应的两个文件替换为项目提供的版本，以实现该模块的功能。如果希望将该模块应用到自己的项目中，建议参考原论文的说明和代码，并根据实际需求进行重新设计和实现。

##  引用

如果您觉得这份代码对您的研究有帮助，请考虑引用我们：

```
@article{li2025enhanced,
  title={Enhanced Multiview Attention Network with Random Interpolation Resize for Few-Shot Surface Defect Detection},
  author={Li, Peng and Tao, Huizhen and Zhou, Heng and others},
  journal={Multimedia Systems},
  volume={31},
  pages={36},
  year={2025},
  publisher={Springer},
  doi={https://doi.org/10.1007/s00530-024-01643-y}
}
```

代码部分参考

```
@InProceedings{Ma_2023_CVPR,
    author    = {Ma, Jiawei and Niu, Yulei and Xu, Jincheng and Huang, Shiyuan and Han, Guangxing and Chang, Shih-Fu},
    title     = {DiGeo: Discriminative Geometry-Aware Learning for Generalized Few-Shot Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
}
```
