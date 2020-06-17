# tensorflow2.2_mnist_cnn_gpus
一、环境配置
系统要求：win10
显卡：Nvidia GeForce RTX 2080 ( 8 GB / 技嘉 )

python包：
tensorflow=2.2
tensflow-gpu=2.2
cuda=10.1
二、步骤
1，数组转化为tfrecords保存
2，读取tfrecords数据进行模型训练
3，实现gpus之间通信训练。
三、tesorboad文件查询
cmd
if exist checkpoint_dir" echo checkpoint_dir
linux
!ls {checkpoint_dir}
