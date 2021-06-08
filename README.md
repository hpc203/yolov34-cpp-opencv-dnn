# 4种YOLO目标检测的C++和Python两种版本的实现
本程序包含了经典的YOLOv3，YOLOv4，Yolo-Fastest和YOLObile这4种YOLO目标检测的实现，
这4种yolo的.cfg和.weights文件，从百度云盘里下载

链接：https://pan.baidu.com/s/1xOZ-MM0G-NgmlYyrW0dGCA 
提取码：8kya 

下载完成后把下载得到的4个文件夹拷贝到和main_yolo.cpp同一目录下，
只要安装了opencv4.4.0及其以上版本的，就可以在windows和linux系统编译并运行main_yolo.cpp

此外，在Net_config配置参数项里，可以添加一个参数swapRB，控制输入图像是否交换RGB通道的，
之所以要添加这个参数，是因为我看到有的YOLO模型的输入图像并没有做交换通道到RGB的处理。
