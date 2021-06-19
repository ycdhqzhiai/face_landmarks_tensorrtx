# face_landmarks_tensorrtx
+ 1.wts生成
具体ckpt转pb和pb生成wts文件可参考https://blog.csdn.net/ycdhqzhiai/article/details/117951582?spm=1001.2014.3001.5501
+ 2.运行
```shell
mkdir build && cd build
cmake ..
./shufflenet_v2 -s #生成trt引擎文件
./shufflenet_v2 -d #加载引擎文件并推理
```
