# ChangeFace
简单的机器学习，有趣的Python换脸术

此文章是转载的，我可没有实力做出这个

原文地址：<https://www.w3cschool.cn/python3/python3-egnr2z81.html>

首先需要的模块，用pip下载

python-opencv模块；dlib模块；numpy模块。
```powershell
pip install python-opencv
pip install dlib
pip install numpy
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200506142943619.png)
## 原理简介
**主要流程：**

（1）利用dlib库检测并获取人脸特征点；

（2）通过一些简单的处理使得第二张人脸的眼睛、鼻子和嘴巴较好地“装”到第一张人脸上。

**一些细节：**

特征检测器：

用的dlib官方提供的预训练好的模型。

第二张图片的人脸特征需要对齐到第一张图片的人脸特征，其实现参考了：

https://en.wikipedia.org/wiki/Procrustes_analysis#Ordinary_Procrustes_analysis

具体实现方式详见相关文件中的源代码。

## 使用演示
修改SwapFace.py文件的图片路径为自己需要操作的图片路径：

（1）特朗普+奥巴马

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUyNjU2NDk1ODkuanBn?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUyNjkzMjMzMzgucG5n?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUyNzM2MDIyNzYuanBn?x-oss-process=image/format,png)

（2）普及+安倍

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUyNzgzMjYxODQucG5n?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUyODIxNjU1NTAucG5n?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUyODY0MTE3MDcuanBn?x-oss-process=image/format,png)

（3）乔布斯+比尔盖茨

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUyOTA4NTU3MTYucG5n?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUyOTUyNDk4MTAuanBn?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUzMDA3NzI3MDAuanBn?x-oss-process=image/format,png)

（4）莱布尼兹+牛顿

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUzMDQ5NzUwMzEuanBn?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUzMTA2ODMwMjcuanBn?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUzMTYxNDExNjQuanBn?x-oss-process=image/format,png)

（5）爱因斯坦+薛定谔

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUzMjAyMjM3OTAuanBn?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUzMjQ4NTkwODUuanBn?x-oss-process=image/format,png)![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93d3cudzNjc2Nob29sLmNuL2F0dGFjaG1lbnRzL2ltYWdlLzIwMTgwODAyLzE1MzMyMDUzMjk2NjczMTcuanBn?x-oss-process=image/format,png)

That's All.

###### 联系我？

博客：<https://blog.csdn.net/cool99781>

邮箱：<3392446642@qq.com>