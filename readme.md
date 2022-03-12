# 数据结构与算法课程二级项目

## Open CV 识别图书ISBN号

燕大学子谨慎参考！

## 一：项目介绍

ISBN文件夹为相关资源文件夹：

```
ISBN/Test/*    /*测试集*/
	/Train/*    /*完整一百幅图*/
	/数字样例/*   /*模板集*/
	/样例/*  /*手动标注模板集*/
	/Submit.cpp  /*主程序*/
	/签名.cpp  /*生成二值签名的程序*/
	/Code2.cpp  /*另一种思路（主要是改了降噪函数，没有使用多线程）*/
```

`submit.cpp`为主程序，文件中所有路径均为相对目录。`签名.cpp`为生成二值化签名所用的程序，平时为全注释状态。



## 二：开发环境

VS 2022 企业版 + Open CV 4.5.5 +Open MP 4.5 + Git hub

环境配置可以参考：[VS2017配置opencv](https://blog.csdn.net/qq_41175905/article/details/80560429?utm_medium=distribute.wap_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.wap_blog_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=3) 使用过程中记得用vc15。

### 三：程序效果

运行训练集时最好情况下正确率在85左右，准确率在93左右；测试集时正确率在72左右，准确率80左右。更换模板集和可以有效提升识别效果，

注：在主程序中，使用多线程时，线程数最优值为当前电脑CPU的物理内核数-1，比如8核16线程，最佳的线程数就是8-1=7.

Open MP一般而言无需额外安装环境，在VS——项目——属性——C/C++——语言——Open MP支持修改为是，并添加`#include<omp.h> `即可。

### 四：已知BUG：

1. 使用测试集的时候运行到第97张样例会报错，但是单拿出去可以正常识别。
2. 多线程运行的时候过程中输出的正确率不可靠，准确率可靠。单线程运行都是可靠的。
