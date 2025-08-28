# 大作业题目一：Winograd 卷积优化 (GPU 版本)

## 简介
本目录下提供了 Winograd 算法的 GPU CUDA 实现，使用 Thrust device vectors 进行内存管理。主要包括以下文件：

- `naive_conv.cu` 文件中包含了使用朴素卷积算法的基准实现；
- `winograd_conv.cu` 文件中包含了经优化的 F(2x2, 3x3) 的 Winograd 卷积算法实现；
- `main.cu` 文件中包含了测试代码和性能评估。

## 编译和运行
编译规则在 Makefile 文件中定义，运行以下命令来编译。
```bash
make winograd
```
这相当于
```bash
nvcc -O3 -arch=compute_70 -code=sm_70 main.cu naive_conv.cu winograd_conv.cu -o winograd
```
`arch` 和 `code` 参数是针对实验使用的V100的优化选项。

`inputs` 文件夹中提供了一组测试样例，进行性能评估和正确性测试。要运行程序，可以使用以下命令：

```bash
./winograd inputs/config.txt
```
或者使用 `sbatch` 命令来提交作业。
```bash
sbatch run.sh
```
输出结果示例见 run.out。