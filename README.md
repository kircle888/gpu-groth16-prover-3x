## 环境配置与编译

与README-gpu-groth16-3x.md中一致
linux系统，Ubuntu18.04与Ubuntu20.04已经测试可用
CUDA10.0及以后
包配置：

``` bash
sudo apt-get install -y build-essential \
    cmake \
    git \
    libomp-dev \
    libgmp3-dev \
    libprocps-dev \
    python-markdown \
    libboost-all-dev \
    libssl-dev \
    pkg-config \
    nvidia-cuda-toolkit
```
### 编译

``` bash
./build.sh
```
编译成功后当前目录下应有generate_parameters,main,cuda_prover_piecewise三个可执行文件

## 运行

为支持更多测试，参数有一些调整

### 随机输入数据生成

``` bash
generate_parameters [B1] [B2]
```
其中[B1],[B2]为MNT4753，MNT6753的输入大小的以2为底对数值，0表示不生成对应输入数据
执行结果为在当前目录下产生MNT4753-parameters,MNT4753-input,MNT6753-parameters,MNT6753-input四个文件
不输入任何参数默认为输入了20和15

### CPU计算结果

（测试CPU计算耗时或与GPU计算对比验证正确性）

``` bash
main [MNT4753/MNT6753] compute [param] [input] [output]
```
[param],[input],[output]分别为两个输入文件的路径与输出文件的路径

例如：

``` bash
main MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output
main MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output
```
### 预处理

（不测试gpu-groth16-prover-3x的原方法可以不执行这步）

```bash
main [MNT4753/MNT6753] preprocess [param]
```

[param]与前述相同

将产生MNT4753_preprocessed或MNT6753_preprocessed文件

例如：

``` bash
main MNT4753 preprocess MNT4753-parameters 
main MNT6753 preprocess MNT6753-parameters
```

### GPU计算结果

``` bash
cuda_prover_piecewise [MNT4753/MNT6753] compute [param] [input] [output] [pippenger/straus] [PathOrC]
```

[param],[input],[output]与前述相同

输入pippenger将采用修改后的方法，输入straus将采用gpu-groth16-prover-3x的原代码，二者在main函数中就进入不同分支

[PathOrC]对于pippenger可输入参数C，main函数中通过实例化多个不同参数的模板函数实现可变的参数C，因编译过慢该部分通常被注释掉，因此不取消注释的情况下这个参数不会有任何效果，实际采用的参数C为7

[PathOrC]对于straus为preprocess文件的路径，必须正确输入

例如：

```bash
cuda_prover_piecewise MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output_cuda straus MNT6753_preprocessed
cuda_prover_piecewise MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output_cuda pippenger 7
```

### 验证正确性

```bash
sha256sum MNT4753-output MNT6753-output MNT4753-output_cuda MNT6753-output_cuda
```

### 程序输出

使用`std::chrono::high_resolution_clock`在程序中计时并输出

gpu e2e： xxx ms 为从启动GPU到所有multiexp计算完成时间（测试脚本采集该时间为 GPU Time）

cpu1: xxx ms 为CPU启动核函数后到完成计算开始同步等待核函数返回所花费的时间（测试脚本采集该时间为 CPU1）

Total time from input to output： xxx ms 为程序开始到程序结束的时间（测试脚本采集该时间为Total Time）

其余输出参看具体代码

### 测试脚本

simpledata/test.sh包含了运行一次测试的bash命令

tester/tester.py是多组测试与数据提取工具，具体信息参照tester/README.md

### 注

1. 每编译后首次运行时驱动加载与动态链接等耗时很长，至少成功执行一次cuda_prover_piecewise后再采集时间数据
2. `Fail to load xxx./Fail to allocate xxx./Segmentation Fault.`检查磁盘、内存、显存空间是否足够，输入文件是否正确生成并且路径正确

## 文件结构

```
- cuda-fixnum
  -- ...
- libsnark
  -- generate_parameters.cpp #输入数据生成
  -- main.cpp #CPU计算与预处理
  -- prover_reference_functions.cpp #构造Groth16系统用到的函数
  -- ...
- multiexp
  -- curves.cu #GPU的MNT4753与MNT6753椭圆曲线运算
  -- loader.cu #数据加载与内存分配函数
  -- reduce.cu #multiexp核函数
  -- oprs.cu #简单椭圆曲线操作核函数，大部分仅在debug时用到
  -- ...
- cuda_prover_piecewise.cu #GPU计算的主函数
```
