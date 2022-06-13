## 环境

python3

### 包依赖

```
pandas
openpyxl
```

## 配置

```
目前是在class TestSuit的__init__方法中硬编码测试数据
class RunningItem的每个实例将产生一条测试结果，使用一个DataConfig与MethodConfig创建RunningItem
class DataConfig("MNT4753"/"MNT6753"，输入数据量的对数，重复次数)
class MethodConfig("straus"/"pippenger"/"cpu",C,R)
只有pippenger的参数C会作为命令行参数传入cuda_prover_piecewise，其余参数目前没有被使用
并且需要cuda_prover_piecewise编译时启用了pippenger的多组参数C的实例
默认编译下实际运行使用的为("straus",5,32),("pippenger",7),("cpu")
```

## 运行

**在上级目录中执行**

```
python tester/tester.py
```

## 输出

将产生目录结构：

```
- data
  -- MNT4753-15-1 #一组测试数据，对应一条DataConfig
    --- params-0 
    --- input-0 
    --- output-0 #最后一次计算结果
    --- preprocessed-0 #仅当启用了straus方法时产生
    --- cpu_log #目录下有CPU方法计算的stdout，对应一条RunningItem
    --- pippenger-7_log #同上
    --- straus-5-32_log #同上
    --- checksum_cuda.txt #最后一次GPU计算输出的checksum值
    --- checksum.txt #最后一次CPU计算输出的checksum值
    --- ...
  -- ... #更多测试数据
- reports
  -- 0601164444.xlsx #6月1日16时44分44秒完成的测试报告
```

## 注

1. 重复执行将在执行一条RunningItem时删除相应的output，checksum，log等输出相关内容并重新计算，不会重新随机生成params，input，preprocessed等输入和预处理数据
2. 先完成CPU计算生成checksum.txt是校验GPU计算正确性必要条件，当checksum.txt存在时，每个GPU计算完成后将比对checksum_cuda.txt与checksum.txt，不一致将报错并终止脚本，不存在checksum.txt时不会进行校验
3. 确保磁盘空间，内存，显存充足，否则某一次测试校验失败将报错并终止脚本

