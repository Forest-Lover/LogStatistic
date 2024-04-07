### 配置文件
1. define.json 预定义的处理行为，被config.json引用
  - field_mapping_def 日志文件中字段对应的关系
    - LINE  逻辑行划分后，再\n分隔计算
    - WHOLE 逻辑行划分后，整行参与计算
    - STANDARD  eg:[2024-04-01 20:54:53.459] [info]  [TcpConnection] [ReadLoop start, remote addr 192.168.119.100:26001] [Connection.go:238]
    - ETCD      eg:{"level":"debug","ts":"2024-04-02T20:46:23.091+0800","caller":"v3rpc/interceptor.go:175","msg":"request stats","start time":"2024-04-02T20:46:23.091+0800","time spent":"15.958µs","remote":"192.168.119.100:53798","response 
    type":"etcdserverpb.Cluster/MemberList","request count":-1,"request size":-1,"response count":-1,"response size":-1,"request content":""}
    - 说明：其它的可以自行声明，声明后可以在config.json中引用
      - seperator 定义分隔字符，其中(none, brackets, json)特殊处理
      - mapping 对应映射关系

  - merge_def 日志文本的聚类方式，参考define.json具体的说明

2. config.json 定义具体的文件处理逻辑
  - input 定义输入的文件
    - file_filter 过滤文件，可以分多个组分别处理输出
    - line_filter [可选 default:\n]逻辑行过滤方式，field是按照逻辑行和对应的mapping来拆分的
    - field_mapping field分隔方式，在define.json中定义
    - field_filters [可选 default:空]field过滤方式，每一项对应一种过滤规则，过滤field匹配的行，各种过滤规则间是'与'的关系
  - output 定义输出的方式
    - toplist 归类输出方式，每一项对应一个输出
    - merge_file 是否需要合并input文件进行统计，如不合并总输出项=#toplist*#input
    - print_line_num [可选 default:0]需要在每个分组输出后面打印的示例日志样本行数
    - print_line_field [可选 default:全部]需要在每个分组输出后面打印的示例日志样本的field

### 算法说明
```
K-means和DBSCAN都是聚类算法，它们用于将数据点分组成若干个簇，使得同一个簇内的点相互之间更相似，而不同簇的点相互之间差异较大。不过，这两种算法在聚类的方法和适用场景上有所不同。

K-means 是一种基于划分的聚类算法，它需要预先指定簇的数量（K）。算法通过迭代过程优化簇内的点与簇中心的距离之和，最终得到相对紧凑的簇。K-means适合于簇大小相似、簇形状为球形的数据集。

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法，它不需要预先指定簇的数量。DBSCAN根据核心点（在指定半径内有足够多邻居的点）和边界点（在半径内邻居数量少于指定阈值但属于某个核心点的邻域的点）的概念来形成簇。DBSCAN能够识别出任意形状的簇，并且能够将噪声点（不属于任何簇的点）识别出来。

K-means输出：
  * 一个字符串列表，其中包含了该簇的质心（中心点）的前10个最重要的术语。这些术语是通过 TfidfVectorizer 提取的特征（单词或短语），并根据它们在质心向量中的权重排序
  * 该簇中与质心最相似的日志条目。这是通过计算簇中每个日志条目与质心的余弦相似度来确定的。在您的代码中，选择了相似度最高的日志条目作为簇的代表

DBSCAN输出：
  * 该簇的第一个代表性消息示例

需要注意的是，DBSCAN的簇编号通常从0开始，而且可能包含一个特殊的簇编号（如-1），用于表示噪声点。在你的输出中，只显示了编号为0的簇，这可能是最大的簇，或者是数据中唯一显著的簇。如果有噪声点，它们可能没有在这个输出中显示(N/A)。
```

```
K-means和DBSCAN是两种不同类型的聚类算法，它们在计算复杂度和运行时间上有所不同。在某些情况下，K-means的计算速度确实可能比DBSCAN快，但这并不是绝对的，因为它取决于多种因素，包括数据集的大小、维度、簇的数量以及算法的实现等。

K-means 的计算复杂度大致是 O(nkt)，其中 n 是数据点的数量，k 是簇的数量，t 是迭代次数。K-means通常需要多次迭代才能收敛，但每次迭代通常都比较快，尤其是当k相对较小且数据集不是非常大时。K-means的一个优点是它的简单性，这使得它在低维数据上非常高效。

DBSCAN 的计算复杂度在最佳情况下是 O(n log n)，但如果使用简单的邻域查询，复杂度可能会达到 O(n^2)。DBSCAN的速度取决于数据点之间距离计算的速度，以及如何有效地查询数据点的邻域。DBSCAN不需要迭代，但是它需要对每个点进行邻域查询，这在高维数据或者数据点之间的距离计算很复杂时可能会很慢。
DBSCAN 需要存储所有数据点之间的距离矩阵，这是一个 O(N^2) 的空间复杂度，其中 N 是数据点的数量。对于大数据集，这个矩阵可能非常大。

以下是一些影响两种算法速度的因素：
  数据集大小：K-means在大数据集上可能会比DBSCAN快，因为DBSCAN的邻域查询在大数据集上可能非常慢。
  簇的数量：K-means需要预先指定簇的数量，如果k值很大，它的速度可能会受到影响。DBSCAN不需要这样的预设，但是它的性能会受到最小点数(minPts)和邻域大小(ε)参数的影响。
  数据维度：随着数据维度的增加，DBSCAN的性能可能会下降得更快，因为高维空间中的距离计算更加复杂。
  数据分布：如果数据集包含大量的噪声点或者簇的形状非常不规则，DBSCAN可能会比K-means表现得更好，因为K-means假设簇是球形的。
  算法实现：算法的具体实现也会影响其性能，例如使用高效的数据结构如KD树或球树可以加速DBSCAN的邻域查询。

总的来说：大数据集用K-means，数据相似度很高用dbscan
```

### 使用方式
python3 -m pip install -r requirements.txt

python3 logStatistic.py --conf=config.cluster.json
python3 logStatistic.py --conf=config.lua_ms.json
python3 logStatistic.py --conf=config.go_ms.json
python3 logStatistic.py --conf=config.etcd.json
python3 logStatistic.py --conf=config.agent.json
