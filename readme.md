### 配置文件
1. define.json 预定义的处理行为，被config.json引用 (一般不用改)
  - field_mapping_def 日志文件中字段对应的关系
    - LINE 逻辑行划分后，再\n分隔计算
    - WHOLE 逻辑行划分后，整行参与计算
    - STANDARD eg:```[2024-04-01 20:54:53.459] [info]  [TcpConnection] [ReadLoop start, remote addr 192.168.119.100:26001] [Connection.go:238]```
    - ETCD eg:```{"level":"debug","ts":"2024-04-02T20:46:23.091+0800","caller":"v3rpc/interceptor.go:175","msg":"request stats","start time":"2024-04-02T20:46:23.091+0800","time spent":"15.958µs", ...}```
    - [新增可以自行声明，声明后可以在config.json中引用]
      - seperator 定义分隔字符，其中(none, brackets, json)特殊处理
      - mapping 对应映射关系

  - merge_def 日志文本的聚类方式
    - [大数据集用K_MEANS，相似度高用DBSCAN]
    - [新增需要代码里面实现，使用参考define.json具体的说明]

2. config.json 定义具体的文件处理逻辑
  - input 定义输入的文件
    - root_path 日志文件路径根目录
    - file_filter 过滤文件，可以分多个组分别处理输出
    - field_mapping field分隔方式 [define.json].field_mapping_def
    - [可选]line_pattern [default:\n]逻辑行匹配表达式，field是按照逻辑行和对应的mapping来拆分的
    - [可选]line_filter [default:""]逻辑行过滤表达式
    - [可选]line_extract [default:\n]逻辑行提取表达式
    - [可选]field_filters [default:空]field过滤方式，每一项对应一种过滤规则，过滤field匹配的行，各种过滤规则间是'与'的关系
        - "field" : mapping中的字段定义
        - "match" : 对应字段值过滤表达式
  - output 定义输出的方式
    - toplist 归类输出方式，每一项对应一个输出 大数据集用K-means，数据相似度很高用dbscan
      - field 按照哪个域进行聚合 [define.json].field_mapping_def.mapping
      - merge 聚合的比较方式 [define.json].merge_def
      - count 按照数量排序前几名
    - merge_file 是否需要合并input文件进行统计，如不合并总输出项=#toplist*#input
    - [可选]write_log_file[default:false] 是否将input中预处理后的日志输出
    - [可选]p4_caller_info [default:false] 是否对CALLER打印p4的blame信息
    - [可选]p4_caller_ctx_len [default:-1] 是否对CALLER打印p4的blame的上下文代码（-1不打印、0本行、n前后各n行）
    - [可选]print_line_num [default:0]需要在每个分组输出后面打印的示例日志样本行数
    - [可选]print_line_field [default:全部]需要在每个分组输出后面打印的示例日志样本的field

### 使用方式
python3 -m pip install -r requirements.txt

python3 logStatistic.py --conf=config.cluster.json
python3 logStatistic.py --conf=config.service.go_ms.json
python3 logStatistic.py --conf=config.service.lua_ms.json