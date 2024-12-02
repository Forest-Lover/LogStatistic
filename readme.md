### 简介
日志过滤、归类、统计工具。支持多种输入格式，输出到文件(excel,txt)和标准输出。config目录、output目录，已经给出了一些实际的配置和输出(不含日志源文件)，可以参考。
项目配置灵活性比较大、主要工作在于根据实际日志格式编写合适的过滤和解析规则
使用步骤：
1. 根据日志文件格式，在define.json中定义好分割方式、字段映射规则
2. 根据实际的统计需求，在config目录中定义对应的json文件，包含input和output的配置
3. 运行脚本，查看统计结果(默认输出目录：output/配置文件名称/)

### 配置文件
1. define.json 预定义的处理行为，被config.json引用 (一般不用改)
  - field_mapping_def 日志文件中字段对应的关系
    - LINE 逻辑行划分后，再\n分隔计算(逻辑行是指按照分隔符切割后的字符串，根据分隔规则确定，不区分换行符)
    - WHOLE 逻辑行划分后，整行参与计算
    - STANDARD eg:```[2024-12-02 10:53:37.676924]	[debug]	[Player]	[updateCurPetFormation battle	false	Z00hMb8OlebPX2fz	1001100	Player(uid=32906, entityId=Z0hiqZrYVkUqGKtT, actorId=1)]	[[updateCurPetFormation]     ...SpaceEntities\PlayerComponent\PetsFormationComponent.lua:233]```
    - ETCD eg:```{"level":"debug","ts":"2024-12-02T00:01:06.254+0800","caller":"v3rpc/interceptor.go:175","msg":"request stats","start time":"2024-12-02T00:01:06.254+0800","time spent":"23.576µs","remote":"10.8.43.184:64071","response type":"/etcdserverpb.Cluster/MemberList","request count":-1,"request size":-1,"response count":-1,"response size":-1,"request content":""}```
    - BI eg:```{"event_ts":1733108017669,"detail":{"role_name":"lgz111122","item_type":4,"lv":1,"pet_uid":"Z00hMb8OlebPX2fv","pet_id":1001100,"cp":172,"skill_list":[11100010,11100020,10900080],"talent_list":[[5,0],[18,0],[24,0],[11,0]],"feature_list":[10102],"prop_info":[1,1,1,1,0,3],"overall_rating":"INTERFACE_DISPLAY_RATING_1"},"properties":{"role_name":"lgz111122","player_id":"Z0hiqZrYVkUqGKtT","logic_server":8043077,"vip_level":0,"aid":"","fpid_create_ts":0,"game_uid_create_ts":1732797097,"gameserver_id":8043077,"process_id":"8043077-game0","cluster_id":8043077,"game_uid":"32906","level":0},"event":"pet_change_flow","log_source":"gs"}```
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
```
python3 -m pip install -r requirements.txt
python3 logStatistic.py --conf=config/_example.json
```