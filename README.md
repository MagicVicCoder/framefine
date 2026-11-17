# FrameFine：强化学习驱动的条件森林视频压缩

FrameFine 面向问题条件下的视频理解轻量化：从 N 个候选帧中联合保留恰好 K 个真实帧，再交给冻结的 Video-LLM 回答。核心是**学习压缩时哪些帧应当共同保留，以及少量上下文选择变量如何改变这些依赖**。当前实现做时间维度的选帧压缩，不修改视频编码格式，也不微调阅读器参数。


## 1. 研究立意与待检验假设

独立地预测每帧重要性，再取前 K 帧，缺少显式的联合选择机制。动作前后状态可能互补，相似帧可能冗余；问题和背景又会改变这种关系。自回归策略同样可以表达复杂依赖，因而本项目的价值不能建立在“自回归无法建模关系”上。

准备检验的贡献链条是：

1. 通过**等保留帧数的交换反馈**定义任务相关的跨帧依赖，避免把单帧分数直接解释为互补性。
2. 学习少量条件选择变量，使残余帧间的关系随上下文变化，而不是假定一套固定二阶关系适用于所有背景。
3. 将关系监督落实到可归一化的联合压缩策略，精确计算给定结构后的概率、熵、采样和最优 K 帧组合。

核心假设是：**少量条件变量加稀疏残余关系，能够解释主要压缩失误中的依赖变化。** 如果依赖分散在大量帧上，或重要的三阶以上交互不能被少数条件变量吸收，这个表示会受限。代码没有把这个假设写成已经成立的结论。

RL 视频压缩已有 [CaCoVID](https://arxiv.org/abs/2602.01649)，问题条件选帧已有 [ReFoCUS](https://arxiv.org/abs/2506.01274)。条件化、矩阵树定理、计数动态规划和 Rao–Blackwell 化也都是已有工具。潜在新意应落在“任务交换反馈—条件变量学习—可计算联合压缩”的联系和实际效果上。

## 2. 方法与模块

~~~mermaid
flowchart LR
    V[视频与问题] --> C[候选帧与冻结 CLIP 特征]
    C --> A[问题/时间条件 Transformer]
    A --> U[无放回选择 h 个条件变量]
    U --> T[每个条件状态的带虚拟根森林]
    T --> D[固定 K 的精确计数推断]
    D --> S[保留 K 个真实帧]
    S --> R[冻结 Video-LLM]
    R --> G[任务效用与共享交换反馈]
    G --> A
~~~

### 2.1 输入与轻量 actor

候选帧首先按原时间均匀采样，默认 N=32。冻结 CLIP 为帧和问题生成归一化特征；同一个视频的多个问题共用帧特征。actor 只接收帧特征、**问题文本特征**和时间戳，不接收正确答案或选项文本。选项仅用于冻结阅读器的多选题评分。

actor 使用单层、四头 Transformer，默认 hidden=64。其输出包含条件变量选择头、上下文条件化的一元势、边势、结构权重和状态势。候选数不足 N 时保留所有不同候选，不复制帧；若不足 K 或 h，训练/选择会报错，应调整预算或筛除短视频。

### 2.2 条件结构与联合策略

记输入为 x、帧掩码为 z∈{0,1}^N、结构动作为 Ψ。结构先无放回选择**有序**索引 U，|U|=h；默认 h=2。U 表示需要枚举保留状态的变量，**不表示这些帧必须保留**。

对于每个可行状态 c=z_U，actor 产生一个带虚拟根的加权无向生成树 T_c。删除虚拟根即得到残余变量上的森林 E_c。结构策略为

$$
q_\theta(\Psi\mid x)
= q_\theta(U\mid x)\prod_{c\text{ feasible}}q_\theta(T_c\mid U,c,x).
$$

树分布用矩阵树定理归一化，训练使用 Wilson 加权树采样，确定性解码使用最大生成树。结构概率包括**全部可行状态的树**，不能只计算最终掩码访问到的一棵树。

给定结构后的能量和数量约束策略为

$$
F_\theta(z;\Psi,x)
=\gamma_\theta(c,x)
+\sum_{i\notin U}a_{\theta,i}(c,x)z_i
+\sum_{(i,j)\in E_c}b_{\theta,ij}(c,x)z_i z_j,
\qquad c=z_U,
$$

$$
\pi_\theta(z\mid\Psi,x,K)
=\frac{\mathbf{1}[\sum_i z_i=K]\exp(F_\theta(z;\Psi,x)/\tau)}
{Z_\theta(\Psi,x,K)},
$$

$$
Z_\theta=\sum_{c\text{ feasible}}
\exp(\gamma_\theta(c,x)/\tau)\,
Z_{\text{forest}}(c,K-|c|).
$$

计数 DP 在对数域计算森林分区函数，并在**所有可行 c 之间全局归一化**，不能将各分支当成等权混合。它提供精确的 log probability、条件熵、采样和给定结构后的 MAP。

条件位对残余边势的调制可以形成高阶能量项，但不代表任意高阶依赖。固定 K 本身也会引入统计依赖，所以“二阶能量”不等同于“只有二阶统计关联”。确定性条件变量/树解码加条件 MAP，不是对所有 Ψ 边缘化后策略的精确 MAP。

### 2.3 冻结阅读器与任务效用

对 MCQ 的选项字母 A、B、…评分。若 l_a 为答案字母的最后位置 logit，则

$$
u(x,z)=l_{y^*}(x,z)-\log\sum_{a\ne y^*}\exp(l_a(x,z)).
$$

正确答案只在阅读器前向**之后**用于构造训练效用，不写入提示或 actor 输入。预测为字母 logits 的 argmax；准确率与连续效用分别记录。这里只支持 2–26 选项、每个字母为一个 token 的 MCQ，不将开放式问答自动套用此效用。

阅读器始终冻结，不生成长推理。InternVL/LLaVA 仅对最后隐藏状态投影选项对应的词表行；Qwen 使用最后位置 logits 接口。不同阅读器的效用尺度不同，更换底座应另开训练。

### 2.4 等数量交换关系监督

从一个保留 K 帧的种子掩码中，选择 d 对互不重叠的“已选帧—未选帧”，每个位决定是否交换。全部 2^d 个角点均保留 K 帧，默认 d=3：

- 8 个掩码的阅读器评分产生 6 个二维面；
- 每个面给出交换交互 Δ_iΔ_j u=u_11−u_10−u_01+u_00；
- 三阶差分衡量交互随第三个交换变化的程度，作为诊断信号记录。

这测量的是指定背景下的**交换交互**，不自动等于帧的内在重要性或可识别的因果效应。关系辅助损失为

$$
L_{\rm rel}(\theta,\Psi;\mathcal B)
=\operatorname{mean}_{\text{faces in }\mathcal B}
\left(\Delta_i\Delta_j F_\theta-\Delta_i\Delta_j u\right)^2.
$$

关系库在当前任务/结构动作之前由停止梯度的参考策略生成，固定后供本步全部结构共享。默认每 4 步采一块，不能把 8 次评分再乘以结构数。任务动作随后独立采样，避免错误地把与当前结构相关的反馈当作结构策略的独立监督。三阶差分不再额外加一个重复的损失。

### 2.5 训练目标与实际梯度

对本步固定的关系库 B，优化

$$
J_{\mathcal B}(\theta)
=\mathbb E_{\Psi\sim q_\theta}
\left[\mathbb E_{z\sim\pi_\theta}u(x,z)
+\tau H(\pi_\theta(\cdot\mid\Psi,x,K))
-\lambda L_{\rm rel}(\theta,\Psi;\mathcal B)\right].
$$

每步 G 个独立结构与任务动作，默认 G=8；使用 leave-one-out 均值基线。令 sg 为停止梯度，

$$
A_g^{\rm mask}=\operatorname{sg}\!\left(u_g-\operatorname{mean}_{j\ne g}u_j\right),\quad
R_g=u_g+\tau\operatorname{sg}(H_g)-\lambda\operatorname{sg}(L_{{\rm rel},g}),
$$

$$
A_g^{\rm structure}=\operatorname{sg}\!\left(R_g-\operatorname{mean}_{j\ne g}R_j\right),
$$

$$
\mathcal L
=-\operatorname{mean}_g
\left[A_g^{\rm mask}\log\pi_g+A_g^{\rm structure}\log q_g\right]
-\tau\operatorname{mean}_gH_g
+\lambda\operatorname{mean}_gL_{{\rm rel},g}.
$$

熵和关系损失既有连续参数路径，也通过结构回报影响结构选择。关系库的生成分布不反传；这里声称的是**固定库目标的梯度**，没有声称对自适应建库过程的全部路径求导。

一次更新顺序：建共享库（如到周期）→采 G 个结构→各自采一个 K 帧掩码→冻结阅读器评分→计算两类优势与直接损失→裁剪梯度→AdamW 更新 actor。框架没有 PPO 旧策略复用，也不将缓存旧掩码冒充新 on-policy 动作；缓存只复用同一输入的阅读器评分。

`exchange.py` 还提供独立测试过的交换块 Rao–Blackwell 任务梯度工具：对块内条件权重停止梯度，平均**完整策略**的 score gradient。它没有替换主训练器，也不保证每次阅读器调用的收益优于同数独立采样。

## 3. 安装与最小验证

推荐在 GPU 服务器上使用独立 Python 环境，Python ≥3.10。先按服务器 CUDA 版本安装 PyTorch，再安装项目：

~~~bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m framerefine.cli demo --steps 6 --output outputs/demo
~~~

`demo` 是解析合成任务，只检查梯度、优化器、缓存和检查点。测试包含小空间穷举对照分区函数/熵/MAP/梯度、加权树归一化、交换梯度恒等式，以及真实图片文件驱动的本地流水线。模型协议测试用小型替身，不下载权重。

按所选底座安装**一套**可选依赖，建议各自独立环境，不要把三套底座依赖混装：

~~~bash
# Qwen2.5-VL
python -m pip install -e ".[features,video,qwen]"

# InternVL2.5（独立环境）
python -m pip install -e ".[features,video,internvl]"

# LLaVA-Video（独立环境，另需下文的官方源码）
python -m pip install -e ".[features,video,llava]"
~~~

权重、数据与上游源码均由使用者事先放到本地。配置里的路径须指向完整快照目录，`revision` 填不可变快照 commit 或自管校验值，不能保留 REPLACE。模型加载器不会自动下载；InternVL 会执行本地快照内的模型自定义代码。

### 三类模型接口

| family | 对应目标 | 输入/评分约定 | 需填写 |
|---|---|---|---|
| `qwen2_5_vl` | Qwen2.5-VL-3B/7B-Instruct | 原生视频帧列表；实际时间戳写入文本；最后位置字母 logits | 本地模型、revision、devices |
| `internvl` | InternVL2_5-8B 的 InternLM/Llama base-LM 接口 | 每帧一个原生图块；原生 IMG_CONTEXT；有界冻结视觉特征缓存 | 本地模型和自定义代码、revision、devices |
| `llava_video` | LLaVA-Video-7B-Qwen2 | 原生 qwen_1_5 会话和 video 模态；最多 64 帧 | 本地语言模型、SigLIP tower、两者 revision、devices |

单卡先检查接口时，将 `devices` 改为一个设备。八卡模式是一卡一个冻结副本，单个协调器分派独立掩码；没有 DDP 训练阅读器，也没有模型跨卡切分。能否在 48GB 中容纳指定分辨率与全帧基线，仍需实际测量。

Qwen 帧列表使用 `sample_fps` 和普通路径，支持中文及空格路径；它使用名义 FPS，不是任意不规则时间戳的逐帧 mRoPE。InternVL 固定单图块可能与官方多图块最佳配置有精度差异。对比压缩与基线时保持相同提示、像素与时序约定。

LLaVA 额外需要 [官方 LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) 源码，例如放到 `/opt/LLaVA-NeXT` 后执行：

~~~bash
python -m pip install --no-deps -e /opt/LLaVA-NeXT
~~~

这里直接使用 `LlavaQwenForCausalLM` 原生多模态准备接口，避免通用 builder 的远程视觉塔和设备重定向。另行准备 `siglip-so400m-patch14-384` 本地目录。构建期间显式将这个路径交给 `SigLipVisionTower`，避开上游将既存路径优先分派为 CLIP 的问题，并在结束/异常时恢复上游函数。会话使用与原生 qwen_1_5 一致的 CHATML 格式，不导入会初始化额外远程分词器的 conversation 模块。上游依赖版本变化可能需要按固定源码 commit 调整，当前 extras 是接口约束，**不是已经通过真实模型验证的锁定环境**。

官方格式来源：[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[InternVL2.5](https://huggingface.co/OpenGVLab/InternVL2_5-8B)、[LLaVA-Video](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)。

## 4. 数据接口与特征准备

### 4.1 原始 JSONL

一行一个问题，路径相对清单所在目录。视频、帧目录、显式帧列表三种来源只能选一种。例如：

~~~json
{"id":"train-001-q1","video_path":"videos/001.mp4","question":"What happened first?","options":["Opening the door","Sitting down"],"answer_index":0,"start":0.0,"end":20.0,"metadata":{"split":"train"}}
~~~

已有帧可使用 `frame_paths` 加同长度秒级 `timestamps`，或者 `frame_dir` 加真实 `fps`。帧目录按数字自然排序。训练/效用评估必须带零基 `answer_index`；纯压缩推理可省略答案，但仍提供问题和 2–26 个选项。支持字母/答案文本标注的导入转换。

所有候选帧按原时间排序且互不重复。视频解码按容器 FPS 估计时间，变帧率视频需要自行提供真实时间戳帧清单。准备过程执行原始文件哈希、解码、JPEG 缓存和冻结特征提取；这些都属于成本。

~~~bash
python -m framerefine.cli import-data --annotations data/source.jsonl --format jsonl --output data/train.raw.jsonl
# 填写 configs/features.toml 的本地 CLIP 路径与 revision
python -m framerefine.cli prepare --raw-manifest data/train.raw.jsonl --config configs/features.toml --output data/train
python -m framerefine.cli prepare --raw-manifest data/validation.raw.jsonl --config configs/features.toml --output data/validation
~~~

默认使用冻结 `CLIPModel`（如 clip-vit-base-patch32），图像缓存长边 ≤448，最多 32 个候选；问题使用 CLIP 文本长度内截断，不编码选项或答案。长问题截断的影响需关注。缓存按内容、编码器身份、时间范围、候选数量和图像规格确定，复跑复用同视频帧特征。`preparation.json` 记录本次编码器启动、预处理和整条 prepare 命令的耗时，`preparation.history.jsonl` 追加保留每次成功准备的成本。显式 `metadata.source_video` 会保留，用于同视频不同帧子集的分组检查。

### 4.2 基准标注转换

提供本地 [Video-MME](https://github.com/MME-Benchmarks/Video-MME)、[MVBench](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)、[MLVU](https://github.com/JUNJIE99/MLVU) 多选标注转换。数据版本/目录可能不同，请将每个任务真实视频目录传入 `--video-root`，必要时用 `--video-prefix`，不猜测私有下载布局。

~~~bash
python -m framerefine.cli import-data --format videomme --annotations /data/videomme.json --video-root /data/Video-MME/data --output data/videomme.raw.jsonl
python -m framerefine.cli import-data --format mvbench --annotations /data/mvbench/json/action_sequence.json --task action_sequence --video-root /data/mvbench/video/star/Charades_v1_480 --output data/mvbench.raw.jsonl
python -m framerefine.cli import-data --format mlvu --annotations /data/mlvu/1_plotQA.json --video-root /data/mlvu/video/1_plotQA --output data/mlvu.raw.jsonl
~~~

Video-MME 支持 `videoID + questions` 嵌套 JSON；MVBench/MLVU 支持 `video/question/candidates/answer` 平面多选记录，保留 start/end。帧目录任务可补 `--fps`。不支持的任务形式应先转成原始 JSONL；字幕、音轨和开放式问答不在当前适配范围。

训练和验证请按**视频**隔离，同视频多个问题不能随机分到两侧。周期验证会检查 ID 和来源视频重叠；官方测试集应保留为独立评估，不能直接作为训练集。小型周期验证只用于调试/选择检查点，不能代替完整测试。

### 4.3 预计算特征接口

如需接入其他冻结特征编码器，直接生成 tensor-only 文件与清单：

~~~python
torch.save({
    "frame_features": frames,  # [N,D], finite float tensor
    "query_features": query,   # [Q]
    "timestamps": stamps,      # [N], seconds in original order
}, "features/q1.pt")
~~~

~~~json
{"id":"q1","feature_path":"features/q1.pt","question":"What changed?","options":["Opened","Closed"],"answer_index":0,"content_revision":"immutable-frame-bytes-and-timeline-hash","frame_paths":["frames/000.jpg","frames/001.jpg"],"metadata":{"feature_encoder":{"model":"local-encoder","revision":"immutable-v1"},"source_video":"video-001"}}
~~~

`frame_paths` 的数量必须等于 N，上例只是格式示意。也可将帧特征与时间戳放在共享 `frame_feature_path`，问题文件只存 `query_features`。数据集按需加载，文件 LRU 有界；`torch.load(weights_only=True)` 不依赖任意 Python pickle 对象。更改像素、特征或候选映射必须更新对应 revision。训练检查点保存特征编码器身份；标准 prepare 输出在选择/评估时会检查编码器是否一致。旧的自备特征若不声明编码器身份，无法获得这项保护。

新数据集只需实现原始 JSONL 转换；新阅读器只需实现 `identity` 和 `score_one(example, indices) -> ReaderScore(utility, prediction)`，再在 `models/factory.py` 注册。identity 应包含不可变权重、处理器、提示和评分版本，不能仅写模型名称。

## 5. 训练、恢复与推理

配置已按 **8 × 改装 48GB RTX4090；试跑 ≤2 小时；正式训练 ≤48 小时** 设置。先修改 `configs/features.toml` 和所选 `configs/readers/*.toml` 的真实本地路径/revision。

~~~bash
# 首先短跑检查真实接口与吞吐，输出目录独立
python -m framerefine.cli train --config configs/probe.toml --reader-config configs/readers/qwen.toml --manifest data/train/manifest.jsonl --output outputs/profile --limit-steps 10

# 2 小时试跑，含独立周期验证
python -m framerefine.cli train --config configs/probe.toml --reader-config configs/readers/qwen.toml --manifest data/train/manifest.jsonl --validation-manifest data/validation/manifest.jsonl --output outputs/probe

# 在同一语义配置下续跑；累计旧耗时及新启动耗时
python -m framerefine.cli train --config configs/probe.toml --reader-config configs/readers/qwen.toml --manifest data/train/manifest.jsonl --validation-manifest data/validation/manifest.jsonl --output outputs/probe --resume outputs/probe/last.pt

# 正式训练是单独运行
python -m framerefine.cli train --config configs/formal.toml --reader-config configs/readers/qwen.toml --manifest data/train/manifest.jsonl --validation-manifest data/validation/manifest.jsonl --output outputs/formal
~~~

其他底座只需替换 reader-config，不混用旧奖励缓存或旧训练检查点。既有输出目录需要显式 `--resume`。每个更新访问一个问题，按 seed/epoch 可复现地打乱，保存 actor、AdamW、随机状态、配置、训练清单身份、阅读器身份和累计预算。恢复只允许扩展步数上限/修改保存周期；验证样本、验证周期、阅读器与准确率容差也必须一致，避免将不同验证集的 best 分数混用。更换 K、学习率、奖励底座或训练清单应另开实验。强制终止进程仍可能丢失最近检查点后的状态与计费记录。

| 输出 | 含义 |
|---|---|
| `metrics.jsonl` | 更新损失、效用、采样准确率、条件熵、关系误差、三阶诊断、调用/耗时 |
| `scores.sqlite` | 输入身份确定的冻结阅读器评分与预测缓存 |
| `last.pt` | 周期与正常/预算停止后的原子检查点 |
| `validation.jsonl` | 所选/均匀/全候选帧的周期验证 |
| `best.pt` | 已启用周期验证时，selected 准确率最好的检查点 |
| `summary.json` | 完成步数、停止原因、累计调用与墙钟成本 |

关系库对本步 G 个结构共享；重复掩码去重。验证共用冻结副本与缓存，单独记账，同时计入全局预算。验证没有梯度，不据其更新参数。

~~~bash
# 完整独立评估，默认包括全部 N 候选帧基线；可用 --skip-dense 跳过
python -m framerefine.cli evaluate --config configs/formal.toml --reader-config configs/readers/qwen.toml --checkpoint outputs/formal/best.pt --manifest data/test/manifest.jsonl --output outputs/test.json

# 仅选择已有候选帧，不需要阅读器或正确答案
python -m framerefine.cli select --checkpoint outputs/formal/best.pt --manifest data/test/manifest.jsonl --output outputs/selected.jsonl

# 原视频到 K 帧清单，包含候选/CLIP 准备；可以使用无标签原始 JSONL
python -m framerefine.cli compress --checkpoint outputs/formal/best.pt --raw-manifest data/infer.raw.jsonl --features-config configs/features.toml --cache-dir data/infer-cache --output outputs/infer.selected.jsonl
~~~

若没有启用周期验证，请使用 `last.pt`。选择结果包含原候选索引、按时间排序的 K 个图片路径与秒级时间戳，供下游视频模型读取；缺失选中图片时会报错，并保留既有输出。原视频推理仍需冻结特征提取；`select` 的耗时只是缓存特征后的策略耗时，不能直接充当端到端推理延迟。`compress` 另写 `*.timing.json` 和追加历史 `*.timing.history.jsonl`，记录从准备、模型启动到选帧完成的整条命令成本。

评估保存汇总 JSON 与逐题 `*.predictions.jsonl`。报告 selected、uniform K、dense N 的准确率，并计算 `accuracy_drop_points = 100 × (A_dense − A_selected)`；`within_accuracy_tolerance` 使用默认**3 个绝对百分点**。它只是报告约束，没有把 3pp 写成已达成结果或方法定义。dense 只指当前 N 候选的全帧输入，论文若采用其他高预算基线需另行评估。

## 6. 训练与推理成本

真正节省的对象是送入大阅读器的帧/视觉 token 和训练时冻结阅读器调用。采用缓存冻结特征、只训练小 actor、最后字母打分、共享稀疏关系反馈、掩码去重，以及 GPU 独立副本。InternVL 还复用固定单帧视觉编码；Qwen/LLaVA 当前会重新处理不同掩码的视觉输入，不声称通用视觉 KV 复用。

默认每步 G=8 个任务评分，每 4 步增加一个 8 角关系块，平均上界为 10 次子集评分/步。配置中的周期验证为 selected/uniform/dense 三次：

| 配置 | 更新上限 | 训练评分上界 | 周期验证评分上界 | 总评分上界 | 时长上限 | 预留20%后所需有效总吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| probe | 1,000 | 10,000 | 384（每250步×32题） | 10,384 | 2h | 1.80 次/秒 |
| formal | 40,000 | 400,000 | 7,680（每1000步×64题） | 407,680 | 48h | 2.95 次/秒 |

不传 validation-manifest 时不执行周期验证；cost 命令仍按配置计算带验证的保守上界。缓存命中可减少实际调用，但不用于预先保证预算。

~~~bash
python -m framerefine.cli cost --config configs/probe.toml
python -m framerefine.cli cost --config configs/formal.toml --aggregate-rate 3
~~~

`--aggregate-rate` 只是情景输入，输出明确标为假设值，不是实测性能。按总评分次数/完整训练耗时衡量有效吞吐，不能用单卡理想前向速度或 48GB 显存换算。3 次/秒且预留 20% 时，正式配置约 47.19 小时；若低于 2.95 次/秒，应减少步数/反馈频率或优化实现，不能保证 48h 收敛。

固定结构的计数 DP 上界 O(2^h N K²)，树归一化约 O(2^h N³)。当前 DP 是 Python/PyTorch 数值参考实现，包含大量小操作，并未融合/批量编译。本机 CPU 单线程的两步诊断（hidden64、h2、K8、G8；解析阅读器）N32 约2秒/更新、N64 约4.9秒/更新；只是瓶颈诊断，不是稳定性能基准或服务器预测。N64 的40,000次更新仅策略部分就可能超过48h，首轮采用 N32 更合理。直接把这些小操作移到 GPU 不保证更快。

`Budget` 在每次实际调用前检查请求与墙钟上限，模型/数据加载也计入训练墙钟；正在执行的前向或加载不能强制中断。调用在批次前预留，失败请求也计费。20%余量用于预测，不改变2/48h运行上限。离线候选解码、特征生成、模型下载、独立完整验证、推理时的候选/特征提取需另外记录并计入论文总成本；没有宣称缓存免费。

## 7. 文件指引

~~~text
framerefine/
  annotations.py       原始 MCQ 与三类基准标注转换
  preprocessing.py     帧解码、时间采样、冻结 CLIP、内容缓存
  data.py              CachedExample、懒加载特征清单、编码器身份
  policy.py            条件变量/森林头、全局归一化联合掩码策略
  structures.py        矩阵树概率、Wilson 采样、最大生成树
  inference.py         森林数量约束 DP、概率、熵、采样、MAP
  exchange.py          等数量交换块、二/三阶差分、独立 RB 工具
  readers.py           评分协议、Qwen、缓存、预算、多 GPU 副本
  models/
    common.py          本地快照与字母评分公共边界
    factory.py         reader 配置注册与副本构建
    internvl.py        InternVL2.5 原生 token/视觉缓存适配
    llava_video.py     LLaVA-Video 原生多模态/末位投影适配
  training.py          任务/结构梯度、关系监督、恢复、周期验证
  evaluation.py        selected/uniform/dense 与3pp报告
  cost.py              训练及验证评分上界、吞吐情景
  cli.py               数据、训练、评估、选择全部命令
configs/
  features.toml        本地 CLIP 与候选帧规格
  readers/*.toml       三类本地底座与 GPU 设备
  probe.toml           2小时上限试跑
  formal.toml          48小时上限正式训练
  toy.toml             解析任务软件检查
examples/
  raw.example.jsonl    路径需按实际数据修改
tests/                 数学穷举与软件/接口回归测试
.github/workflows/     CPU CI；不运行真实 GPU 实验
~~~

## 8. 后续科学验证的边界

代码完成不等于论文核心假设成立。主要需要区分以下因素的贡献：

- 无条件森林（h=0）与条件森林：上下文选择是否必要。
- relation_weight=0 与交换监督：增益是否来自有针对性的关系学习。
- 相同 actor 容量、候选数和奖励预算下的一元选择/自回归选择。
- 真实问题变化、背景变化与三阶诊断，是否对应可解释、可迁移的压缩行为。
- 同一阅读器高预算输入相对最多3pp损失时，包含准备、训练与推理的实际成本。
- 读者底座、问题类型、视频长度和压缩率变化下的泛化。

h=0 仍有森林二阶势，并非纯一元基线。关系损失降低、合成测试通过或训练效用升高，都不能单独证明真实视频准确率和投稿创新性。
