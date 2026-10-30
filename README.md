# Qwen3.8-27B 情绪故事生成复现包

独立于原毕设目录的 Linux/NVIDIA GPU 运行包。目标是 **171 种情绪 × 100 个共同主题 × 每组12篇 = 205,200篇**，每次请求只写一篇。在一台服务器上执行全部任务，可用单卡或明确指定多张卡分担同一个模型。无需学校 API Key。

**当前版本取消固定512/2048输出token上限**，以 EOS 自然结束。Transformers 的生成循环仍需有限边界，因此代码显式使用“模型上下文容量262144减去当前输入长度”作为容量保护，避免默认20token截断。到达模型容量但未EOS的文本标记未完成，原样保存；不能无限突破模型/显存容量。prompt原有150–350token是软建议，不按此裁剪。长度不受固定限制也意味着异常冗长文本可能增加耗时或导致OOM，代码会缩批并如实记录。

这里有运行代码、固定依赖、171标签/100主题、完整prompt、模型配置/许可与逐文件SHA256清单。**没有打包52GiB模型权重或现有venv二进制**：模型在目标服务器按固定版本下载，环境按依赖清单重建；不能把CUDA驱动、旧绝对路径和动态链接环境当作可移植GitHub源码。模型资产遵循 `model-assets/LICENSE`；本包未另行指定项目代码的开源许可证。

## 1. 资源与版本

- 建议 Linux x86_64、Python3.10、NVIDIA A800/A100 80GB 或相当设备。
- BF16，关闭thinking，SDPA，最大并发16，自适应2→4→8→16；OOM缩批。不量化、不CPU/disk offload、不自动使用全部GPU。单卡用`--gpu N`；同服务器多卡用`--gpus 1,2`，Accelerate按层balanced放置同一个模型，不是多个模型副本或跨服务器分片。多卡支持尚未做GPU端到端验证，需要目标机先跑小样本，不能承诺更快。
- 单卡启动前空闲显存门槛 **62GiB**；多卡要求每卡至少6GiB空闲、每卡预留2GiB后合计至少62GiB可用预算。预算是保守准入检查，不保证所有长文本都不OOM。单张24GB卡放不下BF16模型，多张卡是否可用取决于总可用显存及实际分配。脚本要求计算模式Default，这允许共享，**不保证独占**。先核查他人进程和本机使用约定。
- 工作目录放大容量数据盘，建议先准备至少100GiB空闲空间（模型约52GiB，另有环境/缓存/输出）；下载脚本和worker还会检查剩余空间。
- Torch2.7.1+cu126、Transformers5.3.0，其余见requirements.txt。需要兼容CUDA12.6运行时的宿主NVIDIA驱动；驱动不在venv内。不要在其他正在运行的实验环境中安装。
- 未启用flash-linear-attention/causal-conv1d/torch.compile/vLLM；不要自行替换推理后端，否则属于新实验。

## 2. 下载仓库，创建独立环境

将本目录内容作为新GitHub仓库根目录上传。不要上传外层毕设仓库或其他runs。

```bash
# clone你自己的仓库后进入其根目录，仓库建议放数据盘
cd /path/to/qwen38-story-runner
df -h .
nvidia-smi
bash scripts/setup.sh
```

setup只在本目录创建`.venv`，不会修改系统环境。默认使用python3.10，也可 `PYTHON_BIN=/path/to/python3.10 bash scripts/setup.sh`。requirements固定核心依赖；安装时会下载平台适用的CUDA依赖，实际完整版本保存在environment-installed.freeze.txt。若安装/导入失败，先记录错误和实际版本，不要擅自升级Torch/Transformers后继续同一个实验。

## 3. 下载/校验固定模型

```bash
export QWEN_MODEL_DIR="/data/$USER/models/Qwen3.8-27B"
.venv/bin/python scripts/model_files.py --model-dir "$QWEN_MODEL_DIR" --download --full-hash
```

固定来源：`Qwen/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。下载由huggingface_hub完成，可在连接中断后重试同一命令；不自动换分支、镜像模型或latest版本。网络需按目标服务器实际条件配置；不要把代理、令牌、私钥写进仓库。若网络不通，可以从已有服务器单独rsync整个模型目录，然后运行上面的命令去掉`--download`进行全量哈希校验。18个权重分片及tokenizer必须齐全。

离线传输环境可在相同平台联网机器运行 `.venv/bin/python -m pip download -r requirements.txt -d wheelhouse`，将wheelhouse另行传输，目标venv用 `pip install --no-index --find-links wheelhouse -r requirements.txt` 安装。此wheelhouse不包含操作系统/驱动，不能跨平台保证兼容，也不上传GitHub。

## 4. CPU预检查与小样本

```bash
.venv/bin/python scripts/preflight.py --model-dir "$QWEN_MODEL_DIR" --check-gpu 0
.venv/bin/python scripts/run_full.py --model-dir "$QWEN_MODEL_DIR" --output runs/full-001 --gpu 0 --plan
```

preflight验证文件大小和非权重SHA、依赖、非思考对话模板、标签数、主题数、无固定token上限；仅查询GPU，不加载权重。全权重SHA使用上一步`--full-hash`。`--plan`只打印任务计划，不创建run、下载模型或占GPU。

用户授权实际运行后，先在选定GPU做一次108篇小样本，按目标情绪检查：

```bash
.venv/bin/python -u scripts/generate_qwen38_pilot.py \
  --model-dir "$QWEN_MODEL_DIR" --gpu 0 --batch-size 16 --auto-batch \
  --wait-for-gpu --config configs/pilot.json --output runs/pilot-001
```

当前小样本是angry/happy/sad ×3共同主题×12篇，已取消固定token上限，**不是原512上限的逐字复现**。模型、prompt和其余采样参数延续已检查的27B版本，代码移植后的无固定上限GPU生成尚需在目标机实测。

## 5. 单服务器完整生成（单卡或指定多卡）

输出目录必须是新的；不要预先mkdir这个run目录。日志放其外部：

```bash
mkdir -p runs
nohup .venv/bin/python -B -u scripts/run_full.py \
  --model-dir "$QWEN_MODEL_DIR" --gpu 0 --batch-size 16 \
  --output runs/full-001 > runs/full-001.supervisor.log 2>&1 &
echo $! > runs/full-001.pid
```

关闭SSH不影响nohup进程。使用run.json及进程状态确认启动，不能只凭nohup返回判断运行成功。指定卡不满足相应显存预算时worker每60秒等待，不抢占或终止其他进程。

```bash
.venv/bin/python -c 'import json; print(json.dumps(json.load(open("runs/full-001/run.json")),ensure_ascii=False,indent=2))'
tail -n 20 runs/full-001.supervisor.log
# 细节日志：runs/full-001/logs/emotion-NNN.log
```

### 同服务器多卡用法

先查询各卡资源和拓扑，选择本实验允许使用的卡；例如物理GPU1和2：

```bash
nvidia-smi
nvidia-smi topo -m
.venv/bin/python scripts/preflight.py --model-dir "$QWEN_MODEL_DIR" --check-gpus 1,2
.venv/bin/python -u scripts/generate_qwen38_pilot.py --model-dir "$QWEN_MODEL_DIR" \
  --gpus 1,2 --batch-size 16 --auto-batch --wait-for-gpu \
  --config configs/pilot.json --output runs/pilot-multigpu-001
# 小样本确认正常后，全量命令用同一组卡
.venv/bin/python -B -u scripts/run_full.py --model-dir "$QWEN_MODEL_DIR" \
  --gpus 1,2 --batch-size 16 --output runs/full-multigpu-001
```

`--gpu`和`--gpus`互斥。多卡全量需后台运行时同样使用nohup并将日志放run目录外。脚本用指定卡UUID限制CUDA_VISIBLE_DEVICES，记录实际device_map；若权重落到CPU/disk会报错，不静默offload。所有卡仍需符合本服务器资源约定；显存分担不等于多卡线性提速，PCIe/NVLink和共享负载影响性能。GPU端多卡效果未实测。本地时间统计会同步所选全部卡，peak显存为各卡峰值之和。

## 6. 停止与续跑

用 `kill -TERM <监督进程PID>` 或前台Ctrl-C停止本实验监督进程；它会停止自己的子进程。只操作本任务记录的PID，不用pkill杀其他任务。当前尚未落盘批次可能丢失，已落盘故事保留。

确认旧监督/worker都退出后，使用**完全相同命令加`--resume`**；原命令是`--gpus 1,2`就必须保留它，不换成`--gpu`。例如：

```bash
.venv/bin/python -B -u scripts/run_full.py --model-dir "$QWEN_MODEL_DIR" \
  --gpu 0 --batch-size 16 --output runs/full-001 --resume
```

源码、配置、依赖、GPU列表及顺序、模型路径要保持一致，文件锁防止同run重复运行。只修复最后一条没有换行的未完成写入，不清理有效正文。一次实验开始后不要更新代码或依赖；已有run的bootstrap含绝对路径，**中断任务迁到另一目录/服务器后不保证能直接resume**，本包支持的是其他服务器新开独立实验。

## 7. 实验定义、记录和限制

- Prompt保持恢复版：每请求一篇，人物事件后至结尾保持目标情绪，禁止目标词和直接同义词；虽然实际仍可能出现显性词，全部原样保留，不自动重写/剔除。软长度150–350不用于裁剪。
- Sampling：temperature0.7、top_p0.8、top_k20、repetition_penalty1.05、do_sample=True、use_cache=True。system放prompt，user为空串，tokenizer原生模板且enable_thinking=False。显式校验模板以`</think>`结束。
- 全局ID=`emotion_id*1200+topic_id*12+sample_id-1`；seed=`int(SHA256("42:ID")[:8],16)`。**实际每批RNG用该批第一条seed**，每条seed不等于独立随机流；batch组成/OOM缩批/硬件差异会改变输出，不保证跨机器逐字相同。32位派生种子可能重合，身份以request_id为准。
- 每种情绪单独1200篇，模型顺序重载。run保存Git提交号/dirty、依赖、配置/源码哈希、模型版本、模板、种子、批次历史、GPU、耗时、token/EOS等。没有Git提交时commit为null，不伪造版本号。
- 小样本原512上限27B稳定16并发约19.4篇/分钟，全量外推7.3天，实际曾建议8–10天；**取消上限后的总耗时尚未测定**，不能沿用这个保证作为无固定上限版本的估时。
- 本包没有激活提取、steering或PCA任务，不修改原backbone权重。情绪审查仅作实验外部工作；本程序输出不是自动认证的205200篇合格文本。
- 论文来源：*Emotion Concepts and Their Function in a Large Language Model* §6.5，p.60的emotional stories prompt；原论文多故事封装改为单篇、补充事件/结尾持续等是工程修改，150–350建议和采样参数也不是论文原样。只附方法说明，不打包论文正文。

给其他AI：首先阅读本文件和AGENTS.md，再运行CPU检查。不要把原服务器的已运行9B任务当成这个仓库的一部分。
