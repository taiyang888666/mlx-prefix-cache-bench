# MLX Prefix Cache Benchmark

云清启技术团队 · Yunqingqi Engineering

An independent experiment on SGLang v0.5.23's MLX backend for long, shared-prefix workloads on Apple Silicon. On one M4 Pro with Qwen3-0.6B-4bit, four paired runs with four clients and 12 requests each showed a **10.7% median paired reduction in batch completion time**. P95 time to first visible text went from 160.9 ms to 110.6 ms. One of three multi-turn conversation trials regressed from 2.55 s to 6.51 s. The regression and the request-level results are included.

[完整中文实验报告](docs/report.zh-CN.md) · [后续研究路线](docs/research-roadmap.zh-CN.md) · [逐请求数据](results/) · [补丁](patches/sglang-mlx-prefix-cache.patch)

## 这项实验测了什么

同一台 Apple M4 Pro、同一份模型权重、同一并发波次。四路组使用 896.11 MiB 的 BF16 KV 池和 6,144 MiB 的 MLX 内存分配指导值。每个进程预热同一篇长文档，再以最多四个在途请求完成 12 条分支。切换两种引擎模式时启动新进程，每轮交替先跑哪一端。

逐轮计算耗时降幅，再取中位数。原版与候选版各有 48 条计时请求，回复和输入、输出 token 数逐条一致。分支和对话请求由脚本生成，公开结果覆盖完整的对照轮次，以及改变过一条回复的合并前向试验和连续对话慢轮。该数据覆盖本次硬件、模型和负载。其他模型、CUDA、多卡和生产流量尚待验证。

| 结果 | 数据 |
| --- | --- |
| 四路并发主实验 | [adaptive-parallel.json](results/adaptive-parallel.json) |
| 串行长文档分支与增长上下文 | [adaptive-serial.json](results/adaptive-serial.json) |
| 带真实模型回复的连续对话 | [adaptive-chat-clean.json](results/adaptive-chat-clean.json) |
| 首段变慢的分阶段排查 | [profile.json](results/profile.json)，[数值遥测](telemetry/profile/) |
| 输出变化的合并前向试验 | [group-smoke.json](results/group-smoke.json) |

`telemetry/` 仅保留公开实验需要的计时、缓存池、预算、峰值和缓存重置信息。重启中断及后续补测的来龙去脉见中文实验报告。

## 不运行模型即可核对结果

```sh
python3 verify_results.py
```

此命令从逐请求数据核对配对条件、输出、预算和 10.7% 的计算，并显示连续对话的慢轮。

## 在 Apple Silicon 上复跑

准备 Python 3.12 和 uv，并在仓库根目录执行以下命令。安装依赖和下载权重需要网络。基准运行在本机回环接口。

```sh
mkdir -p .repro
git clone --depth 1 --branch v0.5.23 https://github.com/sgl-project/sglang.git .repro/sglang
test "$(git -C .repro/sglang rev-parse HEAD)" = 297e54fd105e974cdf401e5b4d20253a22c16f90
git -C .repro/sglang apply --check "$PWD/patches/sglang-mlx-prefix-cache.patch"
git -C .repro/sglang apply "$PWD/patches/sglang-mlx-prefix-cache.patch"
cp .repro/sglang/python/pyproject_other.toml .repro/sglang/python/pyproject.toml
SGLANG_BUILD_RUST_EXTS=none uv pip install --target "$PWD/.repro/deps" --python 3.12 \
  "$PWD/.repro/sglang/python[srt_mps]" mlx==0.32.4 mlx-lm==0.32.2
cp .repro/deps/sglang/_version.py .repro/sglang/python/sglang/_version.py
export PYTHONPATH="$PWD/.repro/sglang/python:$PWD/.repro/deps"
python3.12 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "mlx-community/Qwen3-0.6B-4bit",
    revision="73e3e38d981303bc594367cd910ea6eb48349da8",
    local_dir=".repro/model",
)
PY
SGLANG_USE_MLX=1 python3.12 -m unittest -v test_paired_fastpaths
python3.12 paired_bench.py --source .repro/sglang/python --deps .repro/deps --model .repro/model \
  --output .repro/parallel-local.json --rounds 4 --modes stock coalesced \
  --slots 8192 --context 4096 --memory-mib 6144 --scenarios parallel_chat_forks
python3.12 paired_bench.py --source .repro/sglang/python --deps .repro/deps --model .repro/model \
  --output .repro/chat-local.json --rounds 3 --modes stock coalesced \
  --slots 8192 --context 4096 --memory-mib 4096 --scenarios chat_forks chat_history
```

`stock` 关闭实验改动，`coalesced` 启用候选路径。新跑的耗时会随机器和调度状态变化；逐轮结果可与参考数据做对照。使用 `--profile-extend` 单独采集分段计时，正式时延对照关闭此选项。

## 发布候选包验证

补丁已在固定上游提交上应用并通过 123 项 MLX 回归检查。用已有的固定依赖和权重额外跑了一组四路单轮检查，24 条请求的回复和 token 数一致，实际分配峰值均在预算以内。单轮数据只用于检查公开包的运行入口，正式的 10.7% 使用上文的四轮配对结果。[发布包单轮验证数据](results/release-smoke.json)

## 许可与署名

实验脚本及补丁采用 [Apache 2.0](LICENSE)。补丁基于 [SGLang v0.5.23](https://github.com/sgl-project/sglang/tree/v0.5.23)，保留了上游的版权和许可声明，并在被修改的文件中标明变更。附有[上游 AOT 组件第三方声明](docs/sglang-aot-third-party-notices.txt)。模型权重通过[固定版本的模型仓库](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/tree/73e3e38d981303bc594367cd910ea6eb48349da8)获取。
