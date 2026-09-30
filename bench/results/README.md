# 实测结果文件

门禁 issue 关闭时，`gate-notify` 会从这里读 `gate-NNN.json`（NNN = issue 编号，补零到三位），
和 issue 正文里的 `<!-- gate-spec -->` 块逐条做数值比对。不达标就自动重开。

**评论里手打的数字一律不采信。** 唯一的数据来源是这个目录下的文件。

## 格式

```json
{
  "commit": "672f2ca1c0de4d2b8f5a1e9c3b7d4a6f8e2c1d90",
  "host": "bi100-node3 (4x Iluvatar BI-V100, TP=4)",
  "timestamp": "2026-09-30T11:20:00Z",
  "cmd": "torchrun --nproc_per_node=4 bench/latency_micro.py",
  "metrics": {
    "allreduce_4kb_us": 27.4,
    "latency_p50_variance_pct": 2.4
  },
  "runs": [27.1, 27.4, 27.8],
  "doc": "docs/PROFILE_DECODE_STEP_20260930.md"
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `commit` | 是 | 基准运行时的 commit SHA，必须已推送到本仓库，机器人会校验它存在 |
| `host` | 是 | 跑在哪台机器、什么拓扑 |
| `timestamp` | 是 | ISO 8601。超过 14 天视为过期，必须重跑 |
| `metrics` | 是 | 键名必须和 issue 里 `gate-spec` 声明的 metric 完全一致，值必须是数字 |
| `cmd` | 建议 | 产生这份数据的命令，便于别人复现 |
| `runs` | 建议 | 多次运行的原始值 |
| `doc` | 部分 | 声明里带 `doc` 检查的 gate 必填，指向仓库内一个已存在的文件 |

## 流程

1. 在 4×BI-V100 上跑基准，脚本直接写出 `bench/results/gate-NNN.json`
2. 提交推送到 main
3. 在对应 issue 下评论 `/verified`（或直接关闭 issue）
4. 机器人比对。通过则打 `verified` 标签并 @ 负责人；不通过则重开并贴出比对表

## 已知边界

机器人验的是"这个文件里的数字达标、且能追溯到一个真实 commit"，
**不验"这个数字真的是那台机器跑出来的"**。手写一份 JSON 它分不出来。
要堵这个口子，得在 BI-V100 机器上挂 self-hosted runner，由 CI 自己跑基准、自己写文件。
