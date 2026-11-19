# 独立 UAM 走廊环境

仅依赖 Python 3.10+ 标准库，不引用旧训练项目。共享起降位沿用历史 Vertiport5Env 的双端同步预约逻辑；固定机队和响应式回流沿用旧闭环定义。新版重新实现门到门时钟、容量开关及命令行，不包含算法训练、模型或实验矩阵。

## Windows CMD

```cmd
cd /d "C:\Users\w\Documents\Codex\2026-08-05\re\workspace\projects\uam-corridor-env"
conda activate uam5070
set "PYTHONUTF8=1"
python -B corridor_env.py --help
python -B corridor_env.py --geometry
```

默认只打印实际配置、机队分配和需求摘要；加 `--geometry` 只计算 CAV 直达和无排队 UAM，不跑回合或训练。用 `--save-config 文件.json` 保存可重载的配置（不会覆盖已有文件），`--config 文件.json` 加载；CMD 参数优先。配置与需求分别输出 SHA-256。

### 八组基线（K3）

```cmd
python -B -u corridor_env.py --baselines --baseline-output "workspace\artifacts\corridor_k3_base_s123"
python -B -u corridor_env.py --baselines --baseline-output "workspace\artifacts\corridor_k3_tlof1_chargers2_s123" --tlof on --station-set all:pads:1 --finite-charging on --station-set all:chargers:2
```

每条命令运行同一个物理配置下的八组，共用一次生成的需求；切换容量或需求种子时另用结果目录。初始机队按环境设置给四条规则，默认是均分。三个集中组把整支初始机队和乘客**选站**集中到同一个出发站，乘客的真实起终点保持不变。七个航空组都用环境的最长队调机。另有一组 `CAV_DIRECT`，在需求揭示后从真实起点开到真实终点，速度由 `--cav-kmh` 控制，无候车或道路拥堵，也不用机场或飞机。它沿用相同的时间步、截止及完成率口径，即使航空关闭末段，也始终到真实终点才算完成。换 K5 会得到四条规则 + 五个逐站集中组 + 一组 CAV。

- SPF_SRC 保留“较低权重的地面时间 + 估计排队”，不是纯就近。STTF 保留“地面时间 + 估计排队”，QTTI 在此基础上加已路由、仍在接驳途中的乘客拥堵项。旧规则的服务率与 QTTI 分母默认是 3 人/分钟、V0=30/其余站=6，可分别用 `--rule-service-rates` 和 `--qtti-denominators` 逐站覆盖；这些是规则参数，不改变实际服务能力。
- READY 按新版的接驳时间、在途已选站乘客、可用/正在充电飞机和往返时间作近似，不偷看未来请求，也不解资源预约的最优调度。四条规则保留昨天的打分结构，但用新版时间和飞机生命周期，不应当作旧 E1 的逐数值复现。
- 停车位不足以容纳集中机队时，该组为 `INFEASIBLE`，不会私自增加停车位或把飞机放到别站；已运行但截至上限仍未全完成则标为 `CENSORED`，不填写全体 ATT/AWT。完成率与固定截止人时以**已生成**乘客为分母；输出同时列出计划人数。
- `summary.csv` 汇总指标（CAV 车程单列 `cav_drive_min_per_generated`），`manifest.json` 记录配置、需求哈希、代码哈希和规则参数，`arm_*.json` 记录各组分段、选站计数及飞机诊断。中断后同一命令可以继续，已结算组不会重复跑；原七组跑过的目录不覆盖，八组另开目录。避免让两个进程同时写同一个结果目录。

### 地图和物理参数

```cmd
python -B corridor_env.py --destination-shift-km 15 --time-scale 0.5 --tlof on --station-set V1:pads:2 --turnaround on --finite-stands on --station-set all:turnaround_min:3 --finite-charging on --station-set V1:chargers:4 --geometry
```

此例把目的站及终点区向东移 15 km；相对于同地图、同种子、原速率的设置，到达间隔减半时 OD 与顺序保持不变。固定 CSV 列为 `id,time_min,origin_x,origin_y,dest_x,dest_y`（也接受 `time` 作为分钟列），可用 `--time-scale` 缩放时间，但不会自动改写其 OD。

| 可调内容 | 命令行 |
|---|---|
| 全部机场及角色 / 单站坐标 | 重复 `--station V0:departure:0:-8` 等来替换地图；`--station-position V1:-2:0` 只改一站 |
| 起点与终点刷新区域 | `--origin-zone=-5,4,-12,12,1,uniform`、`--destination-zone=47,53,-4,4,1,uniform`，可重复；最后两个字段为权重与分布，支持 uniform / centered |
| 机队与单机服务 | `--fleet 48 --seats 1 --boarding immediate`；集中布局 `--initial-fleet V1:48` |
| 需求 | `--passengers 300 --arrival-min 1 --seed 123 --arrival-mode uniform`；也支持 varying、poisson 或固定 CSV |
| 物理开关 | `--tlof`、`--operation-time`、`--turnaround`、`--finite-stands`、`--finite-charging`、`--parking`、`--energy`、`--egress` 均接受 on/off |
| 按站或全网覆盖 | `--station-set V1:字段:值` / `--station-set all:字段:值` |
| 调机 | `--reposition off/fixed/lq/manual`；固定回流可指定 `--fixed-returns V2:V1`；最长队和手动模式支持 `--lateral on` |
| 时间、速度、比例尺 | `--tick-min`、`--decision-min`、`--cutoff-min`、`--access-kmh`、`--egress-kmh`、`--air-kmh`、`--cav-kmh`、`--km-per-unit` |

机场字段：`pads`、`takeoff_min`、`landing_min`、`separation_min`、`stands`、`turnaround_min`、`chargers`、`charge_soc_per_min`（0–1 SOC/分钟）、`parking`。能耗可用 `--initial-soc`、`--reserve-soc`、`--target-soc`、`--fixed-flight-soc` 和 `--soc-per-km`。

默认是待校准配置：三个出发站、一个到达站，48 架均分在出发站，单客、不凑批。末段、能量和起降耗时开启；TLOF 容量、周转、有限充电位、停车容量默认关闭。开启周转并不会自动加有限周转位；关闭有限充电位不会关闭能耗。关闭 TLOF 容量后不恢复旧的一站一分钟一架限制。TLOF 启用时起降占用同一组 Pad，间隔在占用结束后另计；停车位在承诺航班时保守预留，没有地面位就暂不派发，不模拟无限盘旋。

## 环境接口

- `parse_config(argv)` 提供同一套 CMD 参数解析，供后续独立入口使用。
- `CorridorEnv(config).reset(requests=None)` 返回已揭示需求、已选站接驳人数、站点队列、飞机状态及资源占用，可传入固定 `Request` 列表。
- `step(routes={乘客ID:出发站}, reposition={飞机ID:目标站})` 按决策间隔推进事件，返回 observation、reward、terminated、truncated、info。手动调机模式才接受飞机动作；省略乘客动作表示请求继续等待。固定回流未指定目标时使用排序后的第一个出发站。
- `metrics()` 将等待拆成待选站、排队和待起飞，再独立记录首程、起飞、巡航、落地、末程。奖励为负的已生成未完成人时；只有计划请求全部完成且启用末段，才定义门到门 ATT。关闭末段的完成成绩单列到落地时间。

验证代码：`python -B -m unittest -v test_corridor_env`。只测环境不变量。
