# **UAM 联合调度与异步多智能体强化学习研究 README**

**Research on Joint UAM Scheduling and Asynchronous Multi-Agent Reinforcement Learning**

## **一、研究背景与交通环境**

本研究围绕城市空中交通（Urban Air Mobility, UAM）中的乘客出发站选择与航空器重定位联合调度问题展开，核心是将乘客侧的出发站分配决策与航空器侧的跨站重定位决策纳入同一个动态决策框架，研究两类决策之间的相互影响及其协同优化机制。乘客选择不同出发站会改变各站点未来的需求分布，而航空器重定位则直接影响不同站点的可用运力，两者共同决定乘客等待时间与系统整体服务效率。因此，本研究将乘客分配和飞机调度置于统一的交通状态、决策过程和奖励目标下进行联合建模，以固定机队和有限交通资源为约束，最小化乘客从出发地到最终目的地的门到门平均出行时间（Average Door-to-Door Travel Time, ATT）。项目以 UAGMC（Unified Air-Ground Mobility Coordination）为重要研究基础，完成其原有联合调度框架的完整复现，并在此基础上建立面向多智能体强化学习的统一实验框架，重点研究不同决策主体如何通过策略交互实现乘客需求与航空器服务能力的动态匹配。

当前 E2 主实验采用三个出发站 V0、V1、V3 和一个目的站 V2，配置固定规模的 eVTOL 机队，并通过调整乘客需求压力构建 1.0×、1.1×、1.2×、1.3×、1.4× 五档实验环境。系统采用分钟级策略决策与更细粒度物理状态推进相结合的运行方式。在每个决策周期内，Passenger Agent 根据当前需求和交通状态选择乘客出发站，Aircraft Agent 则根据航空器位置、可用状态及各站点运力需求决定重定位行为。两类决策共同改变后续系统状态：乘客分配影响各站点的待服务需求，航空器重定位改变未来可提供的运输能力，而航空器的飞行、周转和机场资源占用又进一步影响后续乘客的等待与服务过程。由此形成乘客分配与飞机重定位相互耦合的闭环调度系统。最终评价统一考虑乘客地面接驳、出发站等待、航空器服务、空中飞行及最终离站过程，以完整门到门 ATT 衡量联合调度策略的整体性能。

## **二、人工智能算法与核心方法**

**1. MAPPO（Multi-Agent Proximal Policy Optimization）。**

MAPPO 是本研究的基础多智能体强化学习算法，采用集中训练、分散执行（Centralized Training with Decentralized Execution, CTDE）框架，通过集中式 Critic 利用全局状态信息估计团队价值，并使用 PPO 的裁剪策略目标更新 Actor。其优势在于训练机制相对稳定、实现简洁，适合作为 Passenger 与 Aircraft 联合调度任务的基础算法。我们已完成 MAPPO 在 UAM 环境中的迁移与训练，并将其作为后续 ACAC 机制适配和异步 MARL 方法研究的主要对照基线。

论文：Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022。

[论文原文（NeurIPS 2022）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Conference.html?utm_source=chatgpt.com)

**2. HAPPO（Heterogeneous-Agent Proximal Policy Optimization）。**

HAPPO 面向异构智能体协作问题，其核心是 Multi-Agent Advantage Decomposition（多智能体优势分解）与序贯策略更新机制，通过依次更新不同 Agent 的策略，处理异构主体之间的策略依赖关系。与普通 MAPPO 相比，HAPPO 更强调不同智能体策略更新顺序对联合策略优化的影响。我们已将 HAPPO 迁移至 UAM 联合调度环境，用于研究 Passenger 与 Aircraft 在不同决策职责和动作空间下的协作学习表现。当前实验发现，HAPPO 在低需求压力下与 MAPPO 表现接近，但随着需求压力增加，其性能退化更加明显，在 1.3× 和 1.4× 下分别比 MAPPO 慢 3.58 和 5.37 分钟。这一现象为进一步研究异构 Agent 的决策频率、策略更新及协同机制提供了实验依据。

论文：Kuba et al., *Heterogeneous-Agent Reinforcement Learning*, Journal of Machine Learning Research, 2024。

[论文 PDF（JMLR 2024）](https://www.jmlr.org/papers/volume25/23-0488/23-0488.pdf?utm_source=chatgpt.com)

**3. ACAC（Agent-Centric Actor-Critic）。**

ACAC 是本研究目前最重要的异步多智能体强化学习方法。该算法针对不同 Agent 宏动作持续时间不同所引起的异步决策问题，提出 Agent-Centric History Encoder，分别维护和编码不同智能体的历史轨迹，并通过 Attention-based Centralized Critic 融合各 Agent 的历史信息，减少传统 Padding 方式可能引入的异步轨迹错位。同时，算法设计了适用于异步决策过程的 Macro-GAE，使价值估计与不同智能体的决策时间结构相匹配。其核心价值在于将不同 Agent 的独立历史建模与集中式价值评估结合起来，为异步 MARL 提供了一套 Actor-Critic 学习框架。

论文：Jung et al., *Agent-Centric Actor-Critic for Asynchronous Multi-Agent Reinforcement Learning*, ICML 2025。

[论文原文（ICML 2025）](https://proceedings.mlr.press/v267/jung25a.html?utm_source=chatgpt.com) · [PDF 下载](https://raw.githubusercontent.com/mlresearch/v267/main/assets/jung25a/jung25a.pdf?utm_source=chatgpt.com)

## **三、目前正在推进的研究方向**

### **1. 面向决策机会不对称与不规则性的多智能体强化学习**

目前主要研究不同 Agent 在决策次数、决策间隔和决策顺序上存在差异时，如何提高多智能体强化学习的学习效率。现有 UAGMC 等分钟级交通调度框架及常规 PPO 类方法，通常按照统一物理时间步组织各智能体的策略输出，难以充分处理同一分钟内连续发生的多个有效决策；而对于连续多个物理时间步没有合法决策机会的 Agent，传统同步数据组织方式又容易通过 Padding 或无效动作进行时间对齐。ACAC 通过 Agent-Centric History、时间戳和 Macro-GAE 等机制，一定程度上解决了异步轨迹的历史组织和时间对齐问题，但对于瞬时决策场景中的多次连续决策，以及不同决策频率下的历史组织和信用分配，仍有进一步研究空间。

因此，我们希望从决策机会本身出发，研究不同决策密度下的历史表示、策略更新、价值估计和团队信用分配问题。按照决策机会的不同特征，将研究分为高频决策、低频决策和混合异步三个层次。

**（1）高频决策（High-Frequency Decision Making）**

研究同一个 Agent 在一个物理时间步内连续获得多个有效决策机会的情况。例如，同一分钟内有多个乘客需要选择出发站，此时需要解决多个决策对象的历史组织、连续决策之间的状态依赖以及各动作对未来团队收益的贡献问题。

- **物理历史与决策历史分解（Dual-stream History Representation）：** 将系统物理状态的变化与同一时间步内多个决策对象的信息分开组织。前者记录机场资源、飞机状态和系统需求等随物理时间变化的信息，后者记录当前决策对象及此前已经处理的对象和动作，避免不同乘客的决策信息与物理时间历史发生混淆。
- **步内连续决策的序贯分配（Intra-step Sequential Decision）：** 借鉴 HAPPO 的序贯优势分解思想，研究同一 Agent 连续处理多个决策对象时，如何依次生成决策并更新后续决策的条件信息，使前一个动作对系统状态产生的影响能够被后续决策利用。
- **步内连续决策的信用分配（Intra-step Credit Assignment）：** 研究同一物理时间内多个动作共同影响未来团队奖励时，如何估计每个原子决策的实际贡献，避免将多个动作简单赋予相同的整体优势。

此外，将决策对象的顺序与排列不变性作为贯穿上述研究的约束，研究不同乘客处理顺序是否会造成明显的策略性能差异，避免模型学习到缺乏实际业务依据的对象 ID 优先级。

**（2）低频决策（Low-Frequency Decision Making）**

研究 Agent 经过多个物理时间步后才获得下一次合法决策机会的情况。例如，飞机在飞行、周转或等待期间无法立即进行新的重定位决策，但系统中的乘客需求和其他 Agent 的行为仍然在持续变化。重点研究两次有效决策之间的历史信息如何更新，以及如何进行跨时间的价值估计。

- **决策间隔中的物理历史演化（Time-aware History Evolution）：** 研究 GRU 等历史编码器如何处理两次决策之间的时间间隔，以及期间发生的物理状态变化。不仅考虑决策间隔 $\Delta t$，还考虑期间的状态变化 $\Delta X^{phys}$，避免仅根据相邻两次决策观测推断系统演化。
- **决策间隔中的队友行为（Cross-agent Event History）：** 记录一个 Agent 暂时没有决策机会时，其他 Agent 在此期间执行的动作、发生时间及其造成的状态变化，使低频 Agent 在重新获得决策机会时能够利用相关的队友历史信息。训练阶段可以利用集中式信息辅助价值估计，但不能让分散执行的 Actor 使用实际不可获得的信息。
- **异步信用分配（Multi-clock Credit Assignment）：** 研究 ACAC 原有 Global Macro-GAE 是否适合不同决策频率的 Agent，并比较 Global Macro-GAE、Agent-centric GAE 和 Dual-clock GAE。重点分析物理时间折扣因子 $\gamma$ 与优势传播参数 $\lambda$ 是否应该采用不同的时间尺度，研究全局事件数量和 Agent 自身决策次数对长期信用传播的影响。

在上述研究中，还需要明确区分三种情况：没有获得合法决策机会、获得机会但选择不行动、获得机会并执行有效动作。进一步研究决策机会是由外部事件产生，还是受到其他 Agent 行为影响，避免将三种情况统一处理为无效动作或 Padding。

**（3）混合异步（Mixed Asynchronous Decision Making）**

在高频和低频决策同时存在的情况下，研究不同 Agent 的决策机会、历史信息和团队收益之间的相互影响。例如，乘客选站 Agent 可能在同一分钟内连续处理多个乘客，而飞机重定位 Agent 可能经过数分钟才获得一次新的合法决策机会。后续将结合前两个方向的研究成果，重点推进：

- **跨角色历史融合：** 统一组织不同 Agent 的物理历史、决策历史及其间发生的队友事件。
- **多时钟价值估计：** 同时考虑物理时间、Agent 自身决策时间和跨角色事件，建立适应不同决策频率的价值估计机制。
- **统一信用分配：** 将同一物理时间内的连续决策与跨多个物理时间步的稀疏决策纳入统一的团队收益分配框架。

最终希望形成面向决策机会不对称与不规则性的统一异步 Actor-Critic 方法，使不同 Agent 在决策次数、时间间隔和决策顺序存在差异时，仍然能够有效组织历史信息、估计团队价值并完成信用分配。

### **2. 面向地空协同的 CAV–UAM 综合交通环境**

在原有 UAM 环境基础上，我们正在引入自动驾驶车辆（Connected and Automated Vehicles, CAV）地面接驳系统，将原来的乘客选站与飞机重定位联合调度扩展为地面接驳、乘客分配和航空器调度三个主体协同的综合交通系统。CAV 速度设置为 40 km/h，eVTOL 飞行速度为 200 km/h，两类交通工具均采用单座服务，不引入拼车，并设置有限车队规模。系统主要包含三个决策 Agent：

- **乘客选站 Agent（Passenger Selection Agent）：** 乘客产生出行请求后，由该 Agent 根据各出发站的需求和交通状态选择乘客的出发站，决定后续地面接驳目的地。
- **CAV 接客调度 Agent（CAV Dispatch Agent）：** 负责地面车辆派单。CAV 完成上一项任务并重新空闲后，从待服务乘客队列中选择乘客，安排车辆空驶接客，再将乘客送往其选定的出发站。车辆的任务完成时间及乘客抵达时间由车辆当前位置、任务队列和行驶距离共同决定。
- **飞机重定位 Agent（Aircraft Repositioning Agent）：** 根据各出发站的航空器分布、乘客需求和飞机可用状态，决定空闲飞机是否需要跨站重定位，使航空运力能够适应不同站点的需求变化，同时满足飞机周转、起降资源及充电等约束。

三个 Agent 通过乘客需求、车辆接驳和航空器运力形成闭环协作：乘客首先选择出发站，CAV 负责将其接驳至相应站点，乘客抵达后进入候机队列，航空器在具备服务条件时完成运输，而飞机重定位则持续调整各站点的未来可用运力。

整个系统以乘客门到门平均出行时间（ATT）为统一优化目标：

```math
\mathrm{ATT} = \frac{1}{N}\sum_{p=1}^{N} T_p
```

其中，每名乘客的完整出行时间包括：

```math
\begin{aligned}
T_p ={}& T_{\mathrm{pickup\ wait}} + T_{\mathrm{CAV\ empty}} \\
       &+ T_{\mathrm{CAV\ loaded}} + T_{\mathrm{vertiport\ wait}} \\
       &+ T_{\mathrm{flight}} + T_{\mathrm{egress}}.
\end{aligned}
```

六项依次为：接驳等待、CAV 空驶接客、CAV 载客行驶、机场等待、飞行和最终离站。

## **四、实验基线、方法对照与阶段性成果**

**1. UAGMC 交通环境与调度机制复现。**

已完成 UAGMC 原始交通环境、核心调度流程及主要算法机制的完整复现，并在此基础上建立当前统一的 UAM 联合调度实验框架。通过对乘客出发站选择、航空器服务和调度流程进行复现，形成了可用于后续算法迁移、机制对照和独立方法研究的实验基础。UAGMC 同时作为重要的原始对照工作保留，后续研究将进一步围绕决策机会结构、历史表示及异步信用分配展开，而不局限于对原始调度策略的简单改进。

**2. MAPPO 与 HAPPO 的环境迁移。**

已完成 MAPPO 和 HAPPO 在 UAM 联合调度环境中的迁移与训练，建立四训练种子的学习基线，并统一采用门到门 ATT 作为主要评价指标。相关实验覆盖五档需求压力，能够比较不同 MARL 更新机制在需求增长情况下的性能变化。其中 MAPPO 作为基础 CTDE 对照，HAPPO 则用于验证异构 Agent 序贯更新和优势分解机制在 UAM 场景中的适用性。

实验结果显示，两种算法在低需求压力下表现接近，但 HAPPO 随需求压力上升出现了更明显的性能退化：

- **低压力环境（1.0×–1.1×）：** HAPPO 与 MAPPO 基本持平，ATT 分别为 38.60 和 50.65 分钟。
- **中高压力环境（1.2×–1.3×）：** HAPPO 开始明显落后于 MAPPO，1.3× 时 ATT 达到 77.95 分钟，比 MAPPO 高 3.58 分钟。
- **高压力环境（1.4×）：** HAPPO 的 ATT 达到 89.09 分钟，比 MAPPO 高 5.37 分钟。

这一结果表明，HAPPO 的异构 Agent 序贯更新机制并未在当前 UAM 任务中带来预期的高压力性能优势，反而表现出更明显的压力敏感性，值得进一步从策略更新、决策机会和跨角色协同等方面分析。

**3. ACAC 算法的 UAM 适配版与未适配版实现及实验验证。**

在完成 MAPPO 和 HAPPO 环境迁移后，我们进一步围绕 ACAC 开展算法研究，分别实现了面向 UAM 决策特点进行机制适配的 ACAC-UAM 适配版，以及直接采用 ACAC 原有算法机制、不进行 UAM 专门适配的 ACAC 未适配版。两种实现分别对应任务适配与原始机制直接应用两条路径，均完成独立训练和五档需求压力评估。

- **ACAC-UAM 适配版：** 以 MAPPO 为基础，将 ACAC 的核心机制融入 UAM 原有分钟级决策接口，引入 Agent-Centric History、时间编码、注意力历史融合及异步信用分配机制。完成 Full、NoTE、NoSA 和 Micro-GAE 四臂消融实验，每种方法均使用 2101–2104 四个训练种子，并完成五档需求压力回放。
- **ACAC 未适配版：** 直接采用 ACAC 原有算法机制，不针对 UAM 的分钟级决策接口及乘客、飞机的具体决策特点进行额外适配，保留 Agent-Centric History、时间编码、Attention-based Critic 和 Macro-GAE 等核心设计。完成独立的四种子训练，并在相同 UAM 环境下进行五档需求压力评估。
- **两种实现的对照目的：** 通过将两种实现同时纳入 E2 主对照，研究 ACAC 原始机制直接应用与面向具体交通任务进行机制适配之间的性能差异，分析不同算法组织方式在 UAM 联合调度任务中的适用性。

**4. E2 主对照实验：五个规则基线与五个学习方法。**

下表报告不同需求压力下的门到门 ATT，单位为分钟，数值越低越好。学习方法的 `±` 为训练种子间的样本标准差；规则基线为同档四条需求轨迹的均值。

| 方法                             | 1.0×             | 1.1×             | 1.2×             | 1.3×             | 1.4×             |
| ------------------------------ | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| SPF_SRC                        | 71.52            | 93.54            | 95.87            | 101.14           | 119.95           |
| STTF                           | 86.95            | 91.88            | 104.15           | 113.55           | 121.68           |
| QTTI                           | 72.74            | 85.59            | 99.98            | 107.42           | 112.31           |
| READY                          | 53.07            | 74.35            | 91.14            | 117.18           | 124.64           |
| BOTH_V0（池化）                    | 81.22            | 93.08            | 102.06           | 109.34           | 118.98           |
| HAPPO（4 seeds）                 | 38.60 ± 0.51     | 50.65 ± 0.36     | 65.08 ± 2.72     | 77.95 ± 2.40     | 89.09 ± 4.46     |
| MAPPO（4 seeds）                 | 38.52 ± 0.55     | 50.69 ± 0.92     | 63.28 ± 1.22     | 74.37 ± 2.09     | 83.72 ± 2.33     |
| UAGMC_AIR（含调机，2 seeds）         | 54.36 ± 5.70     | 64.97 ± 5.47     | 79.22 ± 0.66     | 89.85 ± 3.15     | 103.49 ± 5.92    |
| **ACAC-UAM 适配版 Full（4 seeds）** | **38.13 ± 0.56** | **50.45 ± 0.91** | **62.74 ± 1.39** | **72.69 ± 1.84** | **81.37 ± 2.70** |
| ACAC 未适配版 Full（4 seeds）        | 40.14 ± 1.28     | 52.33 ± 1.06     | 63.47 ± 1.38     | 73.50 ± 0.62     | 83.34 ± 2.85     |

主要实验发现：

- **ACAC-UAM 适配版的整体表现：** Full 在五档需求压力下均为表内表现最好的学习方法，相较普通 MAPPO，其 ATT 分别改善 0.39、0.24、0.54、1.68 和 2.35 分钟，改善幅度在高压力环境下更加明显。
- **适配版与未适配版的差异：** ACAC-UAM 适配版在五档压力下均优于 ACAC 未适配版，分别改善 2.01、1.88、0.73、0.81 和 1.97 分钟。这一结果初步说明，针对具体交通任务进行机制适配具有实际价值，但两种实现涉及多个算法组织差异，不能仅凭端到端成绩将性能差异归因于某个单独组件。
- **HAPPO 的高压退化：** HAPPO 在 1.3× 和 1.4× 下分别比 MAPPO 慢 3.58 和 5.37 分钟，说明其在当前环境中表现出更明显的高需求压力敏感性。
- **规则基线的压力变化：** READY 在 1.0×、1.1× 和 1.2× 下表现最好；1.3× 时 SPF_SRC 最好；1.4× 时 QTTI 最好。不同规则策略在需求增长过程中表现出不同的适用范围。

**5. ACAC-UAM 适配版的四臂消融实验。**

下表单独呈现 ACAC-UAM 适配版的 Full 与三个消融方法，并保留普通 MAPPO 作为参照。所有方法均使用 2101–2104 四个训练种子，五档需求压力下均完成全部乘客服务。这里不加入 ACAC 未适配版的结果，以保证消融实验对应同一套算法实现。

| 方法          | 1.0×             | 1.1×             | 1.2×             | 1.3×             | 1.4×             |
| ----------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 普通 MAPPO    | 38.52 ± 0.55     | 50.69 ± 0.92     | 63.28 ± 1.22     | 74.37 ± 2.09     | 83.72 ± 2.33     |
| **Full**    | 38.13 ± 0.56     | 50.45 ± 0.91     | **62.74 ± 1.39** | **72.69 ± 1.84** | **81.37 ± 2.70** |
| NoTE（去时间编码） | **37.96 ± 0.38** | **50.02 ± 0.81** | 63.82 ± 2.34     | 78.43 ± 5.19     | 89.46 ± 7.81     |
| NoSA        | 38.58 ± 1.27     | 50.59 ± 0.78     | 63.22 ± 2.14     | 74.69 ± 4.02     | 85.74 ± 5.66     |
| Micro-GAE   | 40.79 ± 0.98     | 52.37 ± 0.31     | 65.70 ± 1.40     | 76.90 ± 1.72     | 86.77 ± 1.85     |

主要消融发现：

- **Full 的高压优势：** ACAC-UAM 适配版 Full 在 1.2×、1.3× 和 1.4× 下均优于普通 MAPPO，其中 1.3× 和 1.4× 分别改善 1.68 和 2.35 分钟，说明当前完整机制在较高需求压力下具有更明显的性能优势。
- **时间编码（NoTE）：** 去除时间编码后，算法在 1.0× 和 1.1× 下反而略有改善，但在 1.2×、1.3× 和 1.4× 下明显退化。尤其在 1.4× 下，NoTE 比 Full 慢 8.09 分钟。这说明时间编码的价值具有需求压力依赖性，在高压力环境下更加重要。
- **注意力融合（NoSA）：** 去除注意力历史融合后，低压力下性能变化较小，但在高压力下退化更加明显。1.4× 下 NoSA 比 Full 慢 4.37 分钟，说明历史信息融合机制在当前高压力设置中具有一定作用。
- **异步信用分配（Micro-GAE）：** Micro-GAE 在五档需求压力下均落后于 Full，1.4× 下差距达到 5.40 分钟。这表明当前完整信用估计机制在该实验设置中具有性能价值，值得进一步研究其与不同决策频率、历史表示之间的关系。

整体来看，四臂消融结果表明，时间编码、注意力历史融合和信用估计机制的作用并不完全相同，且受到需求压力影响。其中时间编码在高压力环境下表现出较明显的作用，Micro-GAE 的替换则在五档压力下均造成性能下降。这些结果为后续进一步研究历史表示与异步信用分配提供了实验依据。

## **五、参考文献**

[1] Pang, A., Wang, M., Sha, Z., Yue, W., Li, C., Chen, C. S., and Pun, M. O. *Heterogeneous Vertiport Selection Optimization for On-Demand Air Taxi Services: A Deep Reinforcement Learning Approach*. IEEE Transactions on Intelligent Transportation Systems, 2026.

[DOI 链接](https://doi.org/10.1109/TITS.2026.3680351?utm_source=chatgpt.com)

[2] Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., and Wu, Y. *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*. Advances in Neural Information Processing Systems (NeurIPS), 2022.

[论文原文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Conference.html?utm_source=chatgpt.com)

[3] Kuba, J. G., et al. *Heterogeneous-Agent Reinforcement Learning*. Journal of Machine Learning Research, 2024.

[论文 PDF](https://www.jmlr.org/papers/volume25/23-0488/23-0488.pdf?utm_source=chatgpt.com)

[4] Jung, W., et al. *Agent-Centric Actor-Critic for Asynchronous Multi-Agent Reinforcement Learning*. Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267, 2025.

[论文原文](https://proceedings.mlr.press/v267/jung25a.html?utm_source=chatgpt.com) · [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/jung25a/jung25a.pdf?utm_source=chatgpt.com)

[5] Menda, K., Chen, Y.-C., Grana, J., Bono, J. W., Tracey, B. D., Kochenderfer, M. J., and Wolpert, D. *Deep Reinforcement Learning for Event-Driven Multi-Agent Decision Processes*. IEEE Transactions on Intelligent Transportation Systems, 2019, 20(4): 1259–1268.

[DOI](https://doi.org/10.1109/TITS.2018.2848264?utm_source=chatgpt.com) · [论文 PDF](https://yi-chun-chen.github.io/DRL_event_driven.pdf?utm_source=chatgpt.com)

[6] Wen, M., Kuba, J. G., Lin, R., Zhang, W., Wen, Y., Wang, J., and Yang, Y. *Multi-Agent Reinforcement Learning Is a Sequence Modeling Problem*. Advances in Neural Information Processing Systems (NeurIPS), 2022.

[论文原文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html?utm_source=chatgpt.com)
