<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=ljc-ouro/ouro)
[![GitHub Repo stars](https://img.shields.io/github/stars/ljc-ouro/ouro?style=social)](https://github.com/ljc-ouro/ljc-ouro/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/ljc-ouro/ouro)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/ljc-ouro/ouro)](https://github.com/ljc-ouro/ljc-ouro/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/ljc-ouro/ouro/pulls)
[![Collection](https://img.shields.io/badge/🤖-Gridman%20%20Collection-blue)](https://hf.co/collections/maphy-ouro/gridman)

</div>

<div align="center">
  <h3>"我来从无聊的世界中拯救你了"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

* 此开源项目旨在完全从 0 开始, 构建第一代带状态 AI 架构 `Ouro`, 并以全新字节级语言模型 `Gridman` 作为体验开端.
* 仅用几十块钱成本与若干小时训练时间，即可训练出规模约为 52M 全新架构的超小语言模型 `Gridman-Mini`.
* `Gridman` 系列从极轻量模型到 B 级别模型全线覆盖，主线版本体积基本和 GPT-2 系列规模相当, Mini 版力求让普通个人 GPU 也能快速完成训练与复现.
* 项目同时开源了完整训练链路，覆盖预训练 (Pretrain), 监督微调 (SFT) 等全过程代码.
* 项目所有核心算法代码均从 0 使用 PyTorch 原生实现, 不依赖第三方库提供的高层抽象接口.
* 这不仅是一个全新架构的大语言模型全阶段开源项目，也是一套面向 `Ouro` 入门与实践的教程.
* 希望此项目能为更多人提供一个可复现, 可理解, 可扩展 `Ouro` 的起点, 一起感受状态 AI 模型的魅力, 并推动更广泛 AI 社区的进步, 为未来世界的变革做好准备.
* 项目交流 QQ 群: 198302483. 答案: State.

> 注：本项目基于 Apache 2.0 协议开源; 训练时长和成本在不同硬件上可能存在较大差异.

---

# 📌 架构优势

<table style="width: 100%; border-collapse: separate; border-spacing: 12px; background: transparent; border: none; table-layout: fixed;">
  <tr>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">⚙️ Theory</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">理论完备</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Ouro complete</div>
    </td>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">⚡ Speed</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">恒定速度</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Constant generation speed</div>
    </td>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">💾 VRAM</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">恒定显存</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Constant VRAM</div>
    </td>
  </tr>
  <tr>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">📦 No Cache</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">无需 KV Cache</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Without KV Cache</div>
    </td>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">✨ Learning</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">持续学习</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Continual learning</div>
    </td>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">∞ Context</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">无限上下文</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Infinite ctxlen</div>
    </td>
  </tr>
</table>

---

# 📌 项目介绍

注意力机制以及 `Transformer` 架构的出现, 拉开了了大语言模型和全民 AI 时代的序幕. 从 2022 年 GPT-3.5 第一次震惊世界开始, 时间连带着模型尺寸飞速增长, 整个 AI 世界在朝前狂奔. 但站在真正起作用的底层架构的视角上回顾, 我们似乎一直在原地踏步. 

那么问题是什么? 不存在更优的架构了吗? 并非如此, 我们犯了范式层级的错误: AI 模型正在逐步退化为一种事实上的极大规模函数. 输入被映射为输出, 人类只需要为这个静态怪物不断增加参数.

该项目尝试对这一默认前提进行一次彻底的反转. 不再将 AI 视为一个输入驱动的函数近似器, 而是将其构建为一个围绕内部 State 持续运行的系统. 在这一视角下: 

- State 不是缓存
- 不是附属变量
- 也不是 prompt 或上下文的延伸或压缩

相反, State 是模型的核心主体. 

这正是 `Ouro` 构建的核心哲学: **State is all you need**.

😊 一起感受状态模型的乐趣吧！

---

#### 🎉 本项目包含以下内容

- 提供完整的理论框架, 给出数学上 AGI 必备的完备性理论.
- 提供完整的 `Ouro` 结构代码，开启全新架构生态.
- 提供完整的 `Gridman` 语言模型训练代码, 预训练/微调权重同时开源.
- 提供 `ByteTokenizer` 无需任何先验分词器, 支持自定义模板标记扩展.
- 覆盖 Pretrain, SFT 完整训练流程.
- 提供全阶段开源数据，覆盖收集, 蒸馏, 清洗与去重后的高质量数据集.
- 提供原生 `StreamLoader` 数据加载器, 保证数据流贴合架构特性. 
- 提供原生多卡训练框架, 一键启动.
- 关键训练算法与核心模块均从 0 实现, 不依赖第三方框架封装.

#### 🎉 已 (预) 发布架构/模型列表

<details> 
<summary> <b>🔥 Ouro-Naxi</b> </summary>

`Ouro` 架构 `v1` 版本命名为 `Naxi`, 源自中国地名纳溪, 取纳溪成川之意. 后统一用 `-Naxi` 指代 `v1` 架构版本及对应的 `Gridman` 模型版本.

使用 `Ouro-Naxi` 架构训练的原生字节级语言模型 `Gridman` 模型列表:

<details> 
<summary> <b>v0d1</b> </summary>

| 模型 | 参数量 | 嵌入维度 | Blocks | Layers | Release |
|------|--------|--------|------|------|---------|
| Gridman-Naxi-Mini-v0d1 | 52.31 (50.74 + 1.57) M | 512 | 2 | 6 | 2026.04.01 |
| Gridman-Naxi-Small-v0d1 | 150.47 (145.75 + 4.72) M | 768 | 2 | 8 | 2026.04.01 |
| Gridman-Naxi-Medium-v0d1 | 396.31 (383.73 + 12.58) M | 1024 | 3 | 8 | 2026.04.01 |
| Gridman-Naxi-Large-v0d1 | 866.51 (840.30 + 26.21) M | 1280 | 4 | 9 | 2026.04.01 |
| Gridman-Naxi-XL-v0d1 (即将发布) | 1712.10 (1660.90 + 51.20) M | 1600  | 4 | 12 | 2026.04.01 |

</details>


<details> 
<summary> <b>v0d2 (即将发布)</b> </summary>

</details>

</details>

> 注：模型参数组成为 `可训练参数大小` + `状态参数大小`, 其中可训练参数被定义为训练时通过反向传播更新的参数; 模型名称后无显式标注 "即将发布" 的均已发布.

---

# 📌 架构理论

#### 💡 无状态模型

为什么我们需要一个状态? 在传统的 RNN 模型中, 状态转移方程通常被如下描述:

$$s_{t+1}, y = f(s_t, x)$$

$x$ 是输入, $y$ 是目标值, 这已经是强约束. 但是对于状态 $s_t$, 这里不存在的一个显式的约束. 这既为架构的设计提供了自由度, 也带来一种冗余性的暗示: $s_t$ 可能是完全多余的. 如果你遵照这种谕示将 $s_t$ 替换为历史输入的一种展平, 那么恭喜你, 你发明了 `Transformer`.

显然, 按照这么理解, `Transformer` 是一种标准的无状态模型.

#### 💡 基于概率的状态转移模型

哦, 等等, `Transformer` 真的抛弃了 $s_t$ 吗? 非也. 如果将上下文看作状态, 那 `Transformer` 不就描述了一个标准的状态转移

$$s_{t+1} = T(s_t)$$

吗?

这里其实存在着微妙的区别. 即使我们将上下文看作这里的状态, $T(s_t)$ 实际上也是给出了下一个状态的分布而非具体的状态. 这实际上是一种基于概率的状态转移模型.

是的, 状态在这里依然存在, 只是从模型内部打包到了外部. 此时状态转移的约束完全来自外部约束.

#### 💡 约束状态的隐变量

`Transformer` 是一种对于状态转移过度简化的模型, 让我们回顾标准的状态转移方程

$$s_{t+1}, y = f(s_t, x)$$

除了 $(x, y)$ 带来的外部约束, 模型自身对 $s_t$ 的约束到底是什么? 这强烈暗示我们这里存在一个隐变量 $\theta$, 仔细一想, 这正是权重的含义.

我们重写标准的状态转移方程

$$s_{t+1}, y = F(s_t, x, W(\theta, s_t))$$

称之为 Ouro 型状态转移方程, $F$ 由 $\theta, s_t$ 确定的权重 $W(\theta, s_t)$ 唯一决定. 现在距离我们得出最终的状态约束只有一步之遥了.

#### 💡 等效原理

为了得到我们想要的约束, 这里必须做出一个深刻的假设: 一个足够好的系统, 其推理 (前向传播) 和学习 (反向传播) 在局部不可区分. 这个假设称之为等效原理.

- 推理: $s_t$ 的改变

- 学习: $W(\theta, s_t)$ 的改变

基于等效原理, 在这里做一些简单的推导.

我们通过反向传播来更新权重, 即更新 $W(\theta, s_t)$. 那么在一次反向传播后权重变为 $W(\theta + \mathrm{d}\theta, s_t + \mathrm{d}s)$. 当模型收敛时展开这个式子得到

$$W(\theta + \mathrm{d}\theta, s_t + \mathrm{d}s)=W(\theta', s_t)+\frac{\partial W}{\partial s}(\theta', s_t)\mathrm{d}s$$

由于等效原理和递推方程我们自然的要求 $s_{t+1} = s_t + \mathrm{d}s$, 带入得到

$$W(\theta', s_{t+1}) + s_t\frac{\partial W}{\partial s}(\theta', s_t)=W(\theta', s_t)+ s_{t+1}\frac{\partial W}{\partial s}(\theta', s_t)$$

令 $J_{t}=\frac{\partial W}{\partial s}(\theta', s_t)$, 重写为

$$W(\theta', s_{t+1})-W(\theta', s_{t})=J_t (s_{t+1} - s_{t})$$

实际上这就是我们需要的约束!

也可以直接写作连续形式

$$\mathrm{d}W=J\mathrm{d}s$$

这告诉我们学习-推理的局部不可区分性本质上来自于链式法则.

#### 💡 Ouro 完备

设数据域：

$$
\mathcal{D} \subseteq \mathcal{X} \times \mathcal{Y}
$$

Ouro 型状态转移方程定义为：

$$
(s_{t+1}, y_t) = F(s_t, x_t, W(\theta, s_t)\big), 
\quad (x_t, y_t') \sim \mathcal{D}
$$

定义总损失：

$$L(\theta) = L_1(\theta) + \lambda L_2(\theta)$$

$L_1$ 任务损失

$$
L_1(\theta)
= \mathbb{E}_{(x,y') \sim \mathcal{D}}
\big[ \ell(y_t, y') \big]
$$

$L_2$ 状态约束损失

$$
L_2(\theta)
= \mathbb{E}_{(x_t,y_t') \sim \mathcal{D}}
\left[
\left\|
W(\theta, s_{t+1}) - W(\theta, s_t)- J_t (s_{t+1} - s_t)
\right\|^2
\right]
$$

其中：

- $s_t \in \mathcal{S}$ 为状态
- $\theta \in \Theta$ 为参数
- $G : \Theta \times \mathcal{S} \to \mathcal{W}$
- $J_t = \frac{\partial W}{\partial s}(\theta, s_t)$

并满足：

$$
\left\\{
\begin{aligned}
&\lim_{t \to \infty} |\nabla_\theta L(\theta_t)| = 0 \\
&\lim_{t \to \infty} |L_2| = 0 \\
&\lim_{t \to \infty} \theta_t = \theta' \\
&\sup_t |s_t| < \infty \\
&G \in C^1(\Theta \times \mathcal{S})
\end{aligned}
\right.
$$

则称 $F$ 在 $\mathcal{D}$ 上是 **Ouro 完备的**.

#### 💡 AGI

若 $F$ 同时满足 Ouro 完备与图灵完备, 则称 $F$ 是 AGI (Artificial General Intelligence).

---

# 📌 模型

## 🚀 全局架构

`Ouro` 并未采用传统的层堆叠架构, 而是通过类树形结构组织起来.

- Ouro 类: 主类, 全局唯一, 为 OuroBlock 类的堆叠. OuroBlock 类之间使用注意力残差 (AttnRes) 连接.

- OuroBlock 类: 为 OuroLayer 类的堆叠. OuroLayer 类之间使用涌现注意力 (Emergent Attention) 连接.

- OuroLayer 类: `Ouro` 最底层结构, 在指定索引处开启动态前馈层 (Dynamic-FFN).

```python
class Ouro:
""" 
Pre-Nonm
OuroBlocks
注意力残差
FFN
残差输出
"""
...


class OuroBlock: 
"""
OuroLayers
涌现注意力 
注意力门控
残差输出
"""
...


class OuroLayer:
"""
前缀注意力
动态前馈层
残差输出
"""
...
```
与目前主流演进方向不同, `Ouro` 并未拥抱所谓 $O(n)$ 的线性注意力, 严格来说是 $O(nd^2)$, 而是全面拥抱 $O(n^2d)$ 复杂度的结构. 即使内部使用了线性注意力, 其实现也也选择了 $O(n^2d)$ 的形式.

这也使得 `Ouro` 在其输入窗口内拥有严格强于 `Transformer` 的表达能力. 同时得益于 `Dyn-FFN` 组件的设计, `Ouro` 在推理时依然能享受到 $O(1)$ 恒定显存和算力需求的福利并实现完全的持续学习.

## 🚀 核心组件

#### Ⅰ 动态前馈层 (Dynamic-FFN)

**动态前馈层 (Dynamic-FFN, Dyn-FFN)** 是整个 `Ouro` 架构的灵魂. 传统的 `FFN` 可以表示为

```
Linear1 [InDim, OutDim] -> SiLU [OutDim] -> Linear2 [OutDim, InDim]
```

为了将 `FFN` 改成输入响应和记忆响应的, 我们将 `Linear1` 视作一个

$$\text{InDim} \times \text{OutDim}$$

的矩阵. 注意到

$$\text{InDim} \times \text{InDim} \times \text{InDim} \times \text{OutDim}$$

依然是一个 `[InDim, OutDim]` 形状的矩阵, 于是我们自然的引入一个 `[InDim, InDim]` 形状的方阵作为记忆矩阵.

此时数据流变为


```
Dyn-Linear1 -> SiLU -> Linear2
```

或者等价的

```
Mem-Linear -> Linear1 -> SiLU -> Linear2
```

只是这里 `Mem` 的权重 (参数) 不被反向传播更新, 而是采取了 `DeltaRule` 作为主动的前向更新策略.

除了输入数据节点流的改变, `Dyn-FFN` 同时返回局部产生的线性注意力打分 `scores`

$$\text{y, scores} = \text{Dyn-FFN(x)}$$

为 `OuroBlock` 中 **涌现注意力 (Emergent Attention)** 的产生做铺垫.

#### Ⅱ 涌现注意力 (Emergent Attention)

涌现注意力的核心思想可以简单的阐述为标准注意力可以被线性注意力及其残差逼近

$$\text{Attn}=\text{LinearAttn}+\Delta\text{LinearAttn}$$

特别的, `Ouro` 中的线性注意力产生于记忆更新的 `DeltaRule` 计算过程中.

在宏观上来看, `OuroBlock` 层级的注意力就是记忆被唤醒时产生的注意力之"和".

## 🚀 Ouro 结构示意图

![structure](./images/ouro_struct.png)

---

# 📌 训练

`Ouro` 是通用架构, 以 `Ouro` 作为核心设计的语言模型称为 `Gridman`. 本章节所指的训练均指 `Gridman` 模型的训练.  

## 🛠️ 数据集

#### Ⅰ  Tokenizer

得益于 `Ouro` 架构的强大, `Gridman` 的实现直接选择了使用纯字节级别的分词器 `ByteTokenizer`. 这也意味着 `Gridman` 是一个原生字节级别的语言模型, 无需进行任何传统意义上的分词即可训练!

从根源上消除了多语言训练困难或 oov 等因分词带来的干扰.

#### Ⅱ 预训练 (Pretrain) 数据集

预训练数据集来自 [Minimind 数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset) 中的 `pretrain_t2t.jsonl` 数据集加上开源中文 `wiki` 数据的乱序混合得到, 标记为 `pretrain.jsonl` 数据集. 数据集下载链接见下方.

#### Ⅲ 微调 (SFT) 数据集

微调数据集来自 [Minimind 数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset) 中的 `sft_t2t.jsonl` 数据集. 本项目未对该数据集进行任何处理, 为了方便与统一. 将该数据集重新标记为 `sft.jsonl`, 数据集下载链接见下方.

#### Ⅳ 数据加载

> `Gridman` 训练数据集下载地址：[ModelScope]() | [HuggingFace]()

## 🛠️ 预训练 (Pretrain)

## 🛠️ 微调 (SFT)

---

# 📌 Acheron Thinking（冥河思考）

> 本 fork 新增：基于 Ouro 状态架构的仿生思考机制。**纯推理侧，零训练改动**，是验证 Ouro 状态动力学的一把钥匙。

## 💡 理念

主流的 CoT / ToT / ReAct 等思考方式本质是 **prompt 层面的技巧**——把推理过程外化为文本 token。Acheron 反其道而行：

- 不生成任何推理 token，让模型内部状态在**吸引子动力学**中自循环演化
- 状态收敛后再生成回答——「先想清楚，再开口」
- 充分利用 Ouro 的三大状态组件：`mem` 矩阵（联想记忆）、`c_state`（工作记忆）、`c_state_queue`（情景缓冲）

## 🚀 快速开始

```bash
python main_think.py                              # 交互模式（默认 focused）
python main_think.py --query "人生的价值是什么"     # 单次查询
python main_think.py --mode divergent             # 默认发散模式
```

### 交互命令

| 命令 | 效果 |
|------|------|
| `/focus 问题` | **聚焦思考**：快速收敛，适合事实性/直接问题 |
| `/diverge 问题` | **发散思考**：噪声注入 + 极紧阈值，探索更深 |
| `/reflect 问题` | **反思思考**：双重 query 重读，通过 temporal attention 回顾上一步状态 |
| `/preview` | 开关心声预览（思考过程可视化） |
| `/trajectory` | 查看上轮思考的完整能量轨迹 |
| `/clear` | 清空工作记忆（保留训练好的长期记忆） |
| `/quit` | 退出 |

## 🧠 三种思考模式

| 模式 | 前向次数 | 方差 | 收敛阈值 | 定位 |
|------|---------|------|---------|------|
| focused | ~4 次重读 | 低 | 1e-3 | 稳定中上，低风险 |
| divergent | 多变 + 噪声 | **高** | 5e-5 | 创意探索，偶有惊艳经常跑偏 |
| reflective | ~8 次（双重重读） | 低 | 1e-3 | 反复确认，更慢更慎重 |

## ⚙️ 工作原理

### 1. Query 重读（核心）

每步思考重新处理完整 query 序列。关键：**seq_len > 1 时 mem 更新阻尼 `1/seq_len` 生效**，知识（mem 矩阵）以温和方式参与联想检索，而非被冻结或被污染。

```python
# thinking.py 中的核心循环
for step in range(max_steps):
    self._think_step(input_ids)      # 重读 query，mem 阻尼演化
    current_state = self._get_current_state()
    energy = ||current_state - prev_state||² / dim
    if energy < threshold: break      # 状态稳定 → 收敛 → 输出
```

### 2. 收敛检测

以能量函数 `E(t) = ||c_t - c_{t-1}||² / dim` 监测 STM 状态变化率。能量低于阈值即认为思考完成——对应大脑「决策落定」的过程。

### 3. 思考可视化（/preview）

每步思考后做一次微型自回归生成，展示「如果此刻停止思考，模型会说什么」——内心独白的演变轨迹。

## 📊 实验发现

在 Gridman-Medium 上的对比实验（同一问题、3 次重复、关键词命中统计）：

| 模式 | 平均关键词命中 |
|------|--------------|
| 不思考（原始 SFT） | 最高，最稳定 |
| focused | 4.0 |
| divergent | 1.7 |

**结论**：思考机制改变了状态演化路径，但**无法超越模型的知识上限**。思考只能重新采样训练中学到的模式——这印证了训练目标的重要性（详见本 fork 提交的 [issue #4](https://github.com/zhihumomo/ouro/issues/4)）。

## 🧬 L2 状态约束损失（等效原理）实现状态

> commit `f9fc435`：本 fork 实现了理论中的 $L_2$ 约束，但**尚未经过端到端验证**。

### 已实现（数学验证通过）

- **随机探针 VJP 形式**：利用 $\mathbb{E}_v[\langle x, v\rangle^2] = \|x\|^2$，把精确的 $\|dW - J\,ds\|^2$ 化为可用 `torch.autograd.grad` 一次反向传播计算的期望形式
- **验证结果**：L2 单独训练 80 步下降 39.9 倍（单调）；带 L2 训练的模型思考时状态演化更稳定（状态变化 0.0010 vs 0.0020）

### 未验证（硬件限制）

**我们不具备足够的计算资源训练一个可目测的 L2 模型。** 本地仅 12GB 笔记本 GPU：

- 只能训练 4000 步的 Gridman-Mini（约 25 分钟），远低于作者训练的百万步量级
- Mini A/B 对照（无 L2 vs 带 L2 λ=1.0）在"思考是否提升续写预测"上**无显著差异**（±0.0005 log-prob，噪声内）
- 测试中发现 λ=0.1 时 L2 被交叉熵压过（训练中 L2 从 1.8e-3 涨到 4.9e-2），λ=1.0 才能压制

**L2 是否真的让"思考"超越"不思考"，需要作者级别的算力才能验证。** 这份实现的意义在于：用数学上严格、成本可控的方式，把理论中的约束变成了可训练的目标——剩下的交给更大规模的训练。

## ⚠️ 已知限制

1. **思考不创造知识**：mem 矩阵承载的知识固定，思考只是改变激活哪些学过的模式
2. 当前模型（4MB 训练）的吸引子较弱，思考效果受限于此
3. L2 实现仅经数学与小规模验证，端到端效果待高算力确认

---

# 🎓 引用

如果 `Ouro` 对您的研究或工作有所帮助，欢迎引用：

```bibtex
@misc{Ouro,
  title = {Ouro: The Next-Gen AI Architecture},
  author = {Jinchang Liu},
  year = {2025},
  url = {https://github.com/ljc-ouro/ouro},
  note = {GitHub repository, accessed 2026}
}
```

---

## 🫶支持者

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ljc-ouro/ouro&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ljc-ouro/ouro&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ljc-ouro/ouro&type=Date"/>
</picture>

<a href="https://github.com/ljc-ouro/ouro/stargazers">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/stars/dark/ljc-ouro/ouro"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/stars/ljc-ouro/ouro"/>
      <img alt="Star poster" src="https://reporoster.com/stars/ljc-ouro/ouro"/>
    </picture>
</a>

<a href="https://github.com/ljc-ouro/ouro/network/members">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/forks/dark/ljc-ouro/ouro"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/forks/ljc-ouro/ouro"/>
      <img alt="Fork poster" src="https://reporoster.com/forks/ljc-ouro/ouro"/>
    </picture>
</a>

---

# ⚖️ 开源协议

本项目采用 [Apache License 2.0](LICENSE) 开源协议.
