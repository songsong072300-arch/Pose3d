# BoxDreamer：通过推测包围盒角点实现可泛化的物体位姿估计

英文标题：**BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation**

作者：Yuanhong Yu¹˒³、Xingyi He¹˒³、Chen Zhao⁴、Junhao Yu⁵、Jiaqi Yang⁶、Ruizhen Hu⁷、Yujun Shen³、Xing Zhu³、Xiaowei Zhou¹、Sida Peng¹˒²（通信作者）。

单位：¹浙江大学；²湘江实验室；³蚂蚁集团；⁴洛桑联邦理工学院（EPFL）；⁵重庆大学；⁶西北工业大学；⁷深圳大学。

项目主页：https://zju3dv.github.io/boxdreamer

原文：[2504.07955v2.pdf](2504.07955v2.pdf)，arXiv:2504.07955v2，2025 年 9 月 30 日版本，共 11 页。

> 翻译说明：本文按原论文的章节顺序翻译正文、图注、表注和致谢，并以 Markdown 重排全部七张实验表。作者、方法名、数据集名及引用编号保持原样。参考文献保留英文书目信息，便于检索。图像本身请对照原 PDF；文末“译者注”是针对当前项目的补充，不属于原论文。原文公式中存在的特殊形式与符号复用均保留，不擅自改写。

## 摘要

本文提出一种基于 RGB 的可泛化物体位姿估计方法，专门应对稀疏视角条件下的挑战。尽管现有方法能够估计未见过物体的位姿，但在存在遮挡且参考视角稀疏的场景中，其泛化能力仍然有限，这制约了它们在真实世界中的应用。

为克服这些限制，我们引入物体包围盒的角点作为物体位姿的中间表示。物体的三维角点可以从稀疏输入视角中可靠地恢复，而目标视角中的二维角点则由一种新颖的、基于参考信息的点合成器估计；即使存在遮挡，该合成器也能有效工作。作为物体的语义点，物体角点能够自然地建立二维—三维对应关系，进而通过 PnP 算法估计物体位姿。

在 YCB-Video 和 Occluded-LINEMOD 数据集上的大量实验表明，我们的方法优于当前先进方法。这些结果突出了所提出表示的有效性，并显著提升了物体位姿估计的泛化能力，而这对于真实世界应用至关重要。

## 1. 引言

估计物体与相机之间的刚性变换，即物体位姿估计，是增强现实（AR）和物体操作等多种任务的基础 [6,16]。我们关注基于稀疏视角 RGB 图像的可泛化物体位姿估计：对于每个物体，只能获得有限数量的参考图像作为先验信息。这一任务要求位姿估计方法既能通过一次前向传播处理任意未见过的物体，又能从有限参考视角中稳健、准确地估计位姿。

如图 1 所示，目前可泛化物体位姿估计的研究主要包括两种范式：基于检索的方法 [2,25,30,52] 和基于匹配的方法 [4,12,35]。

为了估计目标位姿，基于检索的方法首先检索最相似的参考图像，以初始化物体位姿，然后再进行位姿细化。当密集参考视角及其对应位姿标注可用时，这类方法十分有效。然而，在稀疏视角场景中，很难检索到观察角度相近的参考图像，因而会产生不准确的位姿初始化。次优的初始位姿会影响后续细化模块。此外，当查询图像中的物体受到严重遮挡时，选择正确参考图像极具挑战，往往会导致位姿估计失败。

基于匹配的方法先从参考图像中重建目标物体的点云，再建立查询图像与点云之间的对应关系以估计位姿。当能够恢复准确的物体点云时，这些方法可以实现高精度估计。但稳健的二维—三维匹配本质上依赖完整的重建点云，而这在稀疏视角场景中难以实现。另外，物体遮挡限制了图像像素与点云之间的关联，显著削弱了基于对应关系的方法在这类条件下的有效性。

本文提出一个新颖的可泛化六维物体位姿估计框架，只需要少量参考图像即可预测任意物体的位姿，并能有效处理严重遮挡。如图 1 所示，我们的核心思想是将物体包围盒用作几何基本单元，它可以从稀疏参考视角中高效重建。随后，我们在查询图像中准确预测三维包围盒八个角点的二维投影，用于位姿估计。

作为物体的语义点，包围盒角点能够自然建立二维—三维对应关系，而且三维角点和二维角点都可以在没有密集视角的情况下恢复，使我们的方法适用于稀疏视角条件。此外，与需要在查询图像和三维几何表示之间进行密集匹配的方法相比，我们的方法本身对遮挡更加稳健。

具体而言，该方法包含两个主要步骤：恢复三维包围盒，以及预测查询图像中三维包围盒角点的二维投影。首先，我们使用稀疏视角重建工具估计相机位姿，并恢复物体的近似结构，以计算其三维包围盒。为了利用三维角点位置进行位姿估计，我们将三维包围盒角点投影到每张参考图像中，生成角点热力图，然后将其输入端到端 Transformer 解码器，预测查询视角中三维包围盒角点的二维投影。

物体包围盒角点具有全局性，因此我们的方法可以根据物体的可见部分和参考示例推断角点位置。最后，利用二维—三维对应关系，通过 Perspective-n-Point（PnP）算法恢复物体的六自由度位姿。需要指出的是，本方法不要求用于位姿估计的三维包围盒十分精确，第 4.5 节的实验验证了这一点。

为评估方法的有效性，我们在 Occluded LINEMOD [1] 和 YCB-Video [43] 上进行实验，这两个数据集包含大量遮挡等困难场景。我们还在 LINEMOD [14]、OnePose [35] 等多样化数据集上进行额外测试，以展示框架的泛化能力和对稀疏视角的适应性。实验结果表明，在稀疏视角条件下，本方法优于基于检索和匹配的方法，并在多种困难场景中表现出更强的通用性和有效性。

本工作的贡献包括：

- 提出一种新的可泛化物体位姿估计框架，仅使用稀疏 RGB 参考图像即可有效估计物体位姿。
- 提出将物体包围盒角点作为可泛化物体位姿估计的中间表示。
- 提出端到端 Transformer 解码器，在参考角点示例的引导下直接回归查询视角中三维包围盒的二维投影。

**图 1：不同可泛化范式的比较。** 与现有可泛化物体位姿估计方法不同，BoxDreamer 利用包围盒表示处理不完整的物体观测，即使在严重遮挡条件下也能实现稳健的位姿估计。图中比较了检索式方法、匹配式方法与本文方法。

## 2. 相关工作

### 2.1 基于 CAD 模型的物体位姿估计

基于模型的物体位姿估计方法大体可以分为两类。实例级位姿估计器 [5,15,18,22,23,28,31,33,34,43,47] 需要针对每个物体分别训练，通常依赖 CAD 模型来建立二维—三维对应关系 [5,15,28,31,33,34]，或用于渲染和位姿细化 [22,47]。也有方法尝试直接从 RGB 图像回归物体位姿 [18,23,43]。虽然它们对已知物体能取得较高精度，但无法泛化到未见过的物体。

更具泛化能力的类别级位姿估计器 [3,7,10,17,24,38,40,46,48–50] 首先将物体归入预定义类别，然后利用类别特定先验估计位姿。例如，NOCS [38] 根据 CAD 模型为每个类别建立规范坐标系，以提供类别级先验。尽管更大规模的数据集 [49,50] 增强了泛化能力，这些方法仍然面临挑战：真实世界中的物体过于多样，或过于少见而难以归类，限制了它们在通用位姿估计中的适用性。

我们的框架通过消除对 CAD 模型的依赖、转而利用物体三维包围盒来应对这些挑战。该表示获取成本低，可以从稀疏且未标定位姿的 RGB 图像中获得，因此为任意物体位姿估计提供了稳健方案。

### 2.2 不依赖 CAD 模型的物体位姿估计

为解决未见物体的位姿估计问题，Gen6D [25] 提出利用带位姿标注的密集视角数据库进行位姿初始化和细化的框架，在不依赖 CAD 模型的情况下展现了较强泛化能力。后续方法 [2,30,52] 改进了这一框架中的检测与细化模块，但对密集采集视角的依赖仍是一项重要限制。

与此同时，OnePose [35] 及其扩展 OnePose++ [12] 提出了基于重建的方法，在重建点云与查询图像之间建立对应关系。通过引入 Detector-Free SfM [13]，OnePose++ 进一步提高了准确性，但这些方法仍需要密集视角采集，而且所采集图像的可见性和覆盖范围会影响性能。

### 2.3 物体三维表示

为物体提供三维信息是实现可泛化物体位姿估计的关键。OnePose 系列使用密集点云表示物体；Gen6D 根据多视角图像构建体积表示，并用于位姿细化；LatentFusion [32] 建立三维潜在表示，以进行位姿估计和细化；GS-Pose [2] 使用三维高斯 [19] 表示物体，并据此进行位姿细化。

尽管取得了进展，这些方法都没有提供一种适用于任意未见物体六自由度位姿估计的、既简单又稳健的三维表示：它们要么严重依赖密集参考图像，要么难以应对遮挡或无纹理表面等困难条件。相比之下，我们创新性地使用包围盒角点，作为可泛化物体位姿估计的紧凑、有效表示。

### 2.4 稀疏视角物体位姿估计

稀疏视角设置因其实用性和成本效率，在计算机视觉领域受到广泛关注 [26,27,30,36,37,40,51,53]。真实应用中，密集且全面地采集物体视角通常耗时耗资源。因此，从稀疏观测中准确恢复物体位姿，可以极大促进物体操作等实时应用。

近期一些工作研究了稀疏视角位姿估计 [26,27,37]。例如，Luo 等人 [26] 使用物体高斯表示，结合“渲染—细化”策略取得较高精度，但构建该表示大约需要 10 分钟，限制了实用性。Nguyen 等人 [27] 生成新视角嵌入，Sun 等人 [37] 使用新视角合成（NVS）模型；然而，这两类方法都依赖生成能力，可能难以应对真实世界中多样化的物体，而且 NVS 也很耗时。

FoundationPose [42] 通过在逼真合成数据集上大规模训练，在稀疏视角条件下表现突出，但其对深度输入的依赖限制了真实应用。相比之下，我们的方法采用轻量、有效的表示，利用稀疏参考视角实现稳健、实时的位姿估计。

## 3. 方法

图 2 展示了 BoxDreamer 的总体框架。本方法输入一张查询图像 $I_q$、一组参考图像 $\{I_0,\ldots,I_i\}$，以及用于确定目标物体的对应检测结果 $\{\mathcal M_0,\ldots,\mathcal M_i\}$，目标是输出物体位姿 $\xi_q$。

为此，我们使用现成的重建方法获得目标物体点云 $\mathbf P$、三维包围盒 $\mathbf B$ 及对应的参考位姿 $\{\xi_0,\ldots,\xi_i\}$，见第 3.1 节。

接着，我们提出一种新颖的包围盒估计网络，预测查询视角中的包围盒投影角点 $\mathbf b_q$。利用这些预测，在预测二维角点 $\mathbf b_q$ 与重建三维角点 $\mathbf B$ 之间建立二维—三维对应关系，再通过 PnP 算法恢复查询物体位姿 $\xi_q$，见第 3.2 节。

**图 2：方法概览。** 对于每个物体，BoxDreamer 首先使用稀疏视角重建方法，从一组参考图像中恢复其粗略结构。推理时，在参考包围盒角点的引导下，为查询图像预测二维包围盒热力图，建立二维—三维对应关系，并通过 PnP 算法恢复物体位姿。

### 3.1 物体包围盒的准备

#### 以三维包围盒作为物体表示

为了应对 OnePose 的限制，我们避免依赖密集点云，转而使用三维包围盒 $\mathbf B$ 表示物体，将其作为位姿估计的粗略空间先验。与密集点云相比，包围盒的优势在于：可以从稀疏视角输入中恢复三维包围盒，从而在缺少密集观测时仍能恢复物体坐标。

为获得三维包围盒 $\mathbf B$，我们首先收集稀疏参考图像，再使用现成重建方法 [21,39,41,45] 提取物体稀疏几何。随后根据物体检测结果过滤无关点，最终获得包围物体的三维包围盒。

将裁剪后的参考图像输入完全前馈式的重建方法，可以得到物体点云 $\mathbf P$ 和参考视角的相机位姿。以实验使用的 DUSt3R [41] 为例，稀疏视角图像直接输入网络以预测 Pointmap，即建立图像像素与对应三维点之间的密集对应关系。获得 Pointmap 后，利用这些二维—三维对应关系，通过 PnP 算法高效恢复参考相机位姿。这一过程同时恢复三维几何和相机位姿，尤其适合稀疏视角场景。

对于位姿为 $\xi=(R,\mathbf t)$ 的每个参考视角，我们首先使用内参矩阵 $K$ 和透视投影函数 $\pi(\cdot)$，将三维点 $\mathbf p\in\mathbf P$ 投影到图像平面：

$$
\mathbf p'=\pi\bigl(K(R\mathbf p+\mathbf t)\bigr).
$$

其中 $\pi(\cdot)$ 将齐次坐标转换为二维图像坐标。

接着，丢弃投影落在参考图像检测区域之外的点。过滤后的点云定义为：

$$
\tilde{\mathbf P}=\left\{\mathbf p\in\mathbf P\;\middle|\;\forall i,\;\mathbf p'\in\mathcal M_i\right\}.
$$

最后，将过滤后的点云 $\tilde{\mathbf P}$ 平移到以物体为中心的坐标系中，并据此计算包围物体的三维包围盒 $\mathbf B$。

#### 以二维热力图作为包围盒表示

我们观察到，直接使用三维包围盒的八个角点坐标，常常会因为输入信号本身过于稀疏而降低性能。为更好地利用参考信息，我们将三维包围盒投影到图像平面，生成表示其二维空间分布的热力图。该策略与能够有效处理热力图数据的视觉 Transformer 很契合。具体而言，给定包围盒 $\mathbf B$ 的八个三维角点，我们通过透视投影计算它们在图像平面上的投影 $\mathbf b$。

然而，使用 one-hot 热力图表示投影角点，会导致监督信号过度稀疏、缺乏平滑性，从而妨碍模型学习。为缓解这一问题，我们受到 CornerNet [20] 的启发，在每个真实角点周围施加高斯平滑。这样能够减轻一定半径内负样本位置受到的惩罚，并避免正负区域之间的突变。

实践中，我们发现 CornerNet 的超参数不能直接迁移到本任务。为进一步增强热力图的平滑性，我们将热力图函数重新定义为：

$$
\mathbf H(x,y,i)=\exp\left(-\frac{\sqrt{(x-x_i)^2+(y-y_i)^2}}{2\sigma^2}\right).
$$

我们将分母项 $2\sigma^2$ 设置为物体尺寸十分之一的平方；对每个角点 $i$，物体尺寸定义为该角点到物体二维中心的像素距离。

使用三维包围盒而非密集点云作为空间先验，使我们能够设计更适合稀疏视角输入和不完整重建的位姿估计流程。它消除了对点云特征的依赖，并允许任意合适的重建方法生成三维包围盒，从而使流程更加灵活。此外，通过二维热力图表示包围盒，可以避免直接依赖可能带噪声或对尺度敏感的三维点。

### 3.2 物体位姿估计

获得参考视角的包围盒热力图 $\{\mathbf H_0,\ldots,\mathbf H_i\}$ 后，我们使用端到端 Transformer 解码器，直接推断查询视角对应的热力图 $\mathbf H_q$。

具体而言，将参考图像和查询图像输入预训练 DINOv2 [29] 模型，提取图像特征：

$$
\{\mathbf F_0,\ldots,\mathbf F_i,\mathbf F_q\},\qquad
\mathbf F_i\in\mathbb R^{\frac Hp\times\frac Wp\times d},
$$

其中 $p$ 是图像块大小，$d$ 是特征维度。

随后，将每张包围盒热力图 $\mathbf H_i\in\mathbb R^{H\times W\times8}$ 划分成互不重叠的图像块，得到：

$$
\mathbf H_i^p\in\mathbb R^{\frac Hp\times\frac Wp\times8p^2}.
$$

为融合图像特征与分块热力图，我们使用线性层，将热力图 token 投影到与图像特征相同的维度：

$$
\mathbf H_i^p=\operatorname{Linear}(\mathbf H_i^p)
\in\mathbb R^{\frac Hp\times\frac Wp\times d}.
$$

再将投影后的热力图 token 与相应图像特征逐元素相加：

$$
\mathbf F_i'=\mathbf F_i+\mathbf H_i^p.
$$

对于查询图像，我们使用可学习的查询 token $\mathbf Q\in\mathbb R^{\frac Hp\times\frac Wp\times d}$，而不是热力图 token。随后，将参考和查询特征展平，沿图像块维度拼接，得到长度为下式的一维 token 序列：

$$
l=(N+1)\frac{H\times W}{p^2},
$$

其中 $N$ 为参考视角数量。该序列输入具有 $L$ 层全自注意力的 Transformer 解码器，生成查询包围盒特征：

$$
\mathbf F_q'\in\mathbb R^{\frac Hp\times\frac Wp\times d}.
$$

最后，使用线性层将查询特征映射回原热力图维度，再通过 unpatchify 将图像块还原，得到最终查询包围盒热力图：

$$
\mathbf H_q=\operatorname{Sigmoid}\bigl(\operatorname{Linear}(\mathbf F_q')\bigr)
\in\mathbb R^{H\times W\times8}.
$$

给定预测查询热力图和重建三维包围盒，我们依据预定义的通道顺序，将每个热力图通道分配给特定包围盒角点，从而建立二维—三维对应关系。随后使用 PnP 算法恢复物体位姿 $\xi_q$。

### 3.3 训练

**监督方式。** 我们使用 Smooth L1 Loss，同时监督预测包围盒热力图（粗粒度）和各个角点（细粒度）。

粗粒度损失定义为：

$$
L_{\mathrm{coarse}}=\frac1N\sum_{i=1}^{N}
\operatorname{SmoothL1}(h_i,\hat h_i),
$$

其中 $h_i$ 和 $\hat h_i$ 分别表示真实与预测热力图数值。

细粒度损失用于提高角点定位精度：

$$
L_{\mathrm{fine}}=\frac18\sum_{i=1}^{8}
\operatorname{SmoothL1}(b_i,\hat b_i),
$$

其中 $b_i$ 和 $\hat b_i$ 分别为真实与预测角点坐标。最终损失结合两项：

$$
L=L_{\mathrm{coarse}}+\lambda L_{\mathrm{fine}}.
$$

$\lambda$ 是平衡两项损失的超参数，实验中设置为 $2.0$。

### 3.4 实现细节

**训练数据。** 使用 Objaverse 合成数据 [8,9] 和 OnePose 真实数据 [35]，共包含超过 4.5 万个合成物体、50 个真实物体，以及超过 290 万张图像。

**数据增强。** 为提高泛化能力，我们绕随机轴、在 $[-\pi,\pi]$ 范围内随机旋转三维包围盒，打破其与语义信息的直接关联。同时使用运动模糊、噪声等 RGB 增强，将合成图像与随机 SUN2012 [44] 背景合成。此外，通过截断和掩蔽对物体施加随机遮挡，以进一步提高遮挡处理能力。

**网络结构。** Transformer 解码器包含 $L=12$ 层，隐藏维度 $d=768$，注意力头数为 8。遵循 DINOv2-Base [29]，图像块大小设置为 14。

**训练设置。** 使用 AdamW 优化器，初始学习率为 $10^{-4}$，采用余弦衰减调度。训练中，参考图像数量在 1 到 15 之间动态采样，每张 GPU 的 batch size 相应在 144 到 18 之间变化。使用 8 张 A100-SXM4-80GB GPU 训练 100 个 epoch。

## 4. 实验

本节主要在四个数据集上评估本方法：LINEMOD 与 Occluded LINEMOD（第 4.2 节）、YCB-Video（第 4.3 节），以及 OnePose-LowTexture（第 4.4 节）。我们也在 OnePose 上评估泛化能力，对应结果见补充材料。首先在第 4.1 节介绍实验设置和基线选择。

### 4.1 实验设置与基线

**基线。** 我们使用 Gen6D 与 OnePose++，代表可泛化位姿估计中的两种范式。虽然 LocPoseNet [52] 改进了 Gen6D 的检测精度，但所有实验中我们都向 Gen6D 提供准确的真实检测结果，以保证精确的初始平移，因此无需与 LocPoseNet 直接比较。

此外，Cas6D [30] 和 GS-Pose 也提供稀疏视角结果，且使用与本方法相同的视角采样策略，因此在第 4.2 节将它们纳入比较。为进一步展示本方法的有效性，我们还与先进的实例级位姿估计方法 [33,34,43] 比较。

**参考数据库。** 评估采用以下数据库：

- **LINEMOD 与 Occluded LINEMOD：** 使用既有研究 [12,25,33,35] 的标准训练—测试划分，每个物体约有 180 张参考图像。
- **YCB-Video：** 构建三种参考数据库，以评估对参考质量变化的稳健性。稀疏数据库：遵循 FoundationPose [42]，从不同视频序列为每个物体采样 16 张参考图像。最大重叠数据库：选择重叠程度最高的视频序列作为参考数据库，以尽可能充分地测试基线方法。最小遮挡数据库：为减轻最大重叠序列中的遮挡，人工选择同时保持较高重叠度和较少遮挡的参考序列。详细构建方法见补充材料。
- **OnePose 与 OnePose-LowTexture：** 与 [12,35] 一样，使用第一段序列作为参考数据库。

为保证不同数据集与方法之间的一致性，我们统一使用最远点采样（FPS）算法采样稀疏参考视角。对于参考图像数超过五张的实验，除非另有说明，BoxDreamer 首先直接计算查询与参考 DINOv2 特征的余弦相似度，选择五个近邻。

**评估指标。** 遵循各数据集既有研究的惯例：

- LINEMOD、Occluded LINEMOD，以及具有 CAD 模型的 OnePose-LowTexture 子集使用 ADD(s)-0.1d 和阈值为 5 像素的 Proj2D。
- YCB-Video 使用 ADD-AUC 和 ADDs-AUC。

### 4.2 LM 与 Occluded LM 上的结果

首先在 LINEMOD 和 Occluded LINEMOD 两个常用基准上评估。实验使用两种参考数据库密度：5 个参考视角和 25 个参考视角。后者是 OnePose++ 成功重建所有物体所需的最少视角数量。在 LINEMOD 上，另外使用 16 和 32 个参考视角，以便与 GS-Pose、Cas6D 的结果比较。与 Gen6D 和 OnePose++ 的详细比较见补充材料。

表 1 汇总了 LINEMOD 子集上的性能。Gen6D 使用不同子集训练；由于 Cas6D 没有公开检查点，我们采用其论文报告的结果。对于 GS-Pose，我们使用与本文相同的参考数据库重新测试，细化参数采用公开代码的默认设置。

**表 1：LINEMOD 子集上的比较。指标为 ADD(s)-0.1d。** Gen6D† 表示细化使用全部参考图像；Gen6D‡ 为 GS-Pose 公开的结果；GS-Pose init 为不进行高斯泼溅细化的初始结果。原文用颜色标注前三名，此处保留数值。

| 方法 | 5 张参考 | 16 张参考 | 25 张参考 | 32 张参考 |
|---|---:|---:|---:|---:|
| OnePose++ | 1.1 | 31.4 | 48.6 | 55.0 |
| Gen6D† | 18.0 | — | 58.6 | — |
| Gen6D‡ | — | 29.1 | — | 49.4 |
| Cas6D | — | 32.4 | — | 53.9 |
| GS-Pose init | 4.5 | 15.6 | 21.1 | 23.4 |
| GS-Pose | 25.6 | 62.1 | 69.4 | 74.5 |
| 本文方法 | 53.1 | 60.4 | 65.9 | 69.2 |

仅使用五张参考图像时，本方法优于所有对比方法，性能达到第二名 GS-Pose 的两倍。随着参考图像增加，本方法总体性能接近 GS-Pose，同时优于其初始结果。值得注意的是，与 GS-Pose 在细化阶段每张查询图像耗时 0.96 秒相比，本方法快 40 倍以上。

在 Occluded LINEMOD 上，如表 2 所示，本方法在所有设置下均优于 OnePose++ 和 Gen6D，表明 OnePose 系列在处理遮挡时存在局限。即使 Gen6D 在 LINEMOD 子集上微调并获得真实检测结果，它在遮挡场景中仍面临显著挑战。另外，本方法在一些物体上达到与实例级方法 PVNet 相当甚至更高的性能，展现了其作为基于 RGB 的可泛化方法处理遮挡的稳健性。

**图 3：Occluded-LINEMOD 与 YCB-Video 上的定性比较。** 绿色框表示真值，蓝色框表示预测结果。定量与定性结果均展示了本方法在遮挡条件下的有效性。比较方法为 Gen6D、OnePose++ 和本文方法。

**表 2：Occluded LINEMOD 上的比较。** † 表示向 Gen6D 提供真实检测结果。原表用斜体标出进入 Gen6D 训练集的物体，并标出前三名；下表保留物体名、星号及数值，未重现斜体标记。星号沿用原表。

ADD(s)-0.1d：

| 参考数 | 方法 | ape | can | cat | driller | duck | eggbox* | glue* | holepuncher | 平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | PoseCNN | 9.6 | 45.2 | 0.9 | 41.4 | 19.6 | 22.0 | 38.5 | 22.1 | 24.9 |
| — | PVNet | 15.8 | 63.3 | 16.7 | 25.2 | 65.7 | 50.2 | 49.6 | 39.7 | 40.8 |
| 5 | OnePose++ | — | 0.0 | 0.0 | 0.0 | — | — | — | 0.0 | — |
| 5 | Gen6D | 5.2 | 8.6 | 0.3 | 0.3 | 5.2 | 5.8 | 7.4 | 4.8 | 4.7 |
| 5 | Gen6D† | 6.4 | 8.8 | 0.5 | 4.8 | 5.4 | 29.6 | 15.0 | 11.1 | 10.2 |
| 5 | 本文方法 | 21.5 | 48.4 | 4.0 | 56.9 | 16.1 | 34.7 | 20.6 | 9.2 | 26.5 |
| 25 | OnePose++ | 0.0 | 0.2 | 2.6 | 0.0 | 1.0 | 21.5 | 2.6 | 2.2 | 3.8 |
| 25 | Gen6D | 12.3 | 23.4 | 5.6 | 2.5 | 17.2 | 25.2 | 13.4 | 28.5 | 16.0 |
| 25 | Gen6D† | 14.2 | 29.9 | 7.4 | 21.1 | 15.4 | 45.8 | 31.1 | 38.9 | 25.5 |
| 25 | 本文方法 | 21.7 | 61.6 | 54.7 | 53.1 | 30.4 | 27.6 | 57.5 | 41.9 | 43.6 |

Proj-2d@5px：

| 参考数 | 方法 | ape | can | cat | driller | duck | eggbox* | glue* | holepuncher | 平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | PoseCNN | 34.6 | 15.1 | 10.4 | 31.8 | 7.4 | 1.9 | 13.8 | 23.1 | 17.2 |
| — | PVNet | 69.1 | 86.1 | 65.1 | 61.4 | 73.1 | 8.4 | 55.4 | 69.8 | 61.1 |
| 5 | OnePose++ | — | 0.0 | 0.0 | 0.0 | — | — | — | 0.0 | — |
| 5 | Gen6D | 13.4 | 12.2 | 0.5 | 0.4 | 9.6 | 1.7 | 5.7 | 7.0 | 6.3 |
| 5 | Gen6D† | 23.4 | 13.2 | 2.1 | 3.7 | 12.1 | 8.5 | 14.1 | 15.6 | 11.6 |
| 5 | 本文方法 | 41.7 | 31.2 | 10.4 | 8.8 | 42.8 | 1.8 | 26.6 | 14.5 | 21.9 |
| 25 | OnePose++ | 0.0 | 0.0 | 8.9 | 0.0 | 6.3 | 7.2 | 0.0 | 2.5 | 3.1 |
| 25 | Gen6D | 37.3 | 28.9 | 18.6 | 2.2 | 36.5 | 3.9 | 11.4 | 48.0 | 23.4 |
| 25 | Gen6D† | 53.8 | 41.1 | 28.9 | 18.1 | 41.3 | 7.7 | 32.3 | 65.2 | 36.1 |
| 25 | 本文方法 | 59.7 | 58.9 | 54.7 | 27.8 | 56.3 | 1.9 | 53.6 | 71.4 | 47.9 |

### 4.3 YCB-Video 上的结果

YCB-Video 因遮挡、运动模糊和光照变化而具有挑战性。对于稀疏数据库，表 3 显示，本方法仅用五张参考图像即可显著优于 OnePose++。使用 16 张参考图像时，性能进一步提升，在 ADD-S 和 ADD 两项指标上均比使用真实检测的 Gen6D 高 11.4 个百分点。

**表 3：YCB-Video 稀疏数据库上的性能。** † 表示向 Gen6D 提供真实检测结果；‡ 表示使用真实掩码去除背景以改善 Gen6D 检测。每格按“ADD-S / ADD”记录，指标依第 4.1 节为 AUC。原表中缺失项保留为“—”。

| 物体 | Gen6D‡，16 张 | Gen6D†，16 张 | OnePose++，16 张 | 本文，5 张 | 本文，16 张 |
|---|---|---|---|---|---|
| 002 master chef can | 75.6 / 23.8 | 69.5 / 29.4 | 46.1 / 17.4 | 77.6 / 28.9 | 71.4 / 22.3 |
| 003 cracker box | 7.2 / 1.7 | 41.5 / 17.2 | 6.9 / 0.7 | 72.3 / 45.8 | 69.1 / 36.3 |
| 004 sugar box | 10.4 / 4.6 | 54.8 / 26.2 | 1.0 / 0.1 | 47.6 / 17.2 | 50.9 / 21.9 |
| 005 tomato soup can | 63.3 / 36.6 | 58.6 / 34.5 | — | 60.4 / 16.6 | 81.5 / 51.2 |
| 006 mustard bottle | 39.9 / 19.4 | 86.8 / 57.2 | 46.7 / 30.6 | 84.1 / 50.1 | 87.4 / 76.4 |
| 007 tuna fish can | 87.7 / 50.7 | 85.2 / 47.8 | 0.0 / 0.0 | 36.9 / 13.7 | 77.4 / 48.3 |
| 008 pudding box | 19.1 / 1.9 | 50.9 / 28.2 | 0.0 / 0.0 | 90.5 / 80.8 | 76.3 / 66.0 |
| 009 gelatin box | 40.8 / 19.3 | 60.3 / 38.7 | 0.5 / 0.2 | 54.7 / 34.0 | 90.2 / 81.9 |
| 010 potted meat can | 54.7 / 31.8 | 58.5 / 40.2 | — | 63.4 / 52.7 | 67.5 / 49.6 |
| 011 banana | 10.7 / 4.0 | 37.2 / 6.3 | 0.0 / 0.0 | 32.4 / 5.1 | 30.5 / 7.4 |
| 019 pitcher base | 44.3 / 12.0 | 80.3 / 58.9 | 27.1 / 15.9 | 80.2 / 60.7 | 70.3 / 40.7 |
| 021 bleach cleanser | 18.6 / 11.7 | 61.6 / 39.6 | 42.0 / 32.2 | 58.1 / 40.9 | 84.9 / 73.8 |
| 024 bowl | 13.3 / 3.9 | 45.9 / 5.7 | 10.5 / 0.3 | 40.0 / 3.2 | 35.2 / 4.8 |
| 025 mug | 73.7 / 29.3 | 72.7 / 41.1 | 3.7 / 0.9 | 88.9 / 76.4 | 83.1 / 65.8 |
| 035 power drill | 5.6 / 0.9 | 39.7 / 9.2 | 22.4 / 12.8 | 57.8 / 42.2 | 60.6 / 38.3 |
| 036 wood block | 10.9 / 1.8 | 16.1 / 1.4 | 11.1 / 0.7 | 33.6 / 2.8 | 20.5 / 0.3 |
| 037 scissors | 1.3 / 0.1 | 39.3 / 17.8 | 0.0 / 0.0 | 16.9 / 7.1 | 17.5 / 4.1 |
| 040 large marker | 30.8 / 20.7 | 49.4 / 39.0 | 0.0 / 0.0 | 75.0 / 61.9 | 68.3 / 56.4 |
| 051 large clamp | 28.7 / 7.2 | 53.2 / 17.1 | 1.7 / 0.2 | 68.7 / 26.5 | 66.8 / 24.7 |
| 052 extra large clamp | 6.5 / 2.1 | 36.8 / 8.3 | 4.4 / 0.4 | 49.7 / 5.9 | 63.2 / 22.4 |
| 061 foam brick | 49.2 / 22.9 | 60.4 / 36.9 | 0.0 / 0.0 | 65.3 / 29.1 | 51.8 / 22.5 |
| 平均 | 33.0 / 14.6 | 55.2 / 28.6 | 11.8 / 5.9 | 59.8 / 31.8 | 66.6 / 40.0 |

另外两种参考数据库的平均比较结果见表 4。为更好地展示本方法的优势，我们与使用密集参考图像的 Gen6D、OnePose++ 比较，它们均从数据库中采样 200 张图像；详细结果见补充材料。虽然 OnePose++ 在密集数据库上表现更好，但 Gen6D 对遮挡和低质量参考视角较敏感，因此没有从密集采样中获益。相比之下，本方法只使用五个参考视角仍能保持稳定表现，而且最小遮挡参考数据库进一步提高了准确性。

**表 4：不同参考数据库上的 YCB-Video 平均性能。** 报告 ADD 和 ADD-S。Gen6D† 获得真实检测结果。

| 参考数据库 | 参考图像数 | 方法 | ADD | ADD-S |
|---|---:|---|---:|---:|
| 最大重叠 | 200 | OnePose++ | 24.5 | 43.3 |
| 最大重叠 | 200 | Gen6D | 14.6 | 29.1 |
| 最大重叠 | 200 | Gen6D† | 22.9 | 51.1 |
| 最大重叠 | 5 | 本文方法 | 35.4 | 65.6 |
| 最小遮挡 | 200 | OnePose++ | 22.6 | 41.7 |
| 最小遮挡 | 200 | Gen6D | 12.2 | 26.7 |
| 最小遮挡 | 200 | Gen6D† | 20.2 | 50.2 |
| 最小遮挡 | 5 | 本文方法 | 37.8 | 66.9 |

### 4.4 OnePose-LowTexture 上的结果

我们也在包含无纹理物体的 OnePose-LowTexture 数据集上进行实验，见表 5。本方法只使用 10 张参考图像，就优于 OnePose++ 和 Gen6D；10 张也是 OnePose++ 能够重建所有物体所需的最少数量。本方法与使用完整参考数据库的结果相比也保持竞争力。

Gen6D 没有在这一以物体为中心的数据集上训练；参考与查询图像中的物体尺度相近，使其检测困难，即使使用真实检测结果，性能仍然较低。OnePose 与 OnePose-LowTexture 的更多结果见补充材料。

**表 5：OnePose-LowTexture 上的性能比较。指标为 ADD-0.1d。** Full 表示完整参考集。列名为物体编号。

| 方法 | 参考图像数 | 0700 | 0706 | 0714 | 0721 | 0727 | 0732 | 0736 | 0740 | 平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PVNet | — | 12.3 | 90.0 | 68.1 | 67.6 | 95.6 | 57.3 | 61.3 | 49.3 | 62.7 |
| OnePose++ | Full | 89.5 | 99.1 | 97.2 | 92.6 | 98.5 | 79.5 | 97.2 | 57.6 | 88.9 |
| OnePose++ | 10 | 44.9 | 64.9 | 56.5 | 78.8 | 88.1 | 56.0 | 71.0 | 3.7 | 58.0 |
| Gen6D† | 10 | 16.0 | 11.7 | 10.6 | 32.3 | 29.0 | 20.5 | 23.9 | 25.4 | 21.2 |
| 本文方法 | 10 | 71.4 | 85.6 | 84.4 | 88.8 | 96.2 | 92.6 | 96.1 | 77.6 | 86.6 |

### 4.5 分析

**参考图像数量的影响。** 我们在 LINEMOD 上，使用 Proj2D@5px、ADD-0.1d 和 ADDs-0.1d 指标评估参考图像数量变化的影响。在该实验中，我们不进行任何选择，而是直接将不同数量的参考图像输入网络，以突出数量本身的作用。如图 4 所示，即使只有两张参考图像，也能估计粗略物体位姿。参考图像增多时，包围盒估计 Transformer 能有效利用额外信息，更准确地预测角点。

**图 4：参考视角数量变化时，LINEMOD 上的性能趋势。** 横轴为参考视角数量，纵轴为成功率；展示 ADD-0.1d、ADDs-0.1d 和 Proj2D@5px 三条曲线。

**带噪三维包围盒的影响。** 我们使用 DUSt3R 恢复物体包围盒，并通过多视角检测过滤点。这一过程可能引入噪声，因此我们评估方法对不同包围盒质量的稳健性。如图 5 所示，本方法对不同包围盒均保持稳健。此外，在 LINEMOD 上使用不同包围盒来源的实验（表 6）也证实了该方法对带噪包围盒估计的适应能力。

**图 5：使用不同包围盒的定性结果。** 左上为真实物体包围盒；右上及其余图像为使用第 4.1 节介绍的三种参考数据库，各取五张参考图像，通过 DUSt3R 恢复的包围盒。

**表 6：不同包围盒来源下，LINEMOD 上的性能。** GT 表示真值。

| 包围盒来源 | 参考图像数 | ADD(s)-0.1d | Proj2D@5px |
|---|---:|---:|---:|
| OnePose++ | 5 | 49.4 | 52.7 |
| DUSt3R | 5 | 51.3 | 43.8 |
| GT | 5 | 49.3 | 51.8 |

**真实数据训练的影响。** 如表 7 所示，加入 OnePose 这样简单的、以物体为中心的数据集进行训练，可以有效增强本方法在其他困难真实数据集上的性能。

**表 7：不同训练数据设置下，LINEMOD 上的性能。**

| 是否使用真实数据训练 | 参考图像数 | ADD(s)-0.1d | Proj2D@5px |
|---|---:|---:|---:|
| 否 | 5 | 40.6 | 25.4 |
| 是 | 5 | 51.3 | 43.8 |

**运行时间。** 在 Intel i7-13700KF CPU 和 NVIDIA RTX 4090 GPU 上，使用五张参考图像进行位姿推理时，本方法处理一张查询图像约需 17 毫秒。其中 DINOv2 编码器约需 5.7 毫秒，包围盒解码器约需 11.1 毫秒，位姿恢复约需 0.15 毫秒，验证了其实时性能。

推理前，使用 DUSt3R 重建五张参考图像约需 5.3 秒，包括模型加载、重建和点云过滤。这个过程离线进行一次，不影响实时推理。

## 5. 结论

本文提出了 BoxDreamer：一种将物体包围盒角点用作中间表示的可泛化物体位姿估计新框架。大量实验表明，BoxDreamer 显著改善了位姿估计性能，尤其是在稀疏视角输入和遮挡等困难场景中。此外，高效的 Transformer 解码器实现了实时性能，增强了方法的实用性。

**局限性与未来工作。** 虽然 BoxDreamer 取得了有前景的结果，但估计对称物体位姿仍然困难。该框架可以处理密集视角输入，但这会提高显存占用。未来的一个关键方向是，在不增加时间或显存开销的前提下，利用密集输入提高精度。另外，由于物体检测与位姿估计目前分开处理，将二维物体检测与三维包围盒角点预测整合起来，会进一步增强实用性。

## 致谢

本工作部分得到以下项目和机构支持：湘江实验室重大项目（编号 24XJJCYJ01004）、国家自然科学基金（编号 62402427、U24B20154）、浙江省自然科学基金（编号 LR25F020003）、蚂蚁研究、浙江大学教育基金会启真学者基金，以及浙江大学信息技术中心与 CAD&CG 国家重点实验室。

作者还感谢 RoboSense（速腾聚创科技有限公司）提供 AC1（Active Camera）传感器，该传感器用于采集部分实验数据和开展验证。

## 译者注：原论文与当前 DJI ACTION4 项目的区别

以下内容为实现说明，**不属于原文**。

1. **输入任务不同。** BoxDreamer 使用参考图像、参考包围盒角点信息和查询图像，处理未见物体；当前项目希望仅输入一张 RGB，处理固定型号的两个 DJI ACTION4。移除参考分支后，属于针对任务的改造，不能称为原网络的完整复现。
2. **热力图公式不同。** 原文指数分子是 $\sqrt{(x-x_i)^2+(y-y_i)^2}$；当前 `generate_corner_heatmaps.py` 使用普通二维高斯核，其指数分子是 $(x-x_i)^2+(y-y_i)^2$，且默认固定 $\sigma=2.5$。两种设计不能混称为相同公式。
3. **角点坐标损失不能简单通过 argmax 反传。** 原文给出细粒度坐标损失，但正文未详细说明热力图到坐标的可微提取实现。常规 argmax 不可微；落实细粒度损失前应核对官方实现，或明确采用适当的可微坐标提取方法。当前每通道双峰，还涉及实例对应关系，直接对整张热力图做一次 soft-argmax 也可能得到两个目标之间的位置。
4. **“框不精确也稳健”不代表任意错误标注都可用。** 原文使用同一三维框生成参考投影并建立查询对应关系。该结果不能证明当前渲染模型与标注之间若有尺度、姿态或坐标系错误也没有影响。
5. **论文不直接给出双实例角点分组方案。** 当前项目中每个通道可能有两个峰，需要单独设计实例分组，或先检测、裁剪每个目标再预测角点。
6. **量化误差不等于渲染几何正确性。** Heatmap 峰接近由 R/t/K 投影得到的标签，只验证了二者一致；要判断标签是否对应渲染机身，还需检查模型坐标、包围盒尺寸与渲染位姿。包围盒角点也不必落在可见轮廓上。
7. **符号与实现表述。** 原文在 token 数公式中用 $N$ 表示参考图像数，在粗损失中再次用 $N$ 表示热力图求平均的项数；阅读时应按上下文区分。原文将全自注意力模块称作 decoder，不能仅凭这个名称假定其使用标准 encoder–decoder cross-attention 结构。

## 参考文献（保留英文书目信息）

以下保留原文编号、作者、标题和发表信息，修复 PDF 换行；条目末尾的数字为原论文的文内页码回引。原 PDF 中可见的书目信息缺漏不补写。


[1] Eric Brachmann, Alexander Krull, Frank Michel, Stefan Gumhold, Jamie Shotton, and Carsten Rother. Learning 6d object pose estimation using 3d object coordinates. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part II 13, pages 536–551. Springer, 2014. 2

[2] Dingding Cai, Janne Heikkilä, and Esa Rahtu. Gs-pose: Generalizable segmentation-based 6d object pose estimation with 3d gaussian splatting. 2024. 1, 3

[3] Junhao Cai, Yisheng He, Weihao Yuan, Siyu Zhu, Zilong Dong, Liefeng Bo, and Qifeng Chen. Ov9d: Openvocabularycategory-level9dobjectposeandsizeestimation. ArXiv, abs/2403.12396, 2024. 2

[4] Pedro Castro and Tae-Kyun Kim. Posematcher: One-shot 6d object pose estimation by deep feature matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2148–2157, 2023. 1

[5] Hansheng Chen, Pichao Wang, Fan Wang, Wei Tian, Lu Xiong, and Hao Li. Epro-pnp: Generalized end-to-end probabilistic perspective-n-points for monocular object pose estimation. InIEEEConferenceonComputerVisionandPattern Recognition (CVPR), 2022. 2

[6] Tianxing Chen, Yao Mu, Zhixuan Liang, Zanxin Chen, Shijia Peng, Qiangyu Chen, Mingkun Xu, Ruizhen Hu, Hongyuan Zhang, Xuelong Li, et al. G3flow: Generative 3d semantic flow for pose-aware and generalizable object manipulation. arXiv preprint arXiv:2411.18369, 2024. 1

[7] Yamei Chen, Yan Di, Guangyao Zhai, Fabian Manhardt, Chenyangguang Zhang, Ruida Zhang, Federico Tombari, Nassir Navab, and Benjamin Busam. Secondpose: Se(3)consistent dual-stream feature fusion for category-level pose estimation. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9959–9969, 2023. 2

[8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. arXiv preprint arXiv:2212.08051, 2022. 5

[9] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A universe of 10m+ 3d objects. arXiv preprint arXiv:2307.05663, 2023. 5

[10] Yan Di, Ruida Zhang, Zhiqiang Lou, Fabian Manhardt, Xiangyang Ji, Nassir Navab, and Federico Tombari. Gpv-pose: Category-level object pose estimation via geometry-guided point-wise voting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6781–6791, 2022. 2

[11] Ross Girshick. Fast r-cnn. In International Conference on Computer Vision (ICCV), 2015. 5

[12] Xingyi He, Jiaming Sun, Yuang Wang, Di Huang, Hujun Bao, and Xiaowei Zhou. Onepose++: Keypoint-free oneshot object pose estimation without CAD models. In Advances in Neural Information Processing Systems, 2022. 1, 3, 5, 6

[13] Xingyi He, Jiaming Sun, Yifan Wang, Sida Peng, Qixing Huang, Hujun Bao, and Xiaowei Zhou. Detector-free structure from motion. CVPR, 2024. 3

[14] Stefan Hinterstoisser, Vincent Lepetit, Slobodan Ilic, Stefan Holzer, Gary Bradski, Kurt Konolige, and Nassir Navab. Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes. In Asian conference on computer vision, pages 548–562. Springer, 2012. 2

[15] Peter Hönig, Stefan Thalhammer, and Markus Vincze. Improving 2d-3d dense correspondences with diffusion models for 6d object pose estimation. ArXiv, abs/2402.06436, 2024. 2

[16] Cheng-Chun Hsu, Bowen Wen, Jie Xu, Yashraj Narang, Xiaolong Wang, Yuke Zhu, Joydeep Biswas, and Stan Birchfield. Spot: Se (3) pose trajectory diffusion for object-centric manipulation. arXiv preprint arXiv:2411.00965, 2024. 1

[17] Takuya Ikeda, Sergey Zakharov, Tianyi Ko, Muhammad Zubair Irshad, Robert Lee, Katherine Liu, Rares Ambrus, and Koichi Nishiwaki. Diffusionnocs: Managing symmetry and uncertainty in sim2real multi-modal categorylevel pose estimation. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7406–7413. IEEE, 2024. 2

[18] Wadim Kehl, Fabian Manhardt, Federico Tombari, Slobodan Ilic, and Nassir Navab. Ssd-6d: Making rgb-based 3d detection and 6d pose estimation great again. In Proceedings of the IEEE international conference on computer vision, pages 1521–1529, 2017. 2

[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 3

[20] Hei Law and Jia Deng. Cornernet: Detecting objects as paired keypoints. In Proceedings of the European Conference on Computer Vision (ECCV), 2018. 4

[21] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, 2024. 3

[22] Yi Li, Gu Wang, Xiangyang Ji, Yu Xiang, and Dieter Fox. DeepIM: Deep iterative matching for 6D pose estimation. In European Conference Computer Vision (ECCV), 2018. 2

[23] Zhigang Li, Gu Wang, and Xiangyang Ji. Cdpn: Coordinates-based disentangled pose network for real-time rgb-based 6-dof object pose estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. 2

[24] Xiao Lin, Minghao Zhu, Ronghao Dang, Guangliang Zhou, Shaolong Shu, Feng Lin, Chengju Liu, and Qi Chen. Clipose: Category-level object pose estimation with pre-trained vision-language knowledge. IEEE Transactions on Circuits and Systems for Video Technology, 34:9125–9138, 2024. 2

[25] Yuan Liu, Yilin Wen, Sida Peng, Cheng Lin, Xiaoxiao Long, Taku Komura, and Wenping Wang. Gen6d: Generalizable model-free 6-dof object pose estimation from rgb images. In ECCV, 2022. 1, 2, 5

[26] Luqing Luo, Shichu Sun, Jiangang Yang, Linfang Zheng, Jinwei Du, and Jian Liu. Object gaussian for monocular 6d pose estimation from sparse views. arXiv preprint arXiv:2409.02581, 2024. 3

[27] Van Nguyen Nguyen, Thibault Groueix, Georgy Ponimatkin, Yinlin Hu, Renaud Marlet, Mathieu Salzmann, and Vincent Lepetit. Nope: Novel object pose estimation from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17923–17932, 2024. 3

[28] Markus Oberweger, Mahdi Rad, and Vincent Lepetit. Making deep heatmaps robust to partial occlusions for 3d object pose estimation. In Proceedings of the European conference on computer vision (ECCV), pages 119–134, 2018. 2

[29] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang- WenLi, WojciechGaluba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 4, 5

[30] Panwang Pan, Zhiwen Fan, Brandon Y Feng, Peihao Wang, Chenxin Li, and Zhangyang Wang. Learning to estimate 6dof pose from limited data: A few-shot, generalizable approach using rgb images. In 2024 International Conference on 3D Vision (3DV), pages 1059–1071. IEEE, 2024. 1, 3, 5

[31] Kiru Park, Timothy Patten, and Markus Vincze. Pix2pose: Pixel-wise coordinate regression of objects for 6d pose estimation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7668–7677, 2019. 2

[32] Keunhong Park, Arsalan Mousavian, Yu Xiang, and Dieter Fox. Latentfusion: End-to-end differentiable reconstruction and rendering for unseen object pose estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020. 3

[33] Sida Peng, Yuan Liu, Qixing Huang, Xiaowei Zhou, and Hujun Bao. Pvnet: Pixel-wise voting network for 6dof pose estimation. In CVPR, 2019. 2, 5

[34] Mahdi Rad and Vincent Lepetit. Bb8: A scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth. In Proceedings of the IEEE international conference on computer vision, pages 3828–3836, 2017. 2, 5

[35] Jiaming Sun, Zihao Wang, Siyu Zhang, Xingyi He, Hongcheng Zhao, Guofeng Zhang, and Xiaowei Zhou. OnePose: One-shot object pose estimation without CAD models. CVPR, 2022. 1, 2, 3, 5, 6

[36] Yujing Sun, Caiyi Sun, Yuan Liu, Yuexin Ma, and Siu ming Yiu. Extreme two-view geometry from object poses with diffusion models. ArXiv, abs/2402.02800, 2024. 3

[37] Yujing Sun, Caiyi Sun, Yuan Liu, Yuexin Ma, and Siu Ming Yiu. Generalizable single-view object pose estimation by two-side generating and matching. arXiv preprint arXiv:2411.15860, 2024. 3

[38] He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, and Leonidas J Guibas. Normalized object coordinate space for category-level 6d object pose and size estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2642–2651, 2019. 2

[39] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21686–21697, 2024. 3

[40] Pengyuan Wang, Takuya Ikeda, Robert Lee, and Koichi Nishiwaki. Gs-pose: Category-level object pose estimation via geometric and semantic correspondence. In European Conference on Computer Vision, pages 108–126. Springer, 2024. 2, 3

[41] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 3

[42] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17868– 17879, 2024. 3, 5

[43] Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes. 2018. 2, 5

[44] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3485–3492. IEEE, 2010. 5

[45] JianingYang, AlexanderSax, KevinJ.Liang, MikaelHenaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 3

[46] Yang You, Ruoxi Shi, Weiming Wang, and Cewu Lu. Cppf: Towardsrobustcategory-level9dposeestimationinthewild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6866–6875, 2022. 2

[47] Sergey Zakharov, Ivan Shugurov, and Slobodan Ilic. Dpod: 6d pose object detector and refiner. In International Conference on Computer Vision (ICCV), 2019. 2

[48] Jiyao Zhang, Min-Yu Wu, and Hao Dong. Genpose: Generative category-level object pose estimation via diffusion models. ArXiv, abs/2306.10531, 2023. 2

[49] Jiyao Zhang, Weiyao Huang, Bo Peng, Mingdong Wu, Fei Hu, Zijian Chen, Bo Zhao, and Hao Dong. Omni6dpose: A benchmark and model for universal 6d object pose estimation and tracking. In European Conference on Computer Vision, 2024. 2

[50] Mengchen Zhang, Tong Wu, Tai Wang, Tengfei Wang, Ziwei Liu, and Dahua Lin. Omni6d: Large-vocabulary 3d object dataset for category-level 6d object pose estimation. ArXiv, abs/2409.18261, 2024. 2

[51] Chen Zhao, Tong Zhang, and Mathieu Salzmann. 3d-aware hypothesis & verification for generalizable relative object pose estimation. arXiv preprint arXiv:2310.03534, 2023. 3

[52] Chen Zhao, Yinlin Hu, and Mathieu Salzmann. Locposenet: Robust location prior for unseen object pose estimation. International Conference on 3D Vision, 2024. 1, 3, 5

[53] Chen Zhao, Tong Zhang, Zheng Dang, and Mathieu Salzmann. Dvmnet: Computing relative pose for unseen objects beyond hypotheses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20485–20495, 2024. 3
