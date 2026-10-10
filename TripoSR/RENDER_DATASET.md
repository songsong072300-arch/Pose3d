# DJI Action 4 目标域合成数据渲染

## 第一视角动作模板（2026-10-09）

目标域带手臂的样本默认使用 `--pose-profile mixed`：抬手展示 raised（25%）、
桌面操作 work（50%）、收拢 fold（25%）。这些模板同时改变观察位置、双腕高度、
间距和视线，不再仅通过改变 look-at 模拟抬手。可指定单一模板逐类检查。
context dropout、透明目标层和 generic 场景继续使用原布局。

在 TripoSR 目录运行以下命令，输出到新目录，不追加 BOP 数据：

```powershell
& 'D:\code_work_space\vscode_Work_space\Pose3d\.conda-envs\pose3d\Scripts\blenderproc.exe' run render_dataset.py 12 `
  --dataset-dir dataset\lm --preview-dir preview_ego_raised `
  --pose-profile raised --context-dropout-probability 0 --render-device cpu `
  --preview-yolo-labels --width 960 --height 720 --samples 32 `
  --seed 20261009 --no-sensor-effects
```

分别替换 pose-profile 和输出目录检查 work/fold；输出目录必须是新的或空的。
`metadata/*.json` 保存动作类型、双腕位置、观察位姿、FOV、机身投影框和偏小目标。
`--min-target-width-fraction 0.075` 是审计阈值，不自动拒绝或放大样本。
投影框是未考虑遮挡的机身几何框；训练框仍使用 `labels` 中分割得到的可见框。

桌布网格已改成表面程序纹理，避免细立体网格的阴影和闪烁；背屏加入完全合成的
程序颜色纹理与轻微发光，并随机改变发光强度。该屏幕纹理不是实际场景回放。
当前手和衣服仍是几何代理，尚未替换成带骨骼人体或真实布料资产。
先审计姿态和目标裁剪，再打开成像退化、扩大数据量。
`--render-device cpu` 在本机 640×480、16 samples 的预览中 RGB 约 2–4 秒、
分割约 0.3–0.6 秒；auto 的首轮 OptiX 路径耗时显著更长，默认 auto 保留不变。

本轮实际检查的最终预览：`preview_ego_raised_batch`、`preview_ego_work_calibrated`、
`preview_ego_fold_calibrated`，每类 3 张，共 18 个分割检测框。
带框审计图位于 `../outputs/yolo/ego_raised_audit.jpg`、
`../outputs/yolo/ego_work_calibrated_audit.jpg`、
`../outputs/yolo/ego_fold_calibrated_audit.jpg`。
抬手投影宽度约 10%–18%，校准后桌面/收拢约 5%–12%。这些是几何审计结果，
未训练检测器，也不代表真实域性能已改善。

本渲染器面向 `frames` 中的第一视角腕戴运动相机场景。当前方案不依赖新增实拍训练图，而是通过消除合成数据中的固定捷径，缩小纯合成训练与真实视频之间的差距。

重点不是继续增加桌子、箱子等背景几何，而是同时随机化相机本体细节、目标数量、腕部上下文、构图、逐帧 FOV 和视频成像退化。

## 先渲染 24 张预览

### 真实服装资产与 frames_soft 光照

`target` 场景现在会从 `Pose3d/3D资产` 载入三个牛仔服装 OBJ，并按约 68%/19%/13% 的比例选择 `牛仔模型`、`牛仔模型_2` 和 `牛仔外套_3`。通过 `--garment-variant main|second|third` 可以固定某一姿态做检查。`左侧杂物` 作为单个真实 OBJ 固定放在桌子左侧的地面上，旧的黑色方块堆已移除。

默认 `--lighting-profile frames_soft` 使用大面积顶光、弱前方补光和较高的环境亮度，随机扰动限制在约 15% 内，以接近 `frames` 的灰冷、低对比度和柔和阴影。需要复现旧分布时使用 `--lighting-profile legacy`。

```powershell
& 'D:\code_work_space\vscode_Work_space\Pose3d\.conda-envs\pose3d\Scripts\blenderproc.exe' run ../../render_dataset.py 24 `
  --preview-dir ../../preview_assets_frames_soft `
  --garment-probability 1 --lighting-profile frames_soft `
  --preview-yolo-labels --render-device cpu
```

每帧 metadata 会记录服装姿态、左侧杂物、光照 profile 及实际灯光参数，便于与实拍帧逐批比较。

`work` / `fold` 动作使用俯视桌面的头部视角，相机更接近桌子前缘上方，注视桌面而不是手腕。metadata 的 `downward_pitch_degrees` 记录向下俯角（相对水平面），用于检查视角分布；`raised` 保留抬手观察的视角。

有手臂上下文时，每只手腕默认固定生成灰色腕带、绿色条状包边和顶部白色塑料件（`--wrist-device-probability 1`）。绿色边缘覆盖腕带环和顶部面板，在俯视角下也能看到；这些部件不计入相机标签。已有命令若显式指定了较低概率，需要改为 1。

预览模式只写 RGB 图片，不会向 `train_pbr` 追加数据：

```powershell
& 'D:\code_work_space\vscode_Work_space\Pose3d\.conda-envs\pose3d\Scripts\blenderproc.exe' run ../../render_dataset.py 24 `
  --preview-dir ../../preview_target_v2 `
  --seed 20260928 `
  --min-cameras 1 `
  --max-cameras 2 `
  --single-camera-probability 0.40 `
  --rear-screen-proxy
```

长时间渲染前，应确认预览中同时出现一台/两台相机、正背侧视角、中央/边缘/部分出画构图，以及背屏、标签、标记、底座和线缆。若目标仍明显偏小，不要直接生成几千张同分布图片。

## 单独生成纯合成困难负样本

下面的图片保留手臂、腕带、衣物和桌面，但隐藏所有相机：

```powershell
& 'D:\code_work_space\vscode_Work_space\Pose3d\.conda-envs\pose3d\Scripts\blenderproc.exe' run ../../render_dataset.py 300 `
  --preview-dir ../../preview_camera_free_v2 `
  --camera-free-preview `
  --seed 20260929
```

这些图片可通过 YOLO 导出器的 `--negative-dir` 加入 train 并生成空标签，用于明确抑制腕带、衣服口袋、按钮和画面边缘误检。

## 生成 BOP 正样本

```powershell
& 'D:\code_work_space\vscode_Work_space\Pose3d\.conda-envs\pose3d\Scripts\blenderproc.exe' run ../../render_dataset.py 800 `
  --seed 20261001 `
  --min-cameras 1 `
  --max-cameras 2 `
  --single-camera-probability 0.40 `
  --context-dropout-probability 0.30 `
  --garment-probability 0.35 `
  --bop-masks
```

建议使用不同 seed 生成至少 4-6 个独立 scene，而不是只生成一个很长的近重复序列。推荐让约 40% 正样本只有一台相机，约 30% 正样本不出现简化手臂，并额外生成相当于正样本量 20%-35% 的无相机困难负样本。

常用参数：

- `--front-probability 0.25 --back-probability 0.35`：其余概率为两个侧面。
- `--lower-middle-probability 0.55`：其余样本覆盖中央、左右边缘和抬手构图。
- `--occlusion-probability 0.35`：随机轻/中/重遮挡。
- `--single-camera-probability 0.40`：允许 1-2 台时，40% 正样本只有一台。
- `--context-dropout-probability 0.30`：30% 正样本不生成简化手臂上下文。
- `--garment-probability 0.35`：避免衣物与相机强绑定。
- `--rear-screen-proxy`：默认开启背屏代理；同时随机添加标签、标记和状态灯。
- `--no-sensor-effects`：关闭曝光、颜色、模糊、暗角、噪声和 JPEG 退化，仅用于消融。
- `--scene-profile generic <cc0textures-path>`：旧的 CC0 广域随机背景，仅用于对照。

不要只训练现有 `000000` 和 `000001` scene；它们属于旧的广域随机分布。新版 target-profile 数据生成后，应排除旧 scene，或只保留极少部分作为正则项。

## 替换 Action 4 模型

新版导出模型使用 `X/Y/Z = 宽/高/前后深度`，与旧 TripoSR mesh 不同。转换时需要显式指定布局：

```sh
conda run -n render python ../../prepare_bop_model.py \
  --input ../../../3D模型/base.obj \
  --texture ../../../3D模型/texture_diffuse.png \
  --source-layout action4-export \
  --origin center
```

转换器会创建带时间戳的备份，将镜头方向映射到局部 `-X`，把模型缩放为 `32.8 x 70.5 x 44.2 mm`，并复制法线、粗糙度和金属度贴图。追加新 BOP scene 前必须先预览。

## 按 scene 生成标签并导出 YOLO

```powershell
conda run -n render python ../../generate_corner_heatmaps.py . `
  --split-mode scene `
  --val-scene-count 1 `
  --test-scene-count 0 `
  --output-root corner_labels_target_v2

& 'D:\code_work_space\vscode_Work_space\Pose3d\.conda-envs\pose3d\python.exe' `
  -m pose3d.train.export_yolo_labels `
  --root TripoSR\dataset\lm `
  --labels-dir corner_labels_target_v2 `
  --negative-dir TripoSR\preview_camera_free_v2 `
  --max-negatives 300 `
  --out checkpoints\action4_yolo_target_v2 `
  --seed 20260928
```

默认的 `scene` 划分保证同一渲染序列不会跨 train/val。至少需要两个 scene 才能建立 train/val；若还需要独立合成 test，至少需要三个 scene。根目录 `camera.json` 只保存基准 FOV，逐帧实际内参以 `scene_camera.json` 为准。

不要仅凭合成 validation 的高 mAP 判断问题已经解决。最终准入标准是：在固定 `frames` 开发集上，腕带/衣物误检明显下降，同时真实相机漏检没有恶化。
