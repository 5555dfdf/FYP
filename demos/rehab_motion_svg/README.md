# 康复动作 SVG 动画

双击公开的 **`demo.html`**，用Chrome或Edge打开即可，离线可用，无需启动服务。GitHub仓库页面会显示HTML源码，需先下载文件再用浏览器打开；仓库首页使用GIF直接播放预览，并提供原生SVG下载。

## 展示内容

- 肩后伸：标准、躯干前倾、屈肘、耸肩、躯干旋转、手臂外偏。
- 肩外展：标准、外展幅度不足、屈肘、耸肩、手臂前偏、头部侧倾。
- 标准与错误并排同步，绿色表示标准骨架，橙色突出所选错误对应部位。
- 可播放/暂停、拖动进度、从头播放、选择0.5/1/1.5倍速，切换正面/斜侧/侧面，叠加同进度标准姿态。
- 桌面与手机布局，支持空格播放/暂停，并尊重系统减少动态效果设置。

## 两种来源

**动作原理示意（默认）**：使用33点骨架的运动学示意，控制标准动作与某一种错误的区别，便于观察。绘制幅度服务于表达，不是患者处方或实测动作。脸部轮廓和颈部连接是由关键点推导的显示辅助线。

**真实数据回放（本地生成的`index.html`）**：读取本地REHAB26现有标注，使用同一参与者Vasil的12次动作。参与者按整体关节可见度选择，各类别从单一错误标签的片段中选取最接近该组中位几何轨迹的示例；不依据模型预测挑选。三路骨架经过固定yaw/尺度/进度对齐后等权融合，仅显示时进行sigma=1.2帧的高斯平滑。此处保留估计骨架的偏差，不把它等同于理想姿态。

真实数据的选取记录在 `source_manifest.json`；橙色部位依据所选错误标签预先映射，不是模型实时检测。真实坐标、包含真实坐标的完整`index.html`和`recorded_svg/`保留本地；公开页面与SVG/GIF均使用自行构建的动作示意。此演示没有改写训练数据、模型、实验配置或成绩。

## 文件

| 文件 | 用途 |
|---|---|
| `demo.html` | 公开交互页面，仅内嵌人工示意骨架 |
| `index.html`（本地生成） | 完整交互页面，包含示意与真实回放 |
| `shoulder_extension_comparison.svg` | 肩后伸的标准与五类错误，原生SVG/SMIL、5秒循环 |
| `shoulder_abduction_comparison.svg` | 肩外展的标准与五类错误，原生SVG/SMIL、5秒循环 |
| `svg/` | 12个单独的示意动画，可各自复用 |
| `recorded_svg/`（本地生成） | 两个动作的真实标注片段SVG回放 |
| `preview_extension.gif` / `preview_abduction.gif` | 从SVG播放器渲染的动图预览 |
| `motion_data.json`（本地生成） | 真实数据的显示坐标 |
| `schematic_data.json` | 示意模式的显示坐标 |
| `qa_report.json` / `svg_verification.json` | 浏览器交互和SVG播放核验 |

独立SVG无脚本依赖，直接在Chrome/Edge中打开即可循环播放；有些文档软件只显示SVG的静态首帧，需用浏览器观看动画。

## 重新生成

仅重建公开演示，不需要原始数据：

```powershell
python demos/rehab_motion_svg/build.py --public-only
```

取得原始数据、完成实验预处理后，可生成本地真实回放：

```powershell
& 'D:\conda\shell\condabin\conda-hook.ps1'
conda activate fyp
python demos/rehab_motion_svg/export_data.py
python demos/rehab_motion_svg/make_schematic.py
python demos/rehab_motion_svg/build.py
```

生成依赖NumPy和SciPy。`build.py`也可直接基于已保存的两份JSON重建，无需重新读取原始数据。

浏览器核验与动图预览另用Playwright、Pillow，以及本机Chrome：

```powershell
python demos/rehab_motion_svg/check_player.py --public
python demos/rehab_motion_svg/render_previews.py
```
