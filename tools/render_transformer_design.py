"""Render the current fusion experiment and reference Transformer diagrams as PNG and editable SVG.

Run inside `conda activate fyp`. Uses Pillow for font measurement and PNG output;
the same drawing primitives produce SVG so the two formats share one layout.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
FONT = Path("C:/Windows/Fonts/msyh.ttc")
BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")
BG = "#F4F7FB"
INK = "#162B45"
MUTED = "#51667D"
BLUE = "#285ED4"
TEAL = "#087E82"
ORANGE = "#AB6423"
PURPLE = "#7350AD"
LINE = "#C8D5E5"


class Canvas:
    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.im = Image.new("RGB", (width, height), BG)
        self.draw = ImageDraw.Draw(self.im)
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      f'<rect width="{width}" height="{height}" fill="{BG}"/>']
        self.text_extents = []

    def font(self, size=25, bold=False):
        return ImageFont.truetype(str(BOLD if bold else FONT), size)

    def rect(self, x, y, w, h, fill="#FFFFFF", border=LINE, radius=18):
        self.draw.rounded_rectangle((x, y, x+w, y+h), radius=radius, fill=fill, outline=border, width=2)
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{border}" stroke-width="2"/>')

    def text(self, x, y, text, size=25, fill=INK, bold=False):
        font = self.font(size, bold)
        # Baseline anchoring keeps raster and SVG typography aligned.
        ascent, _ = font.getmetrics()
        self.draw.text((x, y+ascent), text, font=font, fill=fill, anchor="ls")
        self.parts.append(f'<text x="{x}" y="{y+ascent}" fill="{fill}" font-size="{size}" font-weight="{700 if bold else 400}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">{html.escape(text)}</text>')
        self.text_extents.append((x, y, x+self.draw.textlength(text, font=font), y+size+8, text))

    def wrap(self, text, width, size=25, bold=False):
        font = self.font(size, bold)
        lines = []
        for paragraph in text.split("\n"):
            current = ""
            for char in paragraph:
                if current and self.draw.textlength(current+char, font=font) > width:
                    lines.append(current)
                    current = char
                else:
                    current += char
            lines.append(current)
        return lines

    def paragraph(self, x, y, text, width, size=25, fill=MUTED, bold=False, line_height=None):
        step = line_height or size+10
        lines = self.wrap(text, width, size, bold)
        for i, line in enumerate(lines):
            self.text(x, y+i*step, line, size, fill, bold)
        return y+len(lines)*step

    def box(self, x, y, w, h, title, body, color=BLUE, fill="#FFFFFF", size=25):
        self.rect(x, y, w, h, fill=fill)
        self.rect(x+18, y+24, 6, 34, fill=color, border=color, radius=3)
        title_lines = self.wrap(title, w-68, 29, True)
        for i, title_line in enumerate(title_lines):
            self.text(x+38, y+22+i*39, title_line, 29, color, True)
        end = self.paragraph(x+28, y+31+len(title_lines)*39, body, w-56, size=size)
        if end > y+h-13:
            raise ValueError(f"Text overflow in {title}: {end} > {y+h-13}")

    def arrow(self, points, color=BLUE, width=4, dashed=False):
        self.draw.line(points, fill=color, width=width, joint="curve")
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/>')
        x0, y0 = points[-2]
        x, y = points[-1]
        dx, dy = x-x0, y-y0
        norm = max((dx*dx+dy*dy)**0.5, 1)
        ux, uy = dx/norm, dy/norm
        tri = [(x,y),(x-15*ux+7*uy,y-15*uy-7*ux),(x-15*ux-7*uy,y-15*uy+7*ux)]
        self.draw.polygon(tri, fill=color)
        self.parts.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in tri)+f'" fill="{color}"/>')

    def header(self, eyebrow, title, subtitle):
        self.rect(0, 0, self.w, 188, fill=INK, border=INK, radius=0)
        self.text(64, 22, eyebrow, 21, "#9BC8FE", True)
        self.text(64, 59, title, 45, "#FFFFFF", True)
        self.text(64, 127, subtitle, 24, "#DBE7F4")

    def footer(self, text):
        self.text(64, self.h-48, text, 21, MUTED)

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        for x, y, right, bottom, text in self.text_extents:
            if x < 0 or y < 0 or right > self.w or bottom > self.h:
                raise ValueError(f"Canvas overflow: {text}")
        self.im.save(OUT / f"{name}.png")
        (OUT / f"{name}.svg").write_text("\n".join(self.parts+["</svg>"]), encoding="utf-8")
        return {"name":name,"width":self.w,"height":self.h,"text_items":len(self.text_extents)}


def experiment():
    c = Canvas(1920, 1800)
    c.header("FYP / REHAB26 / CURRENT EXPERIMENT 01", "当前实验：三视角骨架融合与跨人错误识别", "2026-11-10 · conda activate fyp · v3 已完成 984 次正式拟合 · 参考与对照分别报告")
    c.box(64, 226, 550, 286, "01  数据与监督", "肩外展 ex1 + 肩后伸 ex7，分别训练\n1111 次动作 / 3333 个相机观察 / 5 人\n同次动作三路单目骨架，每路 33 点\n五个可共存错误；正确 = 标签全 0\n动作类型、边界已知；非 RGB 输入", size=24)
    c.box(686, 226, 550, 286, "02  三路独立对齐", "每帧髋中心 + 整次动作躯干尺度\n96 帧动作进度重采样\n每路使用整次动作的固定 yaw 对齐\n保留躯干倾斜与动作内相对旋转\n没有精确时间配准或标定三角测量", color=TEAL, size=24)
    c.box(1308, 226, 550, 286, "03  融合与特征", "对比等权、visibility²、对齐 cam2\n融合后重新定中心与尺度\n1111 个样本；19 关节 xyz = 57 维\n加 12 维几何特征 → 参考输入 69 维\n可选再加 19 维可见度 → 88 维", size=24)
    c.arrow([(614,367),(686,367)])
    c.arrow([(1236,367),(1308,367)])
    c.arrow([(1583,512),(1583,548),(570,548),(570,590)])
    c.box(64, 590, 1010, 346, "04  训练：只使用当前内层 3 人", "外层留 1 人测试，剩余 4 人轮流验证；所有视角按人分组\n输入 z-score 仅在训练人拟合；参考 Transformer 为 2 层 / 64 维 / 4 头\n五个 logits → 无类别加权 BCEWithLogitsLoss → 反向传播\nGELU + Pre-LN + Dropout 0.1；AdamW lr 3e-4、衰减 1e-3\nbatch 32；5 epoch warmup + cosine；最多 80 epoch\nTCN、角度 LR / RF 及基础训练因素作为对照", size=25)
    c.box(1146, 590, 712, 346, "05  验证：选择权重、配置与阈值", "当前内层另 1 人验证，不更新权重\n验证 macro-AP 早停：patience 12\n每类阈值由验证 F1 网格选择\n四名内层验证者汇总后选择配置\n先单因素筛查，再按预定规则生成组合\n不使用外层测试者拟合标准化或阈值", color=TEAL, size=24)
    c.arrow([(1074,750),(1146,750)],color=TEAL)
    c.text(1090,701,"评估",20,TEAL)
    c.arrow([(1500,936),(1500,995),(570,995),(570,1030)],color=TEAL)
    c.text(80,960,"完成选择、额外种子与固定输入对照训练 → 再释放 v3 外层测试预测",23,TEAL,True)
    c.box(64, 1030, 1010, 291, "06  测试：四内层模型集成", "每外层平均 4 个内层模型的概率与对应验证阈值\n各个种子汇总五名测试者后计算 macro-F1 / AP\n另报逐类、逐人、正确动作误警与错误动作无警\n主选择模型完成三种子；等权 / cam2 对照仅首种子\nv2 三视角概率融合另作对照，需要旧验证 / 测试预测", color=PURPLE, size=25)
    c.box(1146, 1030, 712, 291, "07  展示与后续验证", "SVG：标准 / 错误动作示意已完成\n橙色部位由错误类型映射，未接分类器\n整次动作标签不直接支持逐帧错误定位\nRGB 上传、提取骨架与预测回放待开发\n五名志愿者上的探索仍需新人确认", color=ORANGE, size=24)
    c.arrow([(1074,1172),(1146,1172)],color=PURPLE)
    c.text(64,1365,"已观察结果：区分输入方案、模型类型与种子数量",30,INK,True)
    c.box(64, 1420, 550, 246, "肩后伸：当前最高", "等权骨架融合 + 参考 Transformer\nmacro-F1 = 0.734（仅 1 个种子）\n误警 19.1% / 漏警 9.7%\n仍待额外种子确认",size=24)
    c.box(686, 1420, 550, 246, "肩后伸：三种子最高", "Transformer + 三视角概率平均\n阈值在旧内层验证中重选\nmacro-F1 = 0.690 ± 0.029\n± 为种子样本标准差",color=TEAL,size=24)
    c.box(1308, 1420, 550, 246, "肩外展：所有模型最高", "角度 RF + 三视角概率平均\n验证重选阈值；3 个种子\nmacro-F1 = 0.641 ± 0.005\n误警 54.3%，仍需改进",color=PURPLE,size=24)
    c.footer("配置与成绩来源：v3 / BEST_SETTINGS_zh.md、protocol.final.json、best_results.json；不是临床有效性结论。")
    return c.save("rehab26_transformer_experiment")


def model():
    c = Canvas(1920, 1950)
    c.header("FYP / REHAB26 / REFERENCE MODEL 02", "已实现参考 Transformer：结构与训练机制", "每帧一个 token · 融合后参考输入 [B,96,69] · 两种动作分别训练 · 75,845 参数")
    c.box(64, 230, 1060, 191, "输入与嵌入  [B,96,69] → [B,96,64]", "对齐融合后 19 关节 xyz（57）+ 几何特征（12）；仅训练组拟合标准化\nLinear(69,64)，不加激活 + 固定正弦位置编码 + Dropout(0.1)\n位置编码保留先后次序；固定 96 帧无需 padding",size=25)
    c.arrow([(594,421),(594,472)])
    c.rect(64,472,1060,700,fill="#EBF2FF",border="#9CB8EC",radius=22)
    c.text(94,490,"Pre-LN 编码块 × 2   |   H : [B,96,64]",31,BLUE,True)
    c.arrow([(550,532),(550,551)])
    c.box(98, 551, 774, 189, "注意力子层", "LayerNorm(64) → 多头自注意力（4 头，每头 16 维）\nsoftmax(QKᵀ / √16)，沿时间 key 维度归一化\n注意力权重 dropout 0.1 → 输出投影 → 残差 dropout 0.1",size=24)
    c.rect(465,774,180,58,fill="#FFFFFF",border=BLUE,radius=12)
    c.text(491,781,"残差相加 +",27,BLUE,True)
    c.arrow([(550,740),(550,774)])
    c.arrow([(550,532),(1020,532),(1020,803),(645,803)],color=TEAL)
    c.text(901,688,"保留 H",23,TEAL,True)
    c.arrow([(550,832),(550,865)])
    c.box(98, 865, 774, 189, "前馈子层 FFN", "LayerNorm(64) → Linear(64,128) → GELU\n→ Dropout(0.1) → Linear(128,64)\n→ 残差 dropout(0.1)；形状保持 [B,96,64]",size=24)
    c.rect(465,1084,180,58,fill="#FFFFFF",border=BLUE,radius=12)
    c.text(491,1091,"残差相加 +",27,BLUE,True)
    c.arrow([(550,1054),(550,1084)])
    c.arrow([(550,848),(1054,848),(1054,1113),(645,1113)],color=TEAL)
    c.text(905,993,"保留 U",23,TEAL,True)
    c.arrow([(550,1142),(550,1222)])
    c.box(64, 1222, 1060, 191, "池化与分类头", "Final LayerNorm(64) → 时间 mean 与 max 拼接 → [B,128]\nLinear(128,32) → GELU → Dropout(0.1) → Linear(32,5)\n输出原始 logits [B,5]；两层编码器独立初始化",size=25)
    c.arrow([(550,1413),(550,1460)])
    c.box(64, 1460, 502, 273, "训练：logits → 损失", "BCEWithLogitsLoss\n五个独立标签，正确 = 全 0\n损失前不要再接 sigmoid\nbackward → clip norm 1.0\n→ AdamW.step()",color=ORANGE,size=25)
    c.box(622, 1460, 502, 273, "推理：logits → 分数", "sigmoid → 5 个错误分数\n每类阈值来自内层验证\n多标签不使用类别 softmax\n分数不默认是校准概率\neval() 关闭 dropout",color=TEAL,size=25)
    c.arrow([(550,1436),(873,1436),(873,1460)],color=TEAL)
    c.box(1186, 230, 670, 291, "三种归一化：作用不同", "① 几何：髋中心 + 躯干尺度，保留错误姿态\n② 通道 z-score：只用训练人拟合均值 / 方差\n③ LayerNorm：每帧 64 维隐藏特征内部归一化\nLayerNorm 不替代输入标准化\n参考网络不采用 BatchNorm",color=TEAL,size=24)
    c.box(1186, 560, 670, 320, "损失权重：已完成三种 BCE 对照", "L0 参考：普通 BCE，正类权重 w = 1\nL1 对照：w = clip(√(Nneg/Npos), 1, 3)\nL2 对照：w = Nneg/Npos（v1 策略）\n正负计数仅来自当前训练人\n加权可能改变分数偏置，不能默认降低误警\n不同损失的数值不直接比较优劣",color=ORANGE,size=24)
    c.box(1186, 920, 670, 320, "正则化与优化：独立控制", "Dropout 0.1：输入 / 注意力 / FFN / 残差 / 头\nAdamW：lr 3e-4；weight decay 1e-3\nbias 与 LayerNorm 参数不做权重衰减\n5 epoch warmup + cosine；batch 32\n最多 80 epoch；早停 patience 12\nfloat32 参考；不叠加过采样和增强",color=PURPLE,size=24)
    c.box(1186, 1280, 670, 291, "统一选权重与诊断", "以验证 macro-AP 选 checkpoint\n同时记录 train loss、val BCE / AP、lr、梯度\n已比较 GELU / ReLU；Pre-LN / Post-LN\n已分别关闭 z-score / dropout / weight decay\n一次只变一个因素，检查欠拟合与过拟合",color=TEAL,size=24)
    c.box(1186, 1611, 670, 188, "注意力的含义", "本结构只显式做时间注意力\n每帧关节通过输入投影共同编码\n注意力权重不等同于错误定位真值",color=BLUE,size=24)
    c.rect(64,1780,1060,79,fill="#FFF4E8",border="#EAC89F",radius=14)
    c.text(87,1799,"参考方案：75,845 参数；等权肩后伸 F1 0.734，仍仅 1 个种子。",25,ORANGE,True)
    c.footer("B = batch size。v2 / v3 已训练并核验；图示为参考结构，各外层入选配置可能不同。")
    return c.save("rehab26_transformer_model")


if __name__ == "__main__":
    print(json.dumps([experiment(), model()], ensure_ascii=False, indent=2))
