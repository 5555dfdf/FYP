"""Build offline HTML and standalone SMIL SVG animations from exported poses."""
import argparse
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGES = [(1,2),(2,3),(3,7),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
         (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
         (24,26),(26,28),(28,30),(30,32),(28,32)]


def esc(s):
    return html.escape(str(s), quote=True)


def animation(attribute, values):
    return f'<animate attributeName="{attribute}" values="'+";".join(f"{v:.2f}" for v in values)+'" dur="5s" repeatCount="indefinite" calcMode="linear"/>'


def element(tag, attributes, sequences):
    attrs=' '.join(f'{k}="{esc(v)}"' for k,v in attributes.items())
    return f'<{tag} {attrs}>'+''.join(animation(k,v) for k,v in sequences.items())+f'</{tag}>'


def pose_group(clip, yaw, scale, ox=210, oy=402):
    poses = clip['points'][::2]
    poses = poses + [clip['points'][-1], poses[0]]
    projected = [[(ox+(p[0]*math.cos(yaw)+p[2]*math.sin(yaw))*scale,oy+p[1]*scale) for p in pose] for pose in poses]
    focus=set(clip['highlight'])
    color='#248c78' if not focus else '#7c958c'
    parts=[]
    for a,b in EDGES:
        active=bool(focus.intersection([a,b])) and not (a>=25 or b>=25)
        stroke='#cd6a40' if active else color
        attrs={'x1':f'{projected[0][a][0]:.2f}','y1':f'{projected[0][a][1]:.2f}','x2':f'{projected[0][b][0]:.2f}','y2':f'{projected[0][b][1]:.2f}',
               'stroke':stroke,'stroke-width':1.5 if a<11 and b<11 else 3.3,'stroke-linecap':'round'}
        parts.append(element('line',attrs,{'x1':[p[a][0] for p in projected],'y1':[p[a][1] for p in projected],'x2':[p[b][0] for p in projected],'y2':[p[b][1] for p in projected]}))
    centers=[]
    for p in projected:
        face=[p[j] for j in [0,7,8]]
        centers.append(((min(v[0] for v in face)+max(v[0] for v in face))/2,(min(v[1] for v in face)+max(v[1] for v in face))/2))
    head_color='#cd6a40' if any(j<11 for j in focus) else color
    parts.append(element('ellipse',{'cx':centers[0][0],'cy':centers[0][1],'rx':10,'ry':12,'fill':'none','stroke':head_color,'stroke-width':1.8},
                         {'cx':[p[0] for p in centers],'cy':[p[1] for p in centers]}))
    parts.append(element('line',{'x1':(projected[0][11][0]+projected[0][12][0])/2,'y1':(projected[0][11][1]+projected[0][12][1])/2,
                                'x2':centers[0][0],'y2':centers[0][1]+12,'stroke':head_color,'stroke-width':1.8,'opacity':.7},
                         {'x1':[(p[11][0]+p[12][0])/2 for p in projected],'y1':[(p[11][1]+p[12][1])/2 for p in projected],
                          'x2':[p[0] for p in centers],'y2':[p[1]+12 for p in centers]}))
    for j in range(33):
        attrs={'cx':f'{projected[0][j][0]:.2f}','cy':f'{projected[0][j][1]:.2f}','r':1.8 if j<11 else 3.4,'fill':'#cd6a40' if j in focus else 'white',
               'stroke':'#b95732' if j in focus else color,'stroke-width':1.2}
        parts.append(element('circle',attrs,{'cx':[p[j][0] for p in projected],'cy':[p[j][1] for p in projected]}))
    return ''.join(parts)


def scale_for(ex):
    points=[p for c in ex['clips'].values() for pose in c['points'] for p in pose]
    radius=max(math.hypot(p[0],p[2]) for p in points)
    height=max(-p[1] for p in points)
    return min(330/height,175/radius,100)


def svg_document(ex, codes, title, columns=3):
    width=columns*440+48
    rows=math.ceil(len(codes)/columns)
    height=rows*485+168
    schematic=ex.get('schematic',False)
    subtitle='动作原理示意 · 33 点骨架 · 5 秒循环' if schematic else '真实标注片段 · 5 秒循环 · 各片段按动作进度同步'
    footer='绿色：标准动作示意　橙色：错误涉及部位；幅度为绘图示意，不代表模型检测。' if schematic else '绿色：数据集正确标注　橙色：所选错误涉及部位；标签展示不代表模型检测。'
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
         f'<title id="title">{esc(title)}</title><desc id="desc">{esc(subtitle)}，绿色为标准示例，橙色标示错误相关部位。</desc>',
         '<style>text{font-family:"Segoe UI","Microsoft YaHei",sans-serif}</style>',
         f'<rect width="{width}" height="{height}" fill="#f5f6f1"/>',
         '<text x="40" y="38" font-size="11" letter-spacing="2" fill="#237f6d">REHAB / MOTION STUDY</text>',
         f'<text x="40" y="79" font-size="27" font-weight="600" fill="#23342f">{esc(title)}</text>',
         f'<text x="40" y="106" font-size="12" fill="#77877b">{esc(subtitle)}</text>']
    scale=scale_for(ex)
    for i,code in enumerate(codes):
        clip=ex['clips'][code]
        x=24+(i%columns)*440
        y=128+(i//columns)*485
        good=code=='correct'
        color='#237f6d' if good else '#bd5c34'
        out += [f'<g transform="translate({x},{y})">','<rect x="0" y="0" width="420" height="465" rx="16" fill="white" stroke="#dfe5dc"/>',
                f'<text x="23" y="32" font-size="18" font-weight="600" fill="{color}">{esc(clip["name"])}</text>',
                f'<text x="23" y="57" font-size="11" fill="#78877d">{esc(clip["hint"])}</text>',
                '<ellipse cx="210" cy="405" rx="76" ry="7" fill="#f0f3ed"/><line x1="65" y1="410" x2="355" y2="410" stroke="#dfe5db"/>',
                pose_group(clip,math.radians(ex['yaw']),scale),
                f'<text x="23" y="445" font-size="10" fill="#87958b">{esc(clip["id"])}</text></g>']
    out += [f'<text x="40" y="{height-18}" font-size="11" fill="#829084">{esc(footer)}</text></svg>']
    return ''.join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-only',action='store_true',help='Build schematic-only demo without reading recorded pose data')
    args=parser.parse_args()
    schematic=(HERE/'schematic_data.json').read_text(encoding='utf-8')
    data=json.loads(schematic)
    template=(HERE/'player.template.html').read_text(encoding='utf-8')
    public=template.replace('/*__MOTION_DATA__*/','null').replace('/*__SCHEMATIC_DATA__*/',schematic)
    public=public.replace('<button type="button" data-source="recorded" aria-pressed="false">真实数据回放</button>','')
    (HERE/'demo.html').write_text(public,encoding='utf-8')
    if not args.public_only:
        text=(HERE/'motion_data.json').read_text(encoding='utf-8')
        (HERE/'index.html').write_text(template.replace('/*__MOTION_DATA__*/',text).replace('/*__SCHEMATIC_DATA__*/',schematic),encoding='utf-8')
    directory=HERE/'svg'
    directory.mkdir(exist_ok=True)
    for ex_id,ex in data['exercises'].items():
        name='shoulder_extension' if ex_id=='7' else 'shoulder_abduction'
        codes=['correct']+ex['errors']
        (HERE/f'{name}_comparison.svg').write_text(svg_document(ex,codes,f'{ex["name"]} · 标准与错误动作对照'),encoding='utf-8')
        for code in codes:
            (directory/f'ex{ex_id}_{code.lower()}.svg').write_text(svg_document(ex,[code],f'{ex["name"]} · {ex["clips"][code]["name"]}',columns=1),encoding='utf-8')
    if not args.public_only:
        recorded=HERE/'recorded_svg'
        recorded.mkdir(exist_ok=True)
        for ex_id,ex in json.loads(text)['exercises'].items():
            (recorded/f'ex{ex_id}_recorded_comparison.svg').write_text(svg_document(ex,['correct']+ex['errors'],f'{ex["name"]} · REHAB26 标注回放'),encoding='utf-8')
    print('Built schematic demo.html, 2 comparison SVGs, 12 individual SVGs')


if __name__=='__main__':
    main()
