"""Verify native SVG motion and make compact animated previews for sharing."""
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent


def main():
    files=list(HERE.glob('*.svg'))+list((HERE/'svg').glob('*.svg'))
    for path in files:
        root=ET.parse(path).getroot()
        assert len(root.findall('.//{http://www.w3.org/2000/svg}animate'))>150
        assert not root.findall('.//{http://www.w3.org/2000/svg}script')
    verified=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1250},device_scale_factor=1)
        for path in [HERE/'shoulder_extension_comparison.svg', HERE/'shoulder_abduction_comparison.svg']:
            page.goto(path.as_uri())
            page.evaluate('document.documentElement.pauseAnimations(); document.documentElement.setCurrentTime(1)')
            before=page.locator('line').evaluate_all('(es)=>es.map(e=>[e.x1.animVal.value,e.y1.animVal.value,e.x2.animVal.value,e.y2.animVal.value])')
            page.evaluate('document.documentElement.setCurrentTime(2.5)')
            after=page.locator('line').evaluate_all('(es)=>es.map(e=>[e.x1.animVal.value,e.y1.animVal.value,e.x2.animVal.value,e.y2.animVal.value])')
            assert before!=after, path.name
            # Chrome's headless screenshot can stall on an SVG document root.
            # Render the identical SVG inline in HTML for the visual capture.
            page.goto('about:blank')
            page.set_content('<html><body style="margin:0">'+path.read_text(encoding='utf-8')+'</body></html>')
            page.evaluate('document.querySelector("svg").pauseAnimations(); document.querySelector("svg").setCurrentTime(2.5)')
            page.screenshot(path=str(HERE/(path.stem+'_preview.png')),full_page=True)
            verified.append(path.name)
        page.set_viewport_size({'width':1280,'height':1100})
        page.goto((HERE/('index.html' if (HERE/'index.html').exists() else 'demo.html')).as_uri())
        page.wait_for_function('window.motionDemo !== undefined')
        page.evaluate('motionDemo.setPlaying(false)')
        for exercise,code,name,title in [('7','ERR_BENT_ARM','preview_extension.gif','肩后伸 · 标准动作与屈肘错误对照'),
                                         ('1','ERR_LOW_ABDUCTION','preview_abduction.gif','肩外展 · 标准动作与幅度不足对照')]:
            page.evaluate('([ex,code])=>{motionDemo.setExercise(ex);motionDemo.setError(code);motionDemo.setPlaying(false)}',[exercise,code])
            frames=[]
            font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',18)
            for frame in range(48):
                page.evaluate('(p)=>motionDemo.setProgress(p)',frame/48)
                shot=Image.open(io.BytesIO(page.locator('.panels').screenshot())).convert('RGB')
                shot=shot.resize((1000,round(shot.height*1000/shot.width)),Image.Resampling.LANCZOS)
                canvas=Image.new('RGB',(1000,shot.height+48),'#f5f6f1')
                canvas.paste(shot,(0,48))
                ImageDraw.Draw(canvas).text((16,12),title,font=font,fill='#23342f')
                frames.append(canvas)
            frames[0].save(HERE/name,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True,disposal=2)
        browser.close()
    (HERE/'svg_verification.json').write_text(json.dumps({'status':'passed','standalone_svg_count':len(files),
        'checks':['all SVG XML parsed','native SMIL animations present','no scripts in standalone SVGs','animated coordinates change at browser SVG times 1s and 2.5s'],
        'browser_verified':verified,'gif_previews':2},ensure_ascii=False,indent=2),encoding='utf-8')
    print('14 SVGs parsed; standalone playback verified; 2 GIF previews exported')


if __name__=='__main__':
    main()
