"""Exercise the offline player in Chrome and capture visual previews."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public',action='store_true')
    args=parser.parse_args()
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe", headless=True, args=["--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width": 1440, "height": 1150}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        entry=HERE / ('demo.html' if args.public or not (HERE/'index.html').exists() else 'index.html')
        page.goto(entry.as_uri(), wait_until="load")
        page.wait_for_function("window.motionDemo !== undefined")
        before = page.evaluate("motionDemo.state.progress")
        page.wait_for_timeout(300)
        assert page.evaluate("motionDemo.state.progress") != before
        page.evaluate("motionDemo.setPlaying(false); motionDemo.setProgress(.52)")
        page.wait_for_timeout(100)
        assert page.evaluate("motionDemo.state.progress") == .52
        checked = []
        for exercise in ["7", "1"]:
            page.locator(f'[data-ex="{exercise}"]').click()
            codes = page.evaluate("motionDemo.DATA.exercises[motionDemo.state.exercise].errors")
            for code in codes:
                page.locator(f'[data-error="{code}"]').click()
                page.evaluate("motionDemo.setPlaying(false); motionDemo.setProgress(.52)")
                assert page.locator("#bad-title").inner_text()
                assert page.locator("#good-svg circle").count() >= 33
                assert page.locator("#bad-svg circle").count() >= 33
                bad = page.locator("#bad-svg").inner_html()
                assert "NaN" not in bad and "Infinity" not in bad
                checked.append({"source": "schematic", "exercise": exercise, "error": code})
            for view in ["front", "oblique", "side"]:
                page.locator(f'[data-view="{view}"]').click()
                assert page.evaluate("motionDemo.state.view") == view
        if page.locator('[data-source="recorded"]').count():
            page.locator('[data-source="recorded"]').click()
            assert page.evaluate("motionDemo.state.source") == "recorded"
            assert "REHAB26" in page.locator("#source-kind").inner_text()
            for exercise in ["7", "1"]:
                page.locator(f'[data-ex="{exercise}"]').click()
                for code in page.evaluate("motionDemo.DATA.exercises[motionDemo.state.exercise].errors"):
                    page.locator(f'[data-error="{code}"]').click()
                    page.evaluate("motionDemo.setPlaying(false); motionDemo.setProgress(.52)")
                    assert "NaN" not in page.locator("#bad-svg").inner_html()
                    assert "ex" in page.locator("#source").inner_text()
                    checked.append({"source": "recorded", "exercise": exercise, "error": code})
            page.locator('[data-source="schematic"]').click()
            assert page.evaluate("motionDemo.state.source") == "schematic"
        page.locator('[data-ex="7"]').click()
        page.evaluate("motionDemo.setPlaying(false); motionDemo.setProgress(.56)")
        page.screenshot(path=str(HERE / "preview_extension.png"), full_page=True)
        page.locator('[data-ex="1"]').click()
        page.evaluate("motionDemo.setPlaying(false); motionDemo.setProgress(.52)")
        page.screenshot(path=str(HERE / "preview_abduction.png"), full_page=True)
        page.locator("#ghost").uncheck()
        assert page.evaluate("motionDemo.state.ghost") is False
        page.locator("#speed").select_option("0.5")
        assert page.evaluate("motionDemo.state.speed") == .5
        page.locator("#play").click()
        assert page.evaluate("motionDemo.state.playing") is True
        page.evaluate("motionDemo.setPlaying(false)")
        page.locator("#progress").fill("780")
        assert abs(page.evaluate("motionDemo.state.progress") - .78) < 1e-9
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(HERE / "preview_mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        browser.close()
    assert not errors, errors
    (HERE / ("qa_public.json" if args.public else "qa_report.json")).write_text(json.dumps({"status": "passed", "entry":entry.name,"labelled_examples_checked": checked,
        "checks": ["offline file URL", "automatic playback", "pause", "seek", "all available sources, both exercises, all ten errors", "three views",
                   "reference toggle", "speed control", "390px mobile without horizontal overflow", "no browser script errors"], "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Player checks passed")


if __name__ == "__main__":
    main()
