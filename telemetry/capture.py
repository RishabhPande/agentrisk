"""Capture client-side telemetry emitted by browser-automation primitives that
LLM agent frameworks use to act on web pages. Each mode completes the same
checkout form; the page's collector records what a behavioral fraud/bot
model would see. Output: telemetry.csv (one row per run)."""
import asyncio, json, pathlib, random, sys
import pandas as pd
from playwright.async_api import async_playwright

HERE = pathlib.Path(__file__).parent
URL = (HERE / "checkout.html").resolve().as_uri()
FIELDS = {"name": "Jane Q. Doe", "email": "jane.doe@example.com",
          "addr": "12 Elm Street, Springfield", "card": "4111111111111111", "cvv": "123"}
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


async def dom_fill(page):            # browser-use / Stagehand style: locator.fill + click
    for k, v in FIELDS.items():
        await page.fill(f"#{k}", v)
    await page.click("#pay")


async def js_set(page):              # pure-DOM scripting: set value + dispatch events
    await page.evaluate("""(F)=>{for(const [k,v] of Object.entries(F)){const e=document.getElementById(k);
        e.focus(); e.value=v; e.dispatchEvent(new Event('input',{bubbles:true}));}
        document.getElementById('pay').click();}""", FIELDS)


async def cdp_insert(page):          # CDP Input.insertText (used by several agent toolkits)
    for k, v in FIELDS.items():
        await page.click(f"#{k}")
        await page.keyboard.insert_text(v)
    await page.click("#pay")


async def kb_type(page):             # synthesized key events, no delay
    for k, v in FIELDS.items():
        await page.locator(f"#{k}").press_sequentially(v)
    await page.click("#pay")


async def kb_type_delay(page):       # synthesized key events, fixed 100 ms delay
    for k, v in FIELDS.items():
        await page.locator(f"#{k}").press_sequentially(v, delay=100)
    await page.click("#pay")


async def vision_cu(page):           # computer-use style: coordinate mouse paths + typing
    await page.mouse.move(random.randint(50, 600), random.randint(50, 400))
    await page.mouse.wheel(0, 200); await page.wait_for_timeout(300)
    await page.mouse.wheel(0, -200)
    for k, v in FIELDS.items():
        box = await page.locator(f"#{k}").bounding_box()
        await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=25)
        await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        await page.keyboard.type(v, delay=40)
        await page.wait_for_timeout(random.randint(400, 1200))   # model "think" latency
    box = await page.locator("#pay").bounding_box()
    await page.mouse.move(box["x"] + 10, box["y"] + 5, steps=25)
    await page.mouse.click(box["x"] + 10, box["y"] + 5)


MODES = {"dom_fill": dom_fill, "js_set": js_set, "cdp_insert_text": cdp_insert,
         "kb_type": kb_type, "kb_type_delay100": kb_type_delay, "vision_cu": vision_cu}


async def main():
    rows = []
    async with async_playwright() as p:
        for stealth in (False, True):
            args = ["--disable-blink-features=AutomationControlled"] if stealth else []
            browser = await p.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium", args=args + ["--no-sandbox"])
            for mode, fn in MODES.items():
                for r in range(REPS):
                    ctx = await browser.new_context()
                    page = await ctx.new_page()
                    await page.goto(URL)
                    await fn(page)
                    t = await page.evaluate("window.__telemetry")
                    t.update(mode=mode, stealth=stealth, rep=r)
                    rows.append(t)
                    await ctx.close()
            await browser.close()
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "telemetry.csv", index=False)
    cols = ["mousemove", "keydown", "input", "untrusted", "keydown_per_input", "iki_mean_ms",
            "iki_sd_ms", "mouse_straightness", "time_to_submit_ms", "webdriver", "ua_headless", "plugins"]
    summ = df.groupby(["stealth", "mode"])[cols].mean(numeric_only=False).round(2)
    print(summ.to_string())
    summ.to_csv(HERE / "telemetry_summary.csv")

asyncio.run(main())
