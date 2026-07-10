import sys
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

APP_URL       = "https://lumen-by-hn.streamlit.app"
WAKE_BUTTON   = "Yes, get this app back up!"
READY_TEXT    = "Get precise answers with page-level sources"
POLL_INTERVAL = 5    # seconds between readiness checks
MAX_WAIT      = 240  # seconds before giving up (cold boots can be slow)


def app_ready(page) -> bool:
    """Return True if READY_TEXT is found in any frame on the page."""
    for frame in page.frames:
        try:
            body = frame.inner_text("body", timeout=1000)
            if READY_TEXT in body:
                print(f"  Ready — matched frame: {frame.url}")
                return True
        except Exception:
            continue
    return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        print(f"Navigating to {APP_URL} ...")
        page.goto(APP_URL, wait_until="domcontentloaded")

        # The wake button is JS-rendered a moment after page load, so
        # .click(timeout=15000) auto-waits for it.  A TimeoutError here
        # simply means the app was already awake — not a failure.
        try:
            page.get_by_role("button", name=WAKE_BUTTON).first.click(timeout=15000)
            print("Wake button clicked — waiting for app to boot...")
        except PlaywrightTimeoutError:
            print("Wake button not found — app appears already awake.")

        # Poll across all frames until the real app UI is visible.
        print(f"Waiting up to {MAX_WAIT}s for app to become ready...")
        deadline = time.time() + MAX_WAIT
        while time.time() < deadline:
            if app_ready(page):
                print("App is ready. Exiting successfully.")
                browser.close()
                sys.exit(0)
            time.sleep(POLL_INTERVAL)

        page.screenshot(path="timeout_screenshot.png")
        print(f"ERROR: App did not become ready within {MAX_WAIT}s.")
        browser.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
