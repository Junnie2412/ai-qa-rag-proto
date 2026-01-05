from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from lxml.html import fromstring


@dataclass
class ScrapeResult:
    url: str
    html: str
    text: str


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def extract_text_from_html(html: str) -> str:
    doc = fromstring(html)
    # loại bỏ script/style cơ bản
    for bad in doc.xpath("//script|//style|//noscript"):
        bad.getparent().remove(bad)
    text = doc.text_content()
    # normalize whitespace
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text


async def scrape_url(
    url: str,
    allowed_domains: Optional[Sequence[str]] = None,
    headless: bool = True,
    timeout_ms: int = 30000,
) -> ScrapeResult:
    if allowed_domains:
        d = _domain(url)
        allow = [x.lower() for x in allowed_domains]
        if not any(d == a or d.endswith("." + a) for a in allow):
            raise ValueError(f"Domain not allowed: {d}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        page.set_default_timeout(timeout_ms)
        await page.goto(url, wait_until="domcontentloaded")
        html = await page.content()
        await browser.close()

    text = extract_text_from_html(html)
    return ScrapeResult(url=url, html=html, text=text)
