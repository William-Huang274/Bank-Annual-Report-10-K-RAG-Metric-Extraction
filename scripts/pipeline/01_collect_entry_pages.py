"""
Collect candidate entry pages for bank annual reports / 10-K filings.

This script crawls bank websites starting from a given entry URL,
applies BFS with domain constraints, scores pages based on configurable
rules, and identifies candidate pages that may contain annual report PDFs.
"""
import json
import time
import re  # Used for extracting 4-digit years from text
from urllib.parse import urljoin, urlparse
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):  # Search up to 10 parent directories for the project root
        if (p / ".git").exists() or (p / "README.md").exists() or (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Cannot locate repo root from: {start}")

ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def get_root_host(url: str) -> str | None:
    """
    Extract the root domain (e.g., example.com) from a URL.
    Returns None if the URL is invalid or cannot be parsed.
    """
    try:
        if not url:
            return None
        if not url.startswith("http"):
            url = "https://" + url.lstrip("/")
        n = urlparse(url)
        if not n.netloc:
            return None
        parts = n.netloc.lower().split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return n.netloc.lower()
    except Exception:
        return None


INPUT_CSV = ROOT/"data"/"input"/"successful bank report list_20251209.csv"
CONFIG_FILE = ROOT/"config"/"entry_page_scoring_config.json"
OUTPUT_CSV = ROOT/"data"/"interim"/"output"/"log"/"bank_candidate_entry_pages_for_pdf_2rd.csv"  # Output candidate entry pages for downstream PDF collection

LIMIT_BANKS = 40  # Optional limit for batch processing; set to None for full run
# LIMIT_BANKS = None

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

ALLOWED_EXTERNAL_ROOTS = {
    "sec.gov",
    "q4cdn.com",
    "s1.q4cdn.com",
    "q4ir.com",
    "materials.proxyvote.com",
    # Common external IR / CDN domains allowed during crawling
    "gcs-web.com",
    "corporate-ir.net",
    "services.corporate-ir.net",
    "s3.amazonaws.com",
}
# Some banks host investor relations or annual reports on external domains.
# These overrides define stable entry points for crawling.
BANK_START_URL_OVERRIDE = {
    # FMStBank: manually verified IR landing page
    "FMStBank": "https://ir.fm.bank/fms",

    # Extract candidate fiscal years from link text and URL.
    "gntybank": "https://www.gnty.com/investors/",

    # AlwaysOurBest: Skyline National Bank investor relations site hosted on a Q4 platform.
    "AlwaysOurBest": "https://investors.skylinenationalbank.com/financials/annual-reports/default.aspx",

    # EphrataNational: optional override to jump directly to the annual reports listing page.
    "EphrataNational": "https://enbfinancial.q4ir.com/financial-information/annual-reports-and-documents/default.aspx",

    # CapitalOne: optional override to start from the investor annual reports hub.
    "CapitalOne": "https://investor.capitalone.com/financial-results/annual-reports"
}


def fetch_html_playwright(url: str, max_total_timeout: int = 30000) -> str | None:
    """
    Robust Playwright-based HTML fetcher with bounded total runtime.

    Design considerations:
    - Enforces a maximum total timeout across retries
    - Uses short navigation and content timeouts per attempt
    - Retries multiple times and ensures browser instances are closed
    """
    start_time = time.time()
    attempts = 3

    for attempt in range(1, attempts + 1):
        try:
            elapsed = (time.time() - start_time) * 1000
            if elapsed > max_total_timeout:
                print(f"[PW] total timeout exceeded for {url}")
                return None

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    java_script_enabled=True,
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    ),
                    viewport={"width": 1600, "height": 1000},
                )
                page = context.new_page()

                page.set_default_navigation_timeout(10000)  # 10s
                page.set_default_timeout(5000)  # 5s content timeout

                print(f"[PW] attempt {attempt} → {url}")

                try:
                    page.goto(url, wait_until="domcontentloaded")
                except PlaywrightTimeout:
                    print(f"[PW] navigation timeout on attempt {attempt}")
                    browser.close()
                    continue

                # give JS time to render, but small
                page.wait_for_timeout(800)

                try:
                    html = page.content()
                except Exception as e:
                    print(f"[PW] content error on attempt {attempt}: {e}")
                    browser.close()
                    continue

                browser.close()
                return html

        except Exception as e:
            print(f"[PW] crash attempt {attempt}: {e}")
            continue

    print(f"[PW] all attempts failed for {url}")
    return None


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def looks_like_ir_domain(url: str) -> bool:
    """
    Heuristically identify investor-relations or Q4-style domains
    that typically require heavier JavaScript rendering.
    """
    patterns = [
        r"(^|\.)ir\.", r"(^|\.)investor", r"q4cdn", r"q4web", r"q4inc",
        r"broadridge", r"app\.q4web"
    ]
    return any(re.search(p, url.lower()) for p in patterns)


def ok_html(text: str) -> bool:
    """Heuristically determine whether the HTML response is valid."""
    if not text:
        return False
    text_low = text.lower()
    if "<html" not in text_low:
        return False
    if len(text_low) < 800:  # Too short; likely blocked, truncated, or non-content page
        return False
    return True


# -------------------------
# Layer 0: requests
# -------------------------

def try_requests(url: str, timeout=8):
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            },
            timeout=timeout
        )
        if resp.status_code >= 400:
            return None
        txt = resp.text
        return txt if ok_html(txt) else None
    except Exception:
        return None


# -------------------------
# Layer 1: Playwright v1 (light mode)
# -------------------------

def try_playwright_v1(url: str) -> str | None:
    """
    Lightweight fallback:
    - headless=True
    - wait_until='domcontentloaded'
    - short timeout
    """
    print(f"[PWv1] → {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                java_script_enabled=True,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            )
            page = context.new_page()
            page.set_default_navigation_timeout(12000)  # 12s

            try:
                page.goto(url, wait_until="domcontentloaded")
            except PlaywrightTimeout:
                print("[PWv1] navigation timeout")
                browser.close()
                return None

            page.wait_for_timeout(500)
            html = page.content()
            browser.close()
            return html if ok_html(html) else None

    except Exception:
        return None


# -------------------------
# Layer 2: Playwright v2 (anti-WAF mode for IR-heavy domains)
# -------------------------

def try_playwright_v2(url: str) -> str | None:
    """
    Anti-WAF mode for IR/Q4 domains:
    - headless=False (higher success rate)
    - hide webdriver
    - wait_until='networkidle'
    - longer timeout
    """
    print(f"[PWv2] → {url}")
    for attempt in range(1, 3):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False)
                context = browser.new_context(
                    java_script_enabled=True,
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    ),
                    viewport={"width": 1600, "height": 1000},
                )
                page = context.new_page()

                # hide webdriver
                page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                """)

                page.set_default_navigation_timeout(35000)  # 35s

                try:
                    page.goto(url, wait_until="networkidle")
                except PlaywrightTimeout:
                    print(f"[PWv2] navigation timeout attempt {attempt}")
                    browser.close()
                    continue

                page.wait_for_timeout(1500)
                html = page.content()
                browser.close()
                if ok_html(html):
                    return html
        except Exception:
            continue

    return None


# -------------------------
# Main function
# -------------------------

def safe_get_html(url: str, depth: int) -> str | None:
    """
    Three-stage HTML retrieval strategy:
    
    Layer 0: requests (fast path for static pages)
    Layer 1: Playwright v1 (lightweight JS rendering)
    Layer 2: Playwright v2 (anti-WAF mode for entry pages and IR domains)
    """
    print(f"[safe_get_html] depth={depth} url={url}")

    # --- Layer 0: requests --------------------------------------------------
    html = try_requests(url)
    if html:
        print("[safe_get_html] → success via requests")
        return html

    # --- Layer 1: PWv1 ------------------------------------------------------
    html = try_playwright_v1(url)
    if html:
        print("[safe_get_html] → success via PWv1")
        return html

    # --- Layer 2: PWv2 (only for depth=0 or IR/Q4 domains) ------------------
    if depth == 0 or looks_like_ir_domain(url):
        print("[safe_get_html] → fallback to PWv2 (Anti-WAF)")
        html = try_playwright_v2(url)
        if html:
            print("[safe_get_html] → success via PWv2")
            return html

    print("[safe_get_html] FAILED for", url)
    return None



def is_allowed_domain(url: str, base_root: str | None, entry_root: str | None) -> bool:
    """
    Determine whether a URL is allowed to be crawled.
    Allowed cases:
    1. Same root domain as base_url
    2. Same root domain as entry_url
    3. Whitelisted external IR / CDN domains (e.g., q4ir, q4cdn, sec.gov)
    """
    try:
        n = urlparse(url)
        if not n.netloc:
            return True  # Relative URL (no netloc); treat as same-site.    

        host = n.netloc.lower()
        parts = host.split(".")
        root = ".".join(parts[-2:]) if len(parts) >= 2 else host

        # allow external IR/CDN
        if root in ALLOWED_EXTERNAL_ROOTS:
            return True

        # allow same domain as original base url
        if base_root and root == base_root:
            return True

        # allow domain of financial report link (entry_url)
        if entry_root and root == entry_root:
            return True

        return False
    except Exception:
        return True


def get_page_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title")
        if not title:
            return ""
        return title.get_text(" ", strip=True)
    except Exception:
        return ""


def score_page(url: str, anchor_text: str, page_title: str, cfg: dict) -> tuple[int, list[str], list[str]]:
    """
    Score a page using URL + anchor_text + page_title.
    Returns:
        (score, tags, reasons)
        - tags: a sorted list of matched rule tags (e.g., "annual_report", "sec_filings").
        - reasons: human-readable matches (e.g., "annual_report::annual report", "NEG::privacy").
    """
    positive_rules = cfg["positive_rules"]
    negative_keywords = cfg["negative_keywords"]
    negative_penalty = cfg.get("negative_penalty", 3)

    text = f"{anchor_text or ''} {page_title or ''}".lower()
    url_l = url.lower()

    score = 0
    tags: set[str] = set()
    reasons: list[str] = []

    # Apply positive scoring rules (keyword hits add weight).
    for rule in positive_rules:
        tag = rule["tag"]
        weight = rule["weight"]
        for kw in rule["keywords"]:
            kw_l = kw.lower()
            if kw_l in url_l or kw_l in text:
                score += weight
                tags.add(tag)
                reasons.append(f"{tag}::{kw}")

    # Apply negative keywords (keyword hits subtract a fixed penalty).  
    for kw in negative_keywords:
        kw_l = kw.lower()
        if kw_l in url_l or kw_l in text:
            score -= negative_penalty
            reasons.append(f"NEG::{kw}")

    return score, sorted(tags), reasons


def classify_level(score: int, cfg: dict) -> str | None:
    """
    Classify a page into HIGH / MEDIUM / LOW based on score thresholds.
    Returns None if the score is below the minimum threshold.
    """
    th = cfg["score_thresholds"]
    if score >= th["HIGH"]:
        return "HIGH"
    if score >= th["MEDIUM"]:
        return "MEDIUM"
    if score >= th["LOW"]:
        return "LOW"
    return None  # Below the minimum threshold; exclude unless PDFs are present (handled separately).

# ========= PDF-related helpers =========

def extract_years_from_text(text: str) -> set[str]:
    """
    Extract 4-digit years (1900–2099) from text.
    Returns a set of year strings.
    """
    if not text:
        return set()
    years = set(re.findall(r"\b(19[0-9]{2}|20[0-9]{2})\b", text))
    return years


def analyze_pdfs_on_page(html: str, page_url: str) -> tuple[int, dict | None]:
    """
    Scan all <a> tags on the page, identify PDF links, and score them.
    Returns:
        pdf_count: Number of PDF links found on the page
        best_pdf: Best PDF candidate information dict, or None if not found.
            {
                "pdf_url": str,
                "score": int,
                "years": "2022,2023",
                "is_10k": bool,
                "anchor_text": str,
            }
    """
    # Common non-annual-report PDF patterns (fees, forms, charters, policies, etc.)
    NEG_PDF_KEYWORDS = [
        "patriot",              # Known non-report PDF pattern observed on CapitalOne (e.g., patriot_2025.pdf).
        "switch-kit", "switch kit",
        "schedule-of-fees", "schedule of fees",
        "fee schedule", "feeschedule",
        "forms", "form ", "application",
        "charter", "bylaws",
        "policy", "policies",
        "privacy", "terms-and-conditions", "terms and conditions",
        "loan-application", "account-agreement"
    ]

    soup = BeautifulSoup(html, "html.parser")
    pdf_count = 0
    best_pdf = None  # (score, info_dict)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_href = urljoin(page_url, href)
        href_low = abs_href.lower()
        text = a.get_text(" ", strip=True) or ""
        text_low = text.lower()

        
        # Only consider direct PDF links.
        if not href_low.endswith(".pdf"):
            continue

        # Filter out common non-report PDFs: if filename or anchor text contains typical irrelevant keywords, skip.
        if any(kw in href_low or kw in text_low for kw in NEG_PDF_KEYWORDS):
            continue

        pdf_count += 1

        # Extract candidate fiscal years from anchor text and URL.
        years = extract_years_from_text(text_low) | extract_years_from_text(href_low)

        is_10k = ("10-k" in text_low) or ("10k" in text_low) or ("form 10-k" in text_low) or ("10-k" in href_low)

        score = 0
        # Base score for being a PDF link.
        score += 10

        # Boost when anchor text or URL contains annual-report keywords.
        if "annual report" in text_low or "annual report" in href_low:
            score += 20      # Strong annual-report signal.
        elif "annual" in text_low or "annual" in href_low:
            score += 10

        # Boost for 10-K / Form 10-K signals (secondary to annual-report terms).
        if is_10k:
            score += 15

        # Additional points for explicit years; multiple years accumulate.
        if years:
            for y in years:
                try:
                    y_int = int(y)
                    if y_int >= 2019:
                        score += 4
                    else:
                        score += 2
                except ValueError:
                    score += 2

        info = {
            "pdf_url": abs_href,
            "score": score,
            "years": ", ".join(sorted(years)),
            "is_10k": is_10k,
            "anchor_text": text,
        }

        if best_pdf is None or score > best_pdf[0]:
            best_pdf = (score, info)

    if best_pdf is None:
        return pdf_count, None
    else:
        return pdf_count, best_pdf[1]


# ========= Main crawling logic (BFS) =========

def collect_candidates_for_bank(
    session: requests.Session,
    cfg: dict,
    idrssd,
    username,
    base_url: str,
    start_url: str,
    base_root: str | None,
    entry_root: str | None,
) -> list[dict]:
    """
    Starting from a bank-specific entry URL, perform BFS crawling to collect
    candidate entry pages and associated PDF information.
    Returns a list of records, each containing:
        - score, level, tags, reasons
        - pdf_count_on_page
        - best_pdf_* fields if applicable
    """
    crawl_cfg = cfg["crawl"]
    max_depth = crawl_cfg["max_depth"]
    max_pages = crawl_cfg["max_pages_per_bank"]

    visited: set[str] = set()
    # Queue item: (url, depth, parent_url, anchor_text)
    queue: list[tuple[str, int, str | None, str | None]] = [(start_url, 0, None, None)]

    candidates: list[dict] = []
    pages_count = 0

    while queue and pages_count < max_pages:
        url, depth, parent_url, anchor_text = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        if not is_allowed_domain(url, base_root, entry_root):
            continue

        html = safe_get_html(url, depth)
        if not html:
            continue

        pages_count += 1
        print(f"    [PAGE] depth={depth} url={url}")

        page_title = get_page_title(html)

        # Analyze PDF links on the page first (used later by the download step).
        pdf_count, best_pdf = analyze_pdfs_on_page(html, url)  # NEW

        # 1) Score the current page to decide whether to keep it as a candidate entry page.
        score, tags, reasons = score_page(
            url=url,
            anchor_text=anchor_text or "",
            page_title=page_title,
            cfg=cfg,
        )
        level = classify_level(score, cfg)

        # Selection criteria:
        # A page is kept as a candidate if it satisfies at least one of the following:
        # - Contains PDF links
        # - Has a positive relevance score
        # - Matches important financial-related tags
        important_tags = {"annual_report", "10-K", "sec_filings", "financial_reports", "investor_relations"}

        has_important_tag = any(t in important_tags for t in tags)
        has_positive_score = score > 0

        # Selection criteria:
        # A page is kept as a candidate if it satisfies at least one of the following:
        # - Contains PDF links
        # - Has a positive relevance score
        # - Matches important financial-related tags
        if pdf_count > 0 or has_positive_score or has_important_tag:
            row = {
                "idrssd": idrssd,
                "username": username,
                "base_url": base_url,
                "start_url": start_url,
                "page_url": url,
                "parent_url": parent_url,
                "depth": depth,
                "score": score,
                "level": level or "",
                "tags": ";".join(tags),
                "reasons": ";".join(reasons),
                "anchor_text": anchor_text or "",
                "page_title": page_title,
                "pdf_count_on_page": pdf_count,
            }

            if best_pdf is not None:
                row["best_pdf_url"] = best_pdf["pdf_url"]
                row["best_pdf_score"] = best_pdf["score"]
                row["best_pdf_years"] = best_pdf["years"]
                row["best_pdf_is_10k"] = "YES" if best_pdf["is_10k"] else "NO"
                row["best_pdf_anchor"] = best_pdf["anchor_text"]
            else:
                row["best_pdf_url"] = ""
                row["best_pdf_score"] = 0
                row["best_pdf_years"] = ""
                row["best_pdf_is_10k"] = ""
                row["best_pdf_anchor"] = ""

            candidates.append(row)


        # 2) Decide whether to expand to the next layer (BFS crawl).
        if depth == max_depth:
            continue

        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            new_url = urljoin(url, href)
            new_url_l = new_url.lower()

            # Skip non-HTML static assets (images/CSS/JS/icons/PDF).
            if any(
                new_url_l.endswith(ext)
                for ext in [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".css", ".js", ".ico"]
            ):
                continue

            if not is_allowed_domain(new_url, base_root, entry_root):
                continue

            link_text = a.get_text(" ", strip=True) or ""
            link_text_low = link_text.lower()

            # At depth=0 we expand more broadly to reach IR/financial hubs quickly.
            if depth == 0:
                if any(
                    kw in new_url_l
                    for kw in [
                        "investor",
                        "investors",
                        "financial",
                        "financials",
                        "sec",
                        "filing",
                        "reports",
                        "annual",
                        "ir",  # NEW
                    ]
                ) or any(
                    kw in link_text_low
                    for kw in [
                        "investor",
                        "financial",
                        "annual",
                        "10-k",
                        "10k",
                        "report",
                        "ir",
                    ]
                ):
                    queue.append((new_url, depth + 1, url, link_text))
            else:
                # At depth>=1 we tighten expansion to links that look like annual reports / 10-K / filings pages.
                if any(
                    kw in new_url_l
                    for kw in [
                        "annual",
                        "report",
                        "10-k",
                        "10k",
                        "sec-filings",
                        "secfilings",
                        "financial-report",
                        "financial-reports",
                        "financials",
                        "sec",
                    ]
                ) or any(
                    kw in link_text_low
                    for kw in ["annual", "10-k", "10k", "financial", "report"]
                ):
                    queue.append((new_url, depth + 1, url, link_text))

    return candidates


def main():
    cfg = load_config(CONFIG_FILE)
    df = pd.read_csv(INPUT_CSV)

    # Only keep banks explicitly marked as having financial reports
    banks = df[df["financial report"] == "YES"].copy()

    if LIMIT_BANKS is not None:
        banks = banks.head(LIMIT_BANKS)

    print(f"Total banks to process: {len(banks)}")

    session = requests.Session()
    all_rows: list[dict] = []

    for idx, row in banks.iterrows():
        idrssd = row.get("idrssd")
        username = row.get("username")
        base_url = row.get("url")

        # Use financial report link as the primary entry URL; fallback to base_url if missing
        entry_url = row.get("financial report link") or base_url
        # Allow username-based overrides for special cases (e.g., cross-domain IR platforms)
        override = BANK_START_URL_OVERRIDE.get(str(username))
        if override:
            print(f"  [OVERRIDE] use custom start_url for {username}: {override}")
            entry_url = override

        if not isinstance(entry_url, str):
            print(f"[SKIP] No valid entry_url for {username} ({base_url})")
            continue

        start_url = entry_url
        if not start_url.startswith("http"):
            start_url = "https://" + start_url.lstrip("/")

        base_root = get_root_host(base_url) if isinstance(base_url, str) else None
        entry_root = get_root_host(entry_url)

        print("=" * 80)
        print(f"[BANK] idrssd={idrssd} username={username}")
        print(f"  base_url  = {base_url}")
        print(f"  start_url = {start_url}")
        print(f"  roots     = base_root={base_root}, entry_root={entry_root}")

        try:
            candidates = collect_candidates_for_bank(
                session=session,
                cfg=cfg,
                idrssd=idrssd,
                username=username,
                base_url=base_url,
                start_url=start_url,
                base_root=base_root,
                entry_root=entry_root,
            )
            print(f"  -> collected {len(candidates)} candidate pages.\n")
            all_rows.extend(candidates)
        except Exception as e:
            print(f"[ERROR] Bank {username} exception: {e}")

        time.sleep(1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n[DONE] wrote {len(all_rows)} rows to {OUTPUT_CSV}")
    else:
        print("\n[DONE] no candidate pages collected.")


if __name__ == "__main__":
    main()
