import json
import time
import re  # NEW: 用于提取年份
from urllib.parse import urljoin, urlparse
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):  # 最多向上找 10 层
        if (p / ".git").exists() or (p / "README.md").exists() or (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Cannot locate repo root from: {start}")

ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def get_root_host(url: str) -> str | None:
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



# ======== 配置路径 ========
INPUT_CSV = ROOT/"data"/"input"/"successful bank report list_20251209.csv"
CONFIG_FILE = ROOT/"config"/"entry_page_scoring_config.json"
OUTPUT_CSV = ROOT/"data"/"interim"/"output"/"log"/"bank_candidate_entry_pages_for_pdf_2rd.csv"  # NEW: 输出文件名改一下，表明是为PDF服务的

# 调试时可以只跑前 N 个银行
LIMIT_BANKS = 40  # 调试时 20，OK 再改成 None
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
    # NEW: 适当加一些常见 IR/CDN 域名（后续可以再补）
    "gcs-web.com",
    "corporate-ir.net",
    "services.corporate-ir.net",
    "s3.amazonaws.com",
}
# 某些银行的 IR/年报入口在另外的域名，需要手动 override
BANK_START_URL_OVERRIDE = {
    # FMStBank: 你的手动检查链接
    "FMStBank": "https://ir.fm.bank/fms",

    # gntybank: 官方 IR 在 gnty.com
    "gntybank": "https://www.gnty.com/investors/",

    # AlwaysOurBest: Skyline National Bank 的 Q4 IR 平台
    "AlwaysOurBest": "https://investors.skylinenationalbank.com/financials/annual-reports/default.aspx",

    # EphrataNational: 直接指向 annual reports 列表页（可选，但有利于稳定抓 pdf）
    "EphrataNational": "https://enbfinancial.q4ir.com/financial-information/annual-reports-and-documents/default.aspx",

    # CapitalOne 也可以直接用 investor 年报页（可选）
    "CapitalOne": "https://investor.capitalone.com/financial-results/annual-reports"
}


def fetch_html_playwright(url: str, max_total_timeout: int = 30000) -> str | None:
    """
    更稳健的 Playwright fallback：
    - 总耗时不会超过 max_total_timeout（默认 30 秒）
    - navigation timeout = 10 秒
    - content timeout = 5 秒
    - 三次重试
    - 失败会强制关闭浏览器避免 hanging
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


# def safe_get(session: requests.Session, url: str) -> str | None:
#     """先用 requests，4xx/空页面再用 Playwright 补救。"""
#     try:
#         resp = session.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
#         status = resp.status_code
#         text = resp.text or ""
#         if 200 <= status < 300 and len(text) > 500 and "<html" in text.lower():
#             return text
#         else:
#             print(f"[WARN] {status} or weak HTML for {url}, fallback PW")
#     except Exception as e:
#         print(f"[ERROR] GET {url}: {e} -> fallback PW")

#     # fallback：Playwright
#     html = fetch_html_playwright(url)
#     if html and len(html) > 500:
#         return html

#     print(f"[ERROR] both requests & PW failed for {url}")
#     return None

# -------------------------
# Helpers
# -------------------------

def looks_like_ir_domain(url: str) -> bool:
    """
    Q4 / Investor relations / heavy JS domains.
    """
    patterns = [
        r"(^|\.)ir\.", r"(^|\.)investor", r"q4cdn", r"q4web", r"q4inc",
        r"broadridge", r"app\.q4web"
    ]
    return any(re.search(p, url.lower()) for p in patterns)


def ok_html(text: str) -> bool:
    """判断 HTML 是否有效"""
    if not text:
        return False
    text_low = text.lower()
    if "<html" not in text_low:
        return False
    if len(text_low) < 800:  # 太短，大概率是被挡住了
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
# Layer 2: Playwright v2 （Anti-WAF）
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
    三段式 HTML 获取策略：
    Layer 0: requests
    Layer 1: PWv1（轻量）
    Layer 2: PWv2（重模式，仅入口页 + IR 域）
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
    """允许：
    1. 与 base_url 同根域
    2. 与 entry_url 同根域
    3. 在允许的外部 IR/CDN（q4ir/q4cdn/sec.gov 等）
    """
    try:
        n = urlparse(url)
        if not n.netloc:
            return True  # 相对路径

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
    根据 URL + anchor_text + title 打分，返回 (score, tags, reasons)
    tags: ["annual_report", "sec_filings", ...]
    reasons: ["annual_report::annual report", "10-K::10-k", ...]
    """
    positive_rules = cfg["positive_rules"]
    negative_keywords = cfg["negative_keywords"]
    negative_penalty = cfg.get("negative_penalty", 3)

    text = f"{anchor_text or ''} {page_title or ''}".lower()
    url_l = url.lower()

    score = 0
    tags: set[str] = set()
    reasons: list[str] = []

    # 正向加分
    for rule in positive_rules:
        tag = rule["tag"]
        weight = rule["weight"]
        for kw in rule["keywords"]:
            kw_l = kw.lower()
            if kw_l in url_l or kw_l in text:
                score += weight
                tags.add(tag)
                reasons.append(f"{tag}::{kw}")

    # 负向减分
    for kw in negative_keywords:
        kw_l = kw.lower()
        if kw_l in url_l or kw_l in text:
            score -= negative_penalty
            reasons.append(f"NEG::{kw}")

    return score, sorted(tags), reasons


def classify_level(score: int, cfg: dict) -> str | None:
    th = cfg["score_thresholds"]
    if score >= th["HIGH"]:
        return "HIGH"
    if score >= th["MEDIUM"]:
        return "MEDIUM"
    if score >= th["LOW"]:
        return "LOW"
    return None  # 分太低，不纳入候选（不过如果有 PDF 我们会破例）


# ========= NEW: PDF 相关辅助函数 =========

def extract_years_from_text(text: str) -> set[str]:
    """
    从文本中提取 1900-2099 的 4 位年份，返回 set[str]。
    """
    if not text:
        return set()
    years = set(re.findall(r"\b(19[0-9]{2}|20[0-9]{2})\b", text))
    return years


def analyze_pdfs_on_page(html: str, page_url: str) -> tuple[int, dict | None]:
    """
    在当前页面中扫描所有 <a>，识别 PDF 链接并打分。
    返回:
      - pdf_count: 此页上 PDF 链接数量
      - best_pdf: 最佳 PDF 候选信息 dict 或 None：
        {
          "pdf_url": str,
          "score": int,
          "years": "2022,2023",
          "is_10k": bool,
          "anchor_text": str,
        }
    """
    # 一些典型“不是年报/10-K”的噪声 PDF（fees、表单、charter 等）
    NEG_PDF_KEYWORDS = [
        "patriot",              # CapitalOne 那个 patriot_2025.pdf
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

        # # 只看 pdf
        # if not href_low.endswith(".pdf"):
        #     continue

        # pdf_count += 1

        # # 收集年份
        # years = extract_years_from_text(text_low) | extract_years_from_text(href_low)

        # is_10k = ("10-k" in text_low) or ("10k" in text_low) or ("form 10-k" in text_low) or ("10-k" in href_low)

        # score = 0
        # # 基础分：是 PDF
        # score += 10

        # # 文本/链接里提到 annual report / annual
        # if "annual report" in text_low or "annual report" in href_low:
        #     score += 10
        # elif "annual" in text_low or "annual" in href_low:
        #     score += 5

        # # 10-K 加分
        # if is_10k:
        #     score += 8

        # # 有年份的加分，多年合计
        # if years:
        #     # 最近几年（>=2019）的稍微多加一点
        #     for y in years:
        #         try:
        #             y_int = int(y)
        #             if y_int >= 2019:
        #                 score += 4
        #             else:
        #                 score += 2
        #         except ValueError:
        #             score += 2
        
        # 只看 pdf
        if not href_low.endswith(".pdf"):
            continue

        # 噪声 PDF 过滤：文件名或文本里出现典型无关词，直接跳过
        if any(kw in href_low or kw in text_low for kw in NEG_PDF_KEYWORDS):
            continue

        pdf_count += 1

        # 收集年份
        years = extract_years_from_text(text_low) | extract_years_from_text(href_low)

        is_10k = ("10-k" in text_low) or ("10k" in text_low) or ("form 10-k" in text_low) or ("10-k" in href_low)

        score = 0
        # 基础分：是 PDF
        score += 10

        # 文本/链接里提到 annual report / annual
        if "annual report" in text_low or "annual report" in href_low:
            score += 20      # 年报强相关，直接 +20
        elif "annual" in text_low or "annual" in href_low:
            score += 10

        # 10-K 加分（仅次于 annual）
        if is_10k:
            score += 15

        # 有年份的加分，多年合计
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


# ========= 主搜索逻辑（BFS） =========


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
    从某个银行的起始 URL 出发，BFS 收集候选入口页 + PDF 信息。
    返回记录列表，每条包含：
      - score / level / tags / reasons
      - pdf_count_on_page / best_pdf_* 等
    """
    crawl_cfg = cfg["crawl"]
    max_depth = crawl_cfg["max_depth"]
    max_pages = crawl_cfg["max_pages_per_bank"]

    visited: set[str] = set()
    # 队列元素: (url, depth, parent_url, anchor_text)
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

        # 先分析当前页面上的 PDF 链接（为下载脚本做准备）
        pdf_count, best_pdf = analyze_pdfs_on_page(html, url)  # NEW

        # 1) 对当前页面打分，判断是否纳入“候选入口页”
        score, tags, reasons = score_page(
            url=url,
            anchor_text=anchor_text or "",
            page_title=page_title,
            cfg=cfg,
        )
        level = classify_level(score, cfg)

        # ---- NEW: 入筛条件拆开 & 放宽 ----
        important_tags = {"annual_report", "10-K", "sec_filings", "financial_reports", "investor_relations"}

        has_important_tag = any(t in important_tags for t in tags)
        has_positive_score = score > 0

        # 满足任一条件就写入候选：
        # 1）页面上有 PDF
        # 2）有正分
        # 3）命中重要 tag
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


        # 2) 决定是否扩展下一层链接（BFS）
        if depth == max_depth:
            continue

        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            new_url = urljoin(url, href)
            new_url_l = new_url.lower()

            # 静态资源过滤
            if any(
                new_url_l.endswith(ext)
                for ext in [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".css", ".js", ".ico"]
            ):
                continue

            if not is_allowed_domain(new_url, base_root, entry_root):
                continue

            link_text = a.get_text(" ", strip=True) or ""
            link_text_low = link_text.lower()

            # depth = 0 时，稍微放宽，只要和 investor/financial 有点关系就扩展
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
                # depth >= 1 时，收紧，只扩展更像财报/年报的链接
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

    # 只取 financial report = YES 的行
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

        # entry_url 作为财报入口（优先），没有就退回 base_url
        entry_url = row.get("financial report link") or base_url
        # 允许按 username 覆盖起始入口（跨域 IR 平台等特殊情况）
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
