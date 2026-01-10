import re
import csv
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ========= 配置区域 =========
INPUT_CSV = r"D:\Annual report LLMproject\input\SampleBank_Website_to collect_2nd_round_output.csv"
OUTPUT_CSV = r"D:\Annual report LLMproject\output\bank_annual_pdf_links_multi_year.csv"

# 建议：先开小样本测试，比如 LIMIT_BANKS = 20，没问题再跑全量
LIMIT_BANKS = 20  # 比如先写 20 调试，确认后改成 None

REQUEST_TIMEOUT = 15
MAX_DEPTH = 2          # BFS 深度
MAX_PAGES_PER_BANK = 20  # 每家银行最多抓多少个页面（防炸）

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

YEAR_PATTERN = re.compile(r"(20[01][0-9]|202[0-5])")  # 2000-2025 大致范围


# ========= 工具函数 =========

def safe_get(session: requests.Session, url: str) -> str | None:
    """安全请求网页，失败返回 None。"""
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400:
            print(f"[WARN] {resp.status_code} for {url}")
            return None
        # 部分网站编码比较乱，直接用 resp.text
        return resp.text
    except Exception as e:
        print(f"[ERROR] Exception for {url}: {e}")
        return None


def is_probably_same_domain(base_url: str, new_url: str) -> bool:
    """限制 BFS 不要跑飞：只在相同域名或者子域名内游走。"""
    try:
        b = urlparse(base_url)
        n = urlparse(new_url)
        if not n.netloc:
            return True  # 相对路径
        return b.netloc.split(".")[-2:] == n.netloc.split(".")[-2:]
    except Exception:
        return True


def detect_year(text_or_url: str) -> int | None:
    """从文本或 URL 中提取年份（优先取最后一个）。"""
    matches = YEAR_PATTERN.findall(text_or_url)
    if not matches:
        return None
    return int(matches[-1])


def detect_report_type(text: str, url: str) -> str:
    """简单判断报告类型：annual / 10-K / other。"""
    t = (text or "").lower()
    u = (url or "").lower()

    if "10-k" in t or "10-k" in u or "10k" in t or "10k" in u:
        return "10-K"

    if "annual" in t or "annual" in u:
        return "annual_report"

    if "proxy" in t or "proxy" in u:
        return "proxy"

    if "sec" in t or "sec" in u:
        return "sec_filing"

    return "other"


def score_pdf_link(text: str, url: str) -> int:
    """对每个 PDF 链接做一个简单打分，用于后续筛选。"""
    t = (text or "").lower()
    u = (url or "").lower()
    score = 0

    # 文件本身
    if ".pdf" in u:
        score += 3

    # 年份
    if YEAR_PATTERN.search(t) or YEAR_PATTERN.search(u):
        score += 3

    # 关键词
    for kw, s in [
        ("annual", 4),
        ("10-k", 4),
        ("10k", 4),
        ("report", 2),
        ("financial", 2),
        ("sec", 1),
        ("statement", 1),
    ]:
        if kw in t or kw in u:
            score += s

    return score


def extract_pdf_links_from_page(
    html: str,
    page_url: str,
    base_url: str,
) -> list[dict]:
    """从单个页面中提取 PDF 链接，返回记录列表。"""
    soup = BeautifulSoup(html, "html.parser")

    results = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(page_url, href)

        # 必须是 PDF 文件
        if ".pdf" not in full_url.lower():
            continue

        # 限制域名
        if not is_probably_same_domain(base_url, full_url):
            continue

        link_text = a.get_text(" ", strip=True) or ""
        score = score_pdf_link(link_text, full_url)
        if score <= 0:
            continue

        year = detect_year(link_text) or detect_year(full_url)
        report_type = detect_report_type(link_text, full_url)

        results.append(
            {
                "source_page_url": page_url,
                "pdf_url": full_url,
                "link_text": link_text,
                "year": year,
                "report_type": report_type,
                "score": score,
            }
        )

    return results


def bfs_collect_pdfs_for_bank(
    session: requests.Session,
    base_url: str,
    entry_url: str,
) -> list[dict]:
    """
    BFS 小抓取：从 entry_url 出发，最多爬 MAX_DEPTH 层、MAX_PAGES_PER_BANK 页，
    收集所有满足条件的 PDF 链接。
    """
    queue: list[tuple[str, int]] = [(entry_url, 0)]
    visited: set[str] = set()

    collected: list[dict] = []
    pages_count = 0

    while queue and pages_count < MAX_PAGES_PER_BANK:
        url, depth = queue.pop(0)
        if url in visited or depth > MAX_DEPTH:
            continue
        visited.add(url)

        html = safe_get(session, url)
        if not html:
            continue

        pages_count += 1
        print(f"   [PAGE] depth={depth} url={url}")

        # 1) 当前页先抓 PDF
        page_pdfs = extract_pdf_links_from_page(html, url, base_url)
        collected.extend(page_pdfs)

        # 2) 再决定是否扩展下一层链接
        if depth == MAX_DEPTH:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            new_url = urljoin(url, href)
            new_url_l = new_url.lower()

            # 限制域名
            if not is_probably_same_domain(base_url, new_url):
                continue

            # 避免静态文件等
            if any(
                new_url_l.endswith(ext)
                for ext in [".pdf", ".jpg", ".png", ".css", ".js"]
            ):
                continue

            # 只扩展“疑似财报相关”的链接
            if not any(
                kw in new_url_l
                for kw in [
                    "annual",
                    "report",
                    "10-k",
                    "10k",
                    "sec-filings",
                    "financials",
                    "financial-information",
                ]
            ):
                continue

            if new_url not in visited:
                queue.append((new_url, depth + 1))

    return collected


# ========= 主流程 =========

def main():
    df = pd.read_csv(INPUT_CSV)

    # 只抓 financial report = YES 的银行
    banks = df[df["financial report"] == "YES"].copy()

    if LIMIT_BANKS is not None:
        banks = banks.head(LIMIT_BANKS)

    print(f"Total banks to process: {len(banks)}")

    session = requests.Session()

    output_rows = []

    for idx, row in banks.iterrows():
        idrssd = row.get("idrssd")
        username = row.get("username")
        base_url = row.get("url")

        entry_url = row.get("financial report link") or base_url
        if not isinstance(entry_url, str) or not entry_url.startswith("http"):
            # 如果没带协议，补一个
            if isinstance(entry_url, str):
                entry_url = "https://" + entry_url.lstrip("/")
            else:
                print(f"[SKIP] No valid entry_url for {username} ({base_url})")
                continue

        print("=" * 80)
        print(f"[BANK] idrssd={idrssd} username={username}")
        print(f" base_url={base_url}")
        print(f" entry_url={entry_url}")

        try:
            pdf_records = bfs_collect_pdfs_for_bank(
                session=session,
                base_url=base_url,
                entry_url=entry_url,
            )

            if not pdf_records:
                print("  -> No PDF found.")
                continue

            # 去重：同一 pdf_url 只保留最高分那条
            dedup = {}
            for rec in pdf_records:
                url = rec["pdf_url"]
                if url not in dedup or rec["score"] > dedup[url]["score"]:
                    dedup[url] = rec

            for rec in dedup.values():
                output_rows.append(
                    {
                        "idrssd": idrssd,
                        "username": username,
                        "base_url": base_url,
                        "entry_url": entry_url,
                        **rec,
                    }
                )

            print(f"  -> Collected {len(dedup)} unique pdf links.")

        except Exception as e:
            print(f"[ERROR] Bank {username} exception: {e}")
            output_rows.append(
                {
                    "idrssd": idrssd,
                    "username": username,
                    "base_url": base_url,
                    "entry_url": entry_url,
                    "source_page_url": None,
                    "pdf_url": None,
                    "link_text": None,
                    "year": None,
                    "report_type": None,
                    "score": None,
                    "error": str(e),
                }
            )

        # 轻微 sleep，避免太猛
        time.sleep(1)

    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"\n[DONE] Wrote {len(output_rows)} rows to {OUTPUT_CSV}")
    else:
        print("\n[DONE] No pdf links collected.")


if __name__ == "__main__":
    main()
