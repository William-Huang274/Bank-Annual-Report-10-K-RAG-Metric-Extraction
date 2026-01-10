import os
import re
import math
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple
import sys

import pandas as pd
import requests

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

# ======== 配置区（你只要改这里）========
# 输入 CSV：可以用你现在那份（有 url / score / pdf_type / year）
INPUT_CSV = ROOT/"data"/"interim"/"output"/"log"/"bank_candidate_entry_pages_for_pdf_2rd.csv" 

# 输出结果 CSV（会重写）
OUTPUT_CSV = ROOT/"data"/"interim"/"output"/"log"/"download_results.csv"

# PDF 保存根目录
OUTPUT_ROOT = ROOT/"data"/"raw"/"pdf_downloads"

# 最低下载分数（默认 0 = 不按分数过滤）
MIN_SCORE_FOR_DOWNLOAD = 0.0

# 请求配置
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # 每次失败后 *2
TIMEOUT = 40

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
}


# ======== 工具函数 ========

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def slugify(text: str, max_len: int = 60) -> str:
    """
    把银行名 / 用户名这些变成安全的文件名片段。
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    # 把空白和连字符统一
    text = re.sub(r"[ \t\n\r]+", "_", text)
    # 只保留字母数字和少量符号
    text = re.sub(r"[^0-9A-Za-z_\-]+", "", text)
    if not text:
        text = "unknown"
    return text[:max_len]


def is_pdf_url(url: str) -> bool:
    """
    粗判：URL 后缀是不是 pdf
    """
    if not isinstance(url, str):
        return False
    u = url.lower().split("?", 1)[0].split("#", 1)[0]
    return u.endswith(".pdf")


def parse_year_from_url_or_text(url: str, extra: str = "") -> Optional[int]:
    """
    从 URL 或别的文本里抓一个像样的年份（2000~今年+1）
    """
    text = f"{url} {extra}"
    years = re.findall(r"(20\d{2})", text)
    if not years:
        return None

    years_int = []
    current = time.localtime().tm_year
    for y in years:
        yi = int(y)
        if 2000 <= yi <= current + 1:
            years_int.append(yi)

    if not years_int:
        return None

    # 一般选最大的那个（最新年报）
    return max(years_int)


def classify_pdf_type_from_row(row: pd.Series) -> str:
    """
    从现有行推断一个大致的 pdf_type，
    如果 CSV 已经给了 pdf_type 就直接用。
    """
    if "pdf_type" in row and isinstance(row["pdf_type"], str) and row["pdf_type"].strip():
        return row["pdf_type"].strip()

    url = str(row.get("url", "")).lower()
    name = str(row.get("username", "")).lower()

    text = f"{url} {name}"

    if "10-k" in text or "form10k" in text.replace(" ", ""):
        return "10k"
    if "annual-report" in text or "annualreport" in text.replace(" ", "") or "annual review" in text:
        return "annual_report"
    if "proxy" in text and "statement" in text:
        return "proxy"
    if "presentation" in text or "slides" in text or "deck" in text:
        return "presentation"
    if "quarterly" in text or re.search(r"q[1-4]\s*20\d{2}", text):
        return "quarterly"
    if "8-k" in text or "6-k" in text:
        return "sec_other"

    return "other"


def download_pdf(
    session: requests.Session,
    url: str,
    dest_path: str,
    referer: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    负责真正下载 PDF，有重试和简单错误信息。
    返回 (success, reason)
    """
    if not isinstance(url, str) or not url.strip():
        return False, "empty_url"

    url = url.strip()

    headers = HEADERS.copy()
    if referer:
        headers["Referer"] = referer

    last_err = ""
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=headers, timeout=TIMEOUT, stream=True)
            if resp.status_code >= 400:
                last_err = f"http_{resp.status_code}"
            else:
                # 简单检查 content-type
                ct = resp.headers.get("Content-Type", "").lower()
                if "pdf" not in ct and not is_pdf_url(url):
                    # 仍然保存，但标记一下
                    last_err = f"non_pdf_content_type:{ct}"

                ensure_dir(os.path.dirname(dest_path))
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # 写完再简单验证一下文件大小
                if os.path.getsize(dest_path) == 0:
                    last_err = "empty_file"
                    # 删除空文件
                    try:
                        os.remove(dest_path)
                    except OSError:
                        pass
                    # 继续重试
                else:
                    # 成功，可能带一个非致命 warning
                    if last_err and "non_pdf_content_type" in last_err:
                        return True, last_err
                    return True, "ok"

        except Exception as e:
            last_err = f"exception:{e.__class__.__name__}"

        # 失败了，准备下一轮
        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= RETRY_BACKOFF

    return False, last_err or "unknown_error"


# ======== 主逻辑 ========

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"找不到输入 CSV：{INPUT_CSV}")

    print(f"读取输入：{INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # 1）检查必要列：这里已经用的是 best_pdf_url
    for col in ["idrssd", "username", "best_pdf_url"]:
        if col not in df.columns:
            raise ValueError(f"输入 CSV 缺少必要列：{col}")

    # 2）兼容旧的 status / local_path
    for col in ["status", "reason", "local_path"]:
        if col not in df.columns:
            df[col] = ""

    # 3）用 best_pdf_url 做真正的 URL 列
    df["best_pdf_url"] = df["best_pdf_url"].fillna("").astype(str)
    df = df[df["best_pdf_url"].str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)

    # 创建一个统一使用的 url 列，后面所有逻辑还用 url 就行
    df["url"] = df["best_pdf_url"]

    total = len(df)
    print(f"有效记录数：{total}")

    session = requests.Session()

    # 结果列表
    results = []

    for idx, row in df.iterrows():
        idrssd = row.get("idrssd", "")
        username = str(row.get("username", "")).strip()
        url = str(row.get("url", "")).strip()
        score = row.get("score", math.nan)

        # 默认先用原来的 pdf_type / year，如果没有就自动推断
        pdf_type = classify_pdf_type_from_row(row)
        year = row.get("year", None)

        # year 可能是 NaN / float / str
        if isinstance(year, float) and math.isnan(year):
            year = None
        if isinstance(year, str) and not year.strip():
            year = None

        if year is None:
            year = parse_year_from_url_or_text(url) or "unknown"

        # 分数过滤（默认 MIN_SCORE_FOR_DOWNLOAD=0 不过滤）
        status = ""
        reason = ""
        local_path = ""

        if not url.lower().startswith("http"):
            status = "skip_invalid_url"
            reason = "not_http_url"
        else:
            strong_hint = ("annual" in url.lower()
                           or "10-k" in url.lower()
                           or "annual-report" in url.lower())

            if (
                isinstance(score, (int, float))
                and not math.isnan(score)
                and score < MIN_SCORE_FOR_DOWNLOAD
                and not strong_hint
            ):
                status = "skip_low_score"
                reason = f"score<{MIN_SCORE_FOR_DOWNLOAD}"
            else:
                # 真正下载
                dir_name = os.path.join(
                    OUTPUT_ROOT,
                    slugify(str(pdf_type)),
                    str(year),
                    # 再嵌一层公司名，避免一个目录太多文件
                    f"{slugify(str(username))}_{slugify(str(idrssd))}",
                )
                ensure_dir(dir_name)

                # 文件名：取 URL 最后段 + idrssd
                url_last = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
                fname_part = url_last.rsplit("/", 1)[-1]
                fname_base = slugify(fname_part, max_len=80)
                if not fname_base.lower().endswith(".pdf"):
                    fname_base = fname_base + ".pdf"

                dest_path = os.path.join(dir_name, fname_base)

                # 已存在则不重复下载
                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                    status = "success"
                    reason = "exists"
                    local_path = dest_path
                else:
                    print(f"[{idx+1}/{total}] 下载 {username} | {pdf_type} | {year}")
                    ok, msg = download_pdf(session, url, dest_path)
                    if ok:
                        status = "success"
                        reason = msg
                        local_path = dest_path
                    else:
                        status = "fail"
                        reason = msg
                        local_path = ""

        results.append(
            {
                "idx": idx,
                "idrssd": idrssd,
                "username": username,
                "url": url,
                "score": score,
                "pdf_type": pdf_type,
                "year": year,
                "status": status,
                "reason": reason,
                "local_path": local_path,
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n完成。结果已写入：{OUTPUT_CSV}")
    print(out_df["status"].value_counts())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("发生未捕获错误：")
        traceback.print_exc()
