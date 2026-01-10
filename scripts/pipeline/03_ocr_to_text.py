import subprocess
from pathlib import Path
import csv
from typing import Optional
import os         
import fitz  # PyMuPDF
import shutil  
import sys
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

CUSTOM_TEMP = ROOT/"_tmp"/"ocr_temp"
os.makedirs(CUSTOM_TEMP, exist_ok=True)

os.environ["TMP"] = CUSTOM_TEMP
os.environ["TEMP"] = CUSTOM_TEMP
os.environ["TMPDIR"] = CUSTOM_TEMP

# ======== 配置区 ========

YEAR = "2024"

PDF_RAW_DIR = ROOT/"data"/"raw"/"pdf_downloads"/"annual_report"/ YEAR
PDF_OCR_DIR = ROOT/"data"/"interim"/"output"/ YEAR
TXT_OUT_DIR = ROOT/"data"/"interim"/"txt"/ YEAR

# 结果日志
LOG_CSV = ROOT/"data"/"interim"/"output"/"log"/f"ocr_extract_{YEAR}.csv"

# 根据 CPU 自动估一个适合的并发数
CPU_COUNT = os.cpu_count() or 4
# 留 1–2 个核给系统，不要全打满
NUM_JOBS = max(CPU_COUNT - 2, 1)

PDF_OCR_DIR.mkdir(parents=True, exist_ok=True)
TXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

print(f"[INFO] 检测到 CPU {CPU_COUNT} 核，ocrmypdf 使用 --jobs {NUM_JOBS}")

# ocrmypdf 命令，可按需要加参数
OCR_CMD_BASE = [
    "ocrmypdf",
    "--force-ocr",          # 全部强制 OCR
    "--deskew",             # 纠偏
    "--optimize", "1",      # 轻度压缩
    "--skip-big", "200",    # 超大文件跳过（MB）
    "--jobs", str(NUM_JOBS) # ✅ 开多进程并行页级 OCR
]
# ========================


def pdf_has_text(pdf_path: Path, max_pages: int = 5) -> bool:
    """
    粗略判断 PDF 是否已有可抽取文本（前 max_pages 页非空就认为有）
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return False

    pages_to_check = min(len(doc), max_pages)
    for i in range(pages_to_check):
        text = doc.load_page(i).get_text("text").strip()
        if text:
            return True
    return False


# def ocr_pdf_if_needed(src: Path, dst: Path) -> str:
#     """
#     对单个 PDF 做 OCR：
#     - 已有文本的话：直接复制（这里简单起见，仍用 ocrmypdf 但很快）
#     - 返回 status: "ok", "skip", "fail"
#     """
#     dst.parent.mkdir(parents=True, exist_ok=True)

#     cmd = OCR_CMD_BASE + [str(src), str(dst)]
#     try:
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             check=False,
#         )
#     except Exception as e:
#         print(f"[OCR ERROR] {src} -> {dst}: {e}")
#         return "fail"

#     if result.returncode == 0:
#         print(f"[OCR OK] {src.name}")
#         return "ok"
#     else:
#         print(f"[OCR FAIL] {src.name} (code={result.returncode})")
#         print("  stdout:", result.stdout[:500])
#         print("  stderr:", result.stderr[:500])
#         return "fail"

def ocr_pdf_if_needed(src: Path, dst: Path) -> str:
    # 先确保目录存在
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 如果已经有 OCR 后 pdf 了，直接复用
    if dst.exists():
        print(f"[OCR SKIP] 目标已存在，直接复用：{dst}")
        return "ok"

    cmd = OCR_CMD_BASE + [str(src), str(dst)]
    print(f"[OCR] 开始 OCR: {src.name}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    rc = result.returncode

    if rc == 0:
        print(f"[OCR OK] {src.name}")
        return "ok"

    if rc == 3:
        # 有文本层，不需要 OCR
        print(f"[OCR SKIP] {src.name}：PDF 已有文本层，跳过 OCR")
        try:
            shutil.copy2(src, dst)  # 保证后面统一用 dst 抽文本
        except Exception as e:
            print(f"[OCR SKIP WARN] 复制原始 PDF 到 {dst} 失败: {e}")
        return "ok"

    # 其它返回码视为真正失败
    print(f"[OCR FAIL] {src.name} (code={rc})")
    print("stdout:", result.stdout[:300])
    print("stderr:", result.stderr[:300])
    return "fail"


def extract_text(pdf_path: Path, txt_path: Path) -> bool:
    """
    用 PyMuPDF 把 PDF 文本抽到 txt
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        txt = "\n".join(parts)
    except Exception as e:
        print(f"[TXT FAIL] {pdf_path}: {e}")
        return False

    try:
        txt_path.write_text(txt, encoding="utf-8", errors="ignore")
        print(f"[TXT OK] {txt_path}")
        return True
    except Exception as e:
        print(f"[TXT WRITE FAIL] {txt_path}: {e}")
        return False


def iter_all_pdfs(root: Path):
    """
    遍历所有银行文件夹下的 pdf
    """
    for bank_dir in sorted(root.glob("*")):
        if not bank_dir.is_dir():
            continue
        for pdf in sorted(bank_dir.glob("*.pdf")):
            yield bank_dir.name, pdf


def infer_bank_and_year(bank_folder: str, pdf_name: str) -> tuple[str, Optional[int]]:
    """
    简单推一下 bank_name 和年份：
    - bank_name 就用文件夹名
    - 年份：在文件名里找 20xx，没有的话用 YEAR
    """
    import re

    bank = bank_folder
    years = re.findall(r"(20\d{2})", pdf_name)
    yr = int(years[0]) if years else int(YEAR)
    # 像 2025-SAR-Annual-Report 这种，可以后面再做规则映射成 2024
    return bank, yr


def main():
    PDF_OCR_DIR.mkdir(parents=True, exist_ok=True)
    TXT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有 PDF，方便打印总进度
    all_pdfs = list(iter_all_pdfs(PDF_RAW_DIR))
    total = len(all_pdfs)
    print(f"[INFO] 在 {PDF_RAW_DIR} 下共找到 {total} 个 PDF 文件")
    if total == 0:
        return

    # 打开日志文件（注意：下面所有 writer.writerow 都必须在这个 with 里面）
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bank_folder",
            "pdf_file",
            "bank_name",
            "report_year",
            "raw_pdf_path",
            "ocr_pdf_path",
            "txt_path",
            "ocr_status",
            "txt_status",
        ])

        for idx, (bank_folder, pdf_path) in enumerate(all_pdfs, start=1):
            rel_bank_dir = bank_folder  # 如: DeutscheBank_214807
            bank_name, year = infer_bank_and_year(rel_bank_dir, pdf_path.name)

            print(f"\n[{idx}/{total}] >>> 开始处理：{rel_bank_dir} / {pdf_path.name}")

            ocr_pdf_path = PDF_OCR_DIR / rel_bank_dir / pdf_path.name
            txt_path = TXT_OUT_DIR / rel_bank_dir / (pdf_path.stem + ".txt")

            # 已有 txt：直接跳过，写一行日志
            if txt_path.exists():
                print(f"    [SKIP] 已存在 txt：{txt_path}")
                writer.writerow([
                    rel_bank_dir,
                    pdf_path.name,
                    bank_name,
                    year,
                    str(pdf_path),
                    str(ocr_pdf_path if ocr_pdf_path.exists() else ""),
                    str(txt_path),
                    "skip_exist",
                    "ok",
                ])
                continue

            # 先 OCR
            print(f"    [OCR] 开始 OCR：{pdf_path.name}")
            ocr_status = ocr_pdf_if_needed(pdf_path, ocr_pdf_path)
            if ocr_status != "ok":
                print(f"    [OCR FAIL] OCR 失败：{pdf_path.name}")
                writer.writerow([
                    rel_bank_dir,
                    pdf_path.name,
                    bank_name,
                    year,
                    str(pdf_path),
                    str(ocr_pdf_path),
                    str(txt_path),
                    ocr_status,
                    "fail",
                ])
                continue
            print(f"    [OCR OK] 输出 OCR pdf：{ocr_pdf_path}")

            # 抽取文本
            print(f"    [TXT] 开始抽取文本：{pdf_path.name}")
            txt_ok = extract_text(ocr_pdf_path, txt_path)  # ✅ 只传两个参数
            txt_status = "ok" if txt_ok else "fail"

            if txt_ok:
                print(f"    [TXT OK] 输出 txt：{txt_path}")
            else:
                print(f"    [TXT FAIL] 文本抽取失败：{txt_path}")

            # 统一写日志
            writer.writerow([
                rel_bank_dir,
                pdf_path.name,
                bank_name,
                year,
                str(pdf_path),
                str(ocr_pdf_path),
                str(txt_path),
                ocr_status,
                txt_status,
            ])


if __name__ == "__main__":
    main()
