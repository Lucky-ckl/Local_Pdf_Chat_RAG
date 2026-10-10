"""生成 P5.1 的合成扫描件样本（image-only PDF，无文本层）。

用途：为评测集提供「扫描件问题」类型（ocr_required=true）所需的文档。
方法：先用 PyMuPDF 把中英文文字画到页面，渲染成 200 DPI 位图，
      再把位图整页插入一个新 PDF —— 结果页面没有文本层，只能靠 OCR 提取。

离线运行，不下载模型、不联网。用法：
    .\.venv\Scripts\python.exe eval\samples\make_scan_sample.py [输出路径]
"""

import pathlib
import sys

import pymupdf

LINES = [
    "设备保养与质保通知",
    "设备名称：HPU-01 液压泵站",
    "制造商：成都液压设备有限公司",
    "首次保养：运行 500 小时更换吸油滤芯",
    "保养周期：每 2000 小时更换液压油",
    "保修期限：验收合格后 12 个月",
    "服务热线：400-816-2233",
    "存放要求：环境温度 5 至 40 摄氏度",
]

FONT_CANDIDATES = ["china-s", "china-ss"]


def _render_source_page(width=760, height=420, fontsize=20):
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    y = 55
    last_error = None
    for fontname in FONT_CANDIDATES:
        try:
            probe = pymupdf.open()
            p = probe.new_page(width=width, height=height)
            for i, line in enumerate(LINES):
                p.insert_text((40, y + i * 40), line, fontsize=fontsize, fontname=fontname)
            probe.close()
            last_error = None
            break
        except Exception as exc:  # 该内置字体不可用，尝试下一个
            last_error = exc
    if last_error is not None:
        raise last_error
    for i, line in enumerate(LINES):
        page.insert_text((40, y + i * 40), line, fontsize=fontsize, fontname="china-s")
    return doc


def build(out_path: pathlib.Path, dpi=200) -> pathlib.Path:
    src = _render_source_page()
    pix = src[0].get_pixmap(dpi=dpi)
    src.close()

    out = pymupdf.open()
    page = out.new_page(width=pix.width * 72 / dpi, height=pix.height * 72 / dpi)
    page.insert_image(page.rect, pixmap=pix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    out.close()
    return out_path


def main():
    default = pathlib.Path(__file__).resolve().parent / "scan_maintenance_notice.pdf"
    target = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default
    path = build(target)
    doc = pymupdf.open(path)
    text_layer = "".join(p.get_text() for p in doc)
    print(f"written: {path}")
    print(f"pages: {doc.page_count}, text-layer chars: {len(text_layer)} (0 = image-only)")
    doc.close()


if __name__ == "__main__":
    main()
