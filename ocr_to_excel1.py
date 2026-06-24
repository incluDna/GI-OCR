import cv2
import easyocr
import re
import pandas as pd
import os
import glob
import numpy as np

# =========================
# CONFIG
# =========================

# IMAGE_PATH = "images1/1611.jpg"

reader = easyocr.Reader(["th", "en"], gpu=False)


# =========================
# PREPROCESS
# =========================

def preprocess_pink_bg(img):
    b, g, r = cv2.split(img)
    gray = b

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray, h=15)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )

    return thresh


# =========================
# HELPERS
# =========================

def normalize_text(text):
    return " ".join(str(text).split()).strip()


def only_digits(text):
    return re.sub(r"\D", "", str(text))


# =========================
# OCR
# =========================

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def ocr_to_items(image_path, debug=True):
    img = imread_unicode(image_path)

    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

    processed = preprocess_pink_bg(img)

    if debug:
        cv2.imwrite("debug_processed.jpg", processed)
        print("saved: debug_processed.jpg")

    results = reader.readtext(processed, detail=1, paragraph=False)

    items = []

    print("\n========== OCR RAW ==========")

    for bbox, text, conf in results:
        x = int(bbox[0][0])
        y = int(bbox[0][1])
        w = int(bbox[1][0]) - x
        h = int(bbox[2][1]) - y

        item = {
            "text": normalize_text(text),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": x + w // 2,
            "cy": y + h // 2,
            "conf": conf
        }

        items.append(item)

        print(
            f'text: {item["text"]} '
            f'-> x={x}, y={y}, w={w}, h={h}, '
            f'cx={item["cx"]}, cy={item["cy"]}, conf={conf:.2f}'
        )

    return items


# =========================
# COMMON FIELDS
# =========================

def find_common_fields(items):
    all_text = " ".join(item["text"] for item in items)

    date_match = re.search(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", all_text)
    date = date_match.group(0) if date_match else ""

    doc_match = re.search(r"\d{10,14}", only_digits(all_text))
    doc_no = doc_match.group(0) if doc_match else ""

    sno = ""
    branch_name = ""

    for item in items:
        text = item["text"]

        if "รหัส" in text or "สาขา" in text:
            nums = re.findall(r"\d{3,5}", text)

            if nums:
                sno = max(nums, key=len)
                idx = text.find(sno)
                branch_name = text[idx + len(sno):].strip(" :-.")
                break

    print("\n========== COMMON FIELDS ==========")
    print("วันที่        :", date)
    print("เลขที่เอกสาร :", doc_no)
    print("Sno.         :", sno)
    print("ชื่อสาขา     :", branch_name)

    return {
        "วันที่": date,
        "เลขที่เอกสาร": doc_no,
        "Sno.": sno,
        "ชื่อสาขา": branch_name
    }


# =========================
# ROW GROUPING
# =========================

def group_rows_by_y(items, y_threshold=80):
    sorted_items = sorted(items, key=lambda i: i["cy"])
    rows = []

    for item in sorted_items:
        placed = False

        for row in rows:
            avg_y = sum(i["cy"] for i in row) / len(row)

            if abs(item["cy"] - avg_y) <= y_threshold:
                row.append(item)
                placed = True
                break

        if not placed:
            rows.append([item])

    for row in rows:
        row.sort(key=lambda i: i["x"])

    return rows


# =========================
# ASSET PARSER
# =========================

def is_asset_code_item(item):
    digits = only_digits(item["text"])

    # รหัสทรัพย์สินส่วนใหญ่ 5-8 หลัก
    if not re.fullmatch(r"\d{5,8}", digits):
        return False

    # ต้องอยู่ฝั่งซ้ายของตาราง
    if item["x"] > 1200:
        return False

    return True


def parse_asset_row(row_items):
    row_items = sorted(row_items, key=lambda i: i["x"])

    code_item = None

    for item in row_items:
        if is_asset_code_item(item):
            code_item = item
            break

    if code_item is None:
        return None

    asset_code = only_digits(code_item["text"])
    code_x = code_item["x"]

    name_parts = []
    qty = ""
    remark_parts = []

    for item in row_items:
        if item is code_item:
            continue

        text = item["text"]
        digits = only_digits(text)

        # หมายเหตุ มักมีคำว่า โอนจาก หรืออยู่ขวาไกล
        if "โอนจาก" in text or item["x"] >= code_x + 1600:
            remark_parts.append(text)
            continue

        # จำนวน มักเป็นเลข 1-3 หลัก อยู่ด้านขวาของชื่อ
        if re.fullmatch(r"\d{1,3}", digits) and item["x"] > code_x + 500:
            qty = digits
            continue

        # ชื่อทรัพย์สิน อยู่ถัดจากรหัส
        if item["x"] > code_x + 150:
            name_parts.append(text)

    asset_name = normalize_text(" ".join(name_parts))
    remark = normalize_text(" ".join(remark_parts))

    if "โอนจาก" in asset_name:
        before, after = asset_name.split("โอนจาก", 1)
        asset_name = normalize_text(before)
        remark = normalize_text("โอนจาก" + after + " " + remark)

    return {
        "รหัสทรัพย์สิน": asset_code,
        "ชื่อทรัพย์สิน": asset_name,
        "จำนวน": qty if qty else "1",
        "หมายเหตุ": remark
    }


def parse_assets(items, debug=True):
    rows = group_rows_by_y(items, y_threshold=80)

    asset_rows = []

    print("\n========== ROW DEBUG ==========")

    for row in rows:
        row_text = " | ".join(item["text"] for item in row)

        parsed = parse_asset_row(row)

        if debug:
            print(row_text)
            if parsed:
                print("  -> parsed:", parsed)

        if parsed:
            parsed["NO."] = len(asset_rows) + 1
            asset_rows.append(parsed)

    return asset_rows


# =========================
# MAIN PARSER
# =========================

def parse_receipt_ocr(items, debug=True):
    print("\n========== PARSING RECEIPT OCR ==========")

    common = find_common_fields(items)
    assets = parse_assets(items, debug=debug)

    final_rows = []

    for asset in assets:
        final_rows.append({
            "วันที่": common["วันที่"],
            "เลขที่เอกสาร": common["เลขที่เอกสาร"],
            "Sno.": common["Sno."],
            "ชื่อสาขา": common["ชื่อสาขา"],
            "NO.": asset["NO."],
            "รหัสทรัพย์สิน": asset["รหัสทรัพย์สิน"],
            "ชื่อทรัพย์สิน": asset["ชื่อทรัพย์สิน"],
            "จำนวน": asset["จำนวน"],
            "หมายเหตุ": asset["หมายเหตุ"]
        })

    return final_rows

def export_to_excel(rows, filename="output3.xlsx"):
    df = pd.DataFrame(rows)

    columns = [
        "วันที่",
        "เลขที่เอกสาร",
        "Sno.",
        "ชื่อสาขา",
        "NO.",
        "รหัสทรัพย์สิน",
        "ชื่อทรัพย์สิน",
        "จำนวน",
        "หมายเหตุ"
    ]

    df = df[columns]

    df.to_excel(filename, index=False)

    print(f"\nบันทึกไฟล์แล้ว: {filename}")

# =========================
# RUN
# =========================

if __name__ == "__main__":

    IMAGE_FOLDER = "ใบGI"
    OUTPUT_FILE = "ocr_output5.xlsx"

    print("current working directory:", os.getcwd())
    print("image folder:", os.path.abspath(IMAGE_FOLDER))

    # หาไฟล์รูปทั้งหมดในโฟลเดอร์ ใบGI
    # รองรับชื่อไฟล์ที่มี space, ภาษาไทย, underscore ฯลฯ
    image_files = []

    for path in glob.glob(os.path.join(IMAGE_FOLDER, "*")):
        lower = path.lower().strip()

        if (
            lower.endswith(".jpg")
            or lower.endswith(".jpeg")
            or lower.endswith(".png")
            or lower.endswith(" jpg")
            or lower.endswith(" jpeg")
            or lower.endswith(" png")
        ):
            image_files.append(path)

    image_files = sorted(image_files)

    print("เจอไฟล์รูปทั้งหมด:", len(image_files))

    for f in image_files[:10]:
        print(" -", f)

    if len(image_files) == 0:
        print("ไม่เจอรูปในโฟลเดอร์ ใบGI")
        print("ให้เช็คว่าโฟลเดอร์ ใบGI อยู่ที่เดียวกับไฟล์ .py หรือไม่")
        exit()

    all_rows = []

    for image_path in image_files:

        print("\n" + "=" * 80)
        print("PROCESS:", image_path)
        print("=" * 80)

        try:
            items = ocr_to_items(image_path, debug=False)

            rows = parse_receipt_ocr(
                items,
                debug=False
            )

            print("rows found:", len(rows))

            for row in rows:
                row["ไฟล์ต้นฉบับ"] = os.path.basename(image_path)

            all_rows.extend(rows)

        except Exception as e:
            print("ERROR:", image_path)
            print(e)

    columns = [
        "ไฟล์ต้นฉบับ",
        "วันที่",
        "เลขที่เอกสาร",
        "Sno.",
        "ชื่อสาขา",
        "NO.",
        "รหัสทรัพย์สิน",
        "ชื่อทรัพย์สิน",
        "จำนวน",
        "หมายเหตุ"
    ]

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("\nไม่พบข้อมูลที่ parse ได้จากทุกรูป")
        print("แต่โปรแกรมเจอรูปจำนวน:", len(image_files))
        print("แปลว่า path รูปถูกแล้ว แต่ parser ยังจับรายการทรัพย์สินไม่ได้")
        exit()

    # ใช้ reindex แทน df[columns] เพื่อไม่ให้พังถ้าบาง column หาย
    df = df.reindex(columns=columns)

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print("\n====================")
    print("SAVE:", OUTPUT_FILE)
    print("TOTAL FILES:", len(image_files))
    print("TOTAL ROWS:", len(df))
    print("====================")