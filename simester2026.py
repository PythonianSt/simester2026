import streamlit as st
import pandas as pd
import requests
import base64
import json
import uuid
import io
import qrcode
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="ระบบตรวจสุขภาพนักศึกษา",
    page_icon="🩺",
    layout="centered"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_OWNER = st.secrets["GITHUB_OWNER"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
CSV_PATH = st.secrets.get("CSV_PATH", "student_physical_exam_master.csv")
APP_BASE_URL = st.secrets["APP_BASE_URL"].rstrip("/")

BKK = ZoneInfo("Asia/Bangkok")


# =========================
# HELPER FUNCTIONS
# =========================

def now_bkk():
    return datetime.now(BKK).strftime("%Y-%m-%d %H:%M:%S")


def github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }


def github_file_url():
    return f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{CSV_PATH}"


def load_csv_from_github():
    url = github_file_url()
    r = requests.get(
        url,
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH}
    )

    if r.status_code == 404:
        df = pd.DataFrame(columns=[
            "visit_id",
            "student_code",
            "registry_time_bkk",
            "vs_time_bkk",

            "bp_sys",
            "bp_dia",
            "pulse",
            "spo2",
            "weight_kg",
            "height_cm",
            "bmi",

            "bp_color",
            "pulse_color",
            "spo2_color",
            "bmi_color",

            "vs_confirmed_by",
            "status"
        ])
        return df, None

    r.raise_for_status()
    data = r.json()
    sha = data["sha"]
    content = base64.b64decode(data["content"]).decode("utf-8")

    df = pd.read_csv(io.StringIO(content), dtype=str)
    return df, sha


def save_csv_to_github(df, sha=None, message="update CSV"):
    csv_text = df.to_csv(index=False)
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")

    payload = {
        "message": message,
        "content": encoded,
        "branch": GITHUB_BRANCH
    }

    if sha:
        payload["sha"] = sha

    r = requests.put(
        github_file_url(),
        headers=github_headers(),
        json=payload
    )

    if r.status_code not in [200, 201]:
        st.error(f"GitHub save error: {r.status_code} {r.text}")
        return False

    return True


def generate_qr_image(text):
    qr = qrcode.QRCode(
        version=1,
        box_size=10,
        border=3
    )
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    return img.convert("RGB")


def image_to_base64(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_vs_with_gpt(bp_img, spo2_img, bw_img, ht_img):
    images = [
        ("BP and pulse monitor", bp_img),
        ("SpO2 monitor", spo2_img),
        ("body weight display", bw_img),
        ("height display", ht_img),
    ]

    content = [
        {
            "type": "text",
            "text": """
You are extracting vital sign values from device photos.

Return JSON only.

Fields:
{
  "bp_sys": number or null,
  "bp_dia": number or null,
  "pulse": number or null,
  "spo2": number or null,
  "weight_kg": number or null,
  "height_cm": number or null,
  "uncertain_fields": []
}

Rules:
- BP systolic and diastolic must be separated.
- Pulse may appear on BP machine or pulse oximeter.
- SpO2 is percentage.
- Weight is kg.
- Height is cm.
- If unclear, return null and list the field in uncertain_fields.
"""
        }
    ]

    for label, img in images:
        b64 = image_to_base64(img)
        content.append({
            "type": "text",
            "text": f"Image: {label}"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def calc_bmi(weight_kg, height_cm):
    try:
        w = float(weight_kg)
        h = float(height_cm) / 100
        if w <= 0 or h <= 0:
            return None
        return round(w / (h * h), 1)
    except Exception:
        return None


def classify_bp(sys, dia):
    try:
        sys = float(sys)
        dia = float(dia)

        if sys >= 140 or dia >= 90:
            return "RED"
        elif sys >= 120 or dia >= 80:
            return "YELLOW"
        else:
            return "GREEN"
    except Exception:
        return "UNKNOWN"


def classify_pulse(pulse):
    try:
        p = float(pulse)

        if p > 120 or p < 45:
            return "RED"
        elif p > 100 or p < 50:
            return "YELLOW"
        else:
            return "GREEN"
    except Exception:
        return "UNKNOWN"


def classify_spo2(spo2):
    try:
        s = float(spo2)

        if s < 93:
            return "RED"
        elif s < 96:
            return "YELLOW"
        else:
            return "GREEN"
    except Exception:
        return "UNKNOWN"


def classify_bmi(bmi):
    try:
        b = float(bmi)

        if b >= 30 or b < 18.5:
            return "RED"
        elif b >= 25:
            return "YELLOW"
        else:
            return "GREEN"
    except Exception:
        return "UNKNOWN"


def color_badge(color):
    if color == "GREEN":
        return "🟢"
    if color == "YELLOW":
        return "🟡"
    if color == "RED":
        return "🔴"
    return "⚪"


# =========================
# QUERY PARAMS
# =========================

params = st.query_params
page = params.get("page", "home")
visit_id_from_url = params.get("visit_id", "")


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🩺 Student Physical Exam")
st.sidebar.write("ระบบตรวจสุขภาพนักศึกษา")

if st.sidebar.button("🏠 Home"):
    st.query_params.clear()
    st.rerun()

if st.sidebar.button("📝 Registry"):
    st.query_params["page"] = "registry"
    st.rerun()

if st.sidebar.button("❤️ VS Station"):
    st.query_params["page"] = "vs"
    st.rerun()

if st.sidebar.button("🔳 Station QR"):
    st.query_params["page"] = "station_qr"
    st.rerun()


# =========================
# HOME
# =========================

if page == "home":
    st.title("🩺 ระบบตรวจสุขภาพนักศึกษา")
    st.info(
        "ระบบนี้ใช้ QR code สำหรับติดตามนักศึกษาแต่ละรายในแต่ละสถานี "
        "และบันทึกข้อมูลลง GitHub CSV"
    )

    st.markdown("""
    ### ลำดับการใช้งาน

    1. Registry station สร้าง QR เฉพาะของนักศึกษา
    2. นักศึกษาถือ QR ติดตัว
    3. ห้อง VS มี QR เดียวสำหรับเข้า VS page
    4. ถ่ายภาพเครื่องวัด BP/P, SpO₂, BW, Ht
    5. GPT ช่วยอ่านข้อความจากภาพ
    6. เจ้าหน้าที่ยืนยันก่อนบันทึก
    """)


# =========================
# REGISTRY PAGE
# =========================

elif page == "registry":
    st.title("📝 Registry Station")
    st.caption("สร้าง QR code เฉพาะสำหรับนักศึกษา")

    if st.button("➕ สร้าง Student QR ใหม่"):
        df, sha = load_csv_from_github()

        today = datetime.now(BKK).strftime("%Y%m%d")
        visit_id = str(uuid.uuid4())
        student_code = f"KUPE-{today}-{len(df) + 1:04d}"

        student_url = f"{APP_BASE_URL}/?page=student_card&visit_id={visit_id}"

        new_row = {
            "visit_id": visit_id,
            "student_code": student_code,
            "registry_time_bkk": now_bkk(),
            "vs_time_bkk": "",

            "bp_sys": "",
            "bp_dia": "",
            "pulse": "",
            "spo2": "",
            "weight_kg": "",
            "height_cm": "",
            "bmi": "",

            "bp_color": "",
            "pulse_color": "",
            "spo2_color": "",
            "bmi_color": "",

            "vs_confirmed_by": "",
            "status": "REGISTERED"
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        ok = save_csv_to_github(
            df,
            sha,
            message=f"Registry new student {student_code}"
        )

        if ok:
            st.success("สร้าง Student QR และบันทึกลง GitHub CSV สำเร็จ")

            st.subheader("Student code")
            st.code(student_code)

            st.subheader("Student QR")
            qr_img = generate_qr_image(student_url)
            st.image(qr_img, width=280)

            buffer = io.BytesIO()
            qr_img.save(buffer, format="PNG")
            st.download_button(
                "⬇️ ดาวน์โหลด QR PNG",
                data=buffer.getvalue(),
                file_name=f"{student_code}_QR.png",
                mime="image/png"
            )

            st.caption("ให้นักศึกษาถือ QR นี้ไปทุกสถานี")


# =========================
# STUDENT CARD PAGE
# =========================

elif page == "student_card":
    st.title("🎫 Student QR Card")

    if not visit_id_from_url:
        st.error("ไม่พบ visit_id")
        st.stop()

    df, sha = load_csv_from_github()
    row = df[df["visit_id"] == visit_id_from_url]

    if row.empty:
        st.error("ไม่พบข้อมูลนักศึกษารายนี้")
        st.stop()

    student_code = row.iloc[0]["student_code"]

    st.success("พบข้อมูลนักศึกษา")
    st.subheader(student_code)
    st.code(visit_id_from_url)

    qr_img = generate_qr_image(f"{APP_BASE_URL}/?page=student_card&visit_id={visit_id_from_url}")
    st.image(qr_img, width=280)

    st.info("แสดง QR นี้ที่แต่ละสถานีเพื่อบันทึกข้อมูล")


# =========================
# STATION QR PAGE
# =========================

elif page == "station_qr":
    st.title("🔳 Station QR Codes")

    st.subheader("VS Station QR")
    vs_url = f"{APP_BASE_URL}/?page=vs"
    vs_qr = generate_qr_image(vs_url)

    st.image(vs_qr, width=280)
    st.code(vs_url)

    buffer = io.BytesIO()
    vs_qr.save(buffer, format="PNG")
    st.download_button(
        "⬇️ ดาวน์โหลด VS Station QR",
        data=buffer.getvalue(),
        file_name="VS_station_QR.png",
        mime="image/png"
    )

    st.caption("ติด QR นี้ไว้ที่ห้อง VS")


# =========================
# VS PAGE
# =========================

elif page == "vs":
    st.title("❤️ VS Station")
    st.caption("ถ่ายภาพเครื่องวัด แล้วให้ GPT ช่วยอ่านข้อความ")

    st.warning(
        "ระบบนี้เป็นเครื่องมือช่วยอ่านค่าเท่านั้น "
        "ต้องให้เจ้าหน้าที่ยืนยันค่าก่อนบันทึกทุกครั้ง"
    )

    df, sha = load_csv_from_github()

    st.subheader("1) ระบุ Student QR")

    visit_id = st.text_input(
        "กรอกหรือสแกน visit_id จาก Student QR",
        value=visit_id_from_url
    )

    if not visit_id:
        st.info("กรุณาสแกน Student QR หรือกรอก visit_id")
        st.stop()

    row = df[df["visit_id"] == visit_id]

    if row.empty:
        st.error("ไม่พบ visit_id นี้ในระบบ")
        st.stop()

    idx = row.index[0]
    student_code = df.loc[idx, "student_code"]

    st.success(f"พบข้อมูล: {student_code}")

    st.subheader("2) ถ่ายภาพ VS")

    bp_img = st.camera_input("📷 ภาพที่ 1: เครื่องวัด BP และ Pulse")
    spo2_img = st.camera_input("📷 ภาพที่ 2: เครื่องวัด SpO₂")
    bw_img = st.camera_input("📷 ภาพที่ 3: เครื่องชั่งน้ำหนัก BW")
    ht_img = st.camera_input("📷 ภาพที่ 4: เครื่องวัดส่วนสูง Ht")

    if bp_img and spo2_img and bw_img and ht_img:
        if st.button("🤖 อ่านค่าจากภาพด้วย GPT"):
            with st.spinner("กำลังอ่านข้อความจากภาพ..."):
                result = extract_vs_with_gpt(
                    bp_img,
                    spo2_img,
                    bw_img,
                    ht_img
                )

            st.session_state["vs_result"] = result
            st.success("อ่านค่าเบื้องต้นสำเร็จ กรุณาตรวจสอบและยืนยัน")

    if "vs_result" in st.session_state:
        result = st.session_state["vs_result"]

        st.subheader("3) ยืนยันค่าโดยเจ้าหน้าที่")

        col1, col2 = st.columns(2)

        with col1:
            bp_sys = st.number_input(
                "Systolic BP",
                value=float(result.get("bp_sys") or 0),
                step=1.0
            )
            bp_dia = st.number_input(
                "Diastolic BP",
                value=float(result.get("bp_dia") or 0),
                step=1.0
            )
            pulse = st.number_input(
                "Pulse",
                value=float(result.get("pulse") or 0),
                step=1.0
            )

        with col2:
            spo2 = st.number_input(
                "SpO₂",
                value=float(result.get("spo2") or 0),
                step=1.0
            )
            weight_kg = st.number_input(
                "Weight kg",
                value=float(result.get("weight_kg") or 0),
                step=0.1
            )
            height_cm = st.number_input(
                "Height cm",
                value=float(result.get("height_cm") or 0),
                step=0.1
            )

        bmi = calc_bmi(weight_kg, height_cm)

        bp_color = classify_bp(bp_sys, bp_dia)
        pulse_color = classify_pulse(pulse)
        spo2_color = classify_spo2(spo2)
        bmi_color = classify_bmi(bmi)

        st.subheader("4) Traffic color")

        st.write(f"BP: {color_badge(bp_color)} {bp_color}")
        st.write(f"Pulse: {color_badge(pulse_color)} {pulse_color}")
        st.write(f"SpO₂: {color_badge(spo2_color)} {spo2_color}")
        st.write(f"BMI: {color_badge(bmi_color)} {bmi_color}")

        if result.get("uncertain_fields"):
            st.warning(f"ค่าที่ GPT ไม่มั่นใจ: {result['uncertain_fields']}")

        confirmed_by = st.text_input("ผู้ยืนยันข้อมูล", value="nurse")

        if st.button("✅ ยืนยันและบันทึกลง GitHub CSV"):
            df.loc[idx, "vs_time_bkk"] = now_bkk()
            df.loc[idx, "bp_sys"] = str(bp_sys)
            df.loc[idx, "bp_dia"] = str(bp_dia)
            df.loc[idx, "pulse"] = str(pulse)
            df.loc[idx, "spo2"] = str(spo2)
            df.loc[idx, "weight_kg"] = str(weight_kg)
            df.loc[idx, "height_cm"] = str(height_cm)
            df.loc[idx, "bmi"] = str(bmi)

            df.loc[idx, "bp_color"] = bp_color
            df.loc[idx, "pulse_color"] = pulse_color
            df.loc[idx, "spo2_color"] = spo2_color
            df.loc[idx, "bmi_color"] = bmi_color

            df.loc[idx, "vs_confirmed_by"] = confirmed_by
            df.loc[idx, "status"] = "VS_DONE"

            ok = save_csv_to_github(
                df,
                sha,
                message=f"Update VS for {student_code}"
            )

            if ok:
                st.success("บันทึก VS ลง GitHub CSV สำเร็จ")
                st.balloons()


# =========================
# UNKNOWN PAGE
# =========================

else:
    st.error("ไม่พบหน้าที่ต้องการ")