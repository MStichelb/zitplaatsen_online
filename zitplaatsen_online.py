# app_streamlit.py
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.utils import ImageReader

import io, os, math, json, zipfile, shutil, tempfile, random

st.set_page_config(page_title="Zitplaatsen", layout="wide")

# =========================
# PDF CROP-PARAMETERS (exacte waarden gebruikt door desktop)
# =========================
PDF_COLS = 5
PDF_PHOTO_W = 236
PDF_PHOTO_H = 236
PDF_MARGIN_LEFT = 140
PDF_MARGIN_TOP = 319
PDF_H_SPACING = 40
PDF_V_SPACING = 88
PDF_DPI = 200

# =========================
# UI/Render instellingen (kleine subset)
# =========================
PAGE_MARGIN_LR = 28
PAGE_MARGIN_TOP = 56
PAGE_MARGIN_BOTTOM = 24
INNER_PAD_X = 8
INNER_PAD_TOP = 8
INNER_PAD_BOTTOM = 12
SEAT_SPACING = 8
ROW_SPACING = 28
BANK_SPACING = 24
SEAT_MIN = 60
SEAT_MAX = 130
CAPTION_GAP = 8
FONT_MAX = 12
FONT_MIN = 7

# =========================
# Layouts definitie (gebruik jouw concrete list)
# =========================
LAYOUTS = {
    "Type 1 (T107) — 5 rijen × 3 banken × 2 stoelen (lange klas)": {
        "regular": True, "rows": 5, "banks": 3, "seats": 2, "orientation": "portrait"
    },
    "Type 2a (T009) — 4 rijen × (2-4-2) (brede klas)": {
        "regular": False,
        "pattern": [[2,4,2], [2,4,2], [2,4,2], [2,4,2]],
        "orientation": "landscape",
        "center_first_row": True
    },
    "Type 2b (T125) — 4 rijen × 4 banken × 2 stoelen (brede klas)": {
        "regular": True, "rows": 4, "banks": 4, "seats": 2, "orientation": "landscape"
    },
    "T105 — 3 rijen × 4 banken × 2 stoelen (brede klas)": {
        "regular": True, "rows": 3, "banks": 4, "seats": 2, "orientation": "landscape"
    },
    "Aardrijkskunde T106 — 6 rijen × 3 banken × 2 stoelen (lange klas)": {
        "regular": True, "rows": 6, "banks": 3, "seats": 2, "orientation": "portrait"
    },
    "Labo T117 — 4 rijen × 2 banken × 4 stoelen (brede klas)": {
        "regular": True, "rows": 4, "banks": 2, "seats": 4, "orientation": "landscape"
    },
    "Labo T120 — 4 rijen × (3-2-2-3) (brede klas)": {
        "regular": False,
        "pattern": [[3,2,2,3], [3,2,2,3], [3,2,2,3], [3,2,2,3]],
        "orientation": "landscape",
        "center_first_row": True
    },
    "Fysica T121 — 3 rijen × 3 banken × 3 stoelen + rij met 4 stoelen (brede klas)": {
        "regular": False,
        "pattern": [[4], [3,3,3], [3,3,3], [3,3,3]],
        "orientation": "landscape",
        "center_first_row": True
    },
    "Bio T122 — 5 rijen × (1-2-2-1) (lange klas)": {
        "regular": False,
        "pattern": [[1,2,2,1],[1,2,2,1],[1,2,2,1],[1,2,2,1],[1,2,2,1]],
        "orientation": "portrait",
        "center_first_row": True
    },
    "Artistiek T123 — 3 rijen × 5 banken × 2 stoelen (brede klas)": {
        "regular": True, "rows": 3, "banks": 5, "seats": 2, "orientation": "landscape"
    },
    "Eigen opstelling": {
        "regular": True, "rows": 4, "banks": 3, "seats": 2, "orientation": "portrait"
    }
}

def parse_pattern_text(raw: str):
    if raw is None:
        raise ValueError("Leeg patroon")
    s = raw.strip()
    if not s:
        raise ValueError("Leeg patroon")
    s = s.replace("], [", ";").replace("],[", ";").replace("][", ";")
    s = s.replace("[", "").replace("]", "")
    s = s.replace("\n", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    pattern = []
    for p in parts:
        nums = [x.strip() for x in p.split(",") if x.strip()]
        if not nums:
            raise ValueError("Lege rij in patroon")
        row = []
        for n in nums:
            if not n.isdigit():
                raise ValueError(f"Niet-numerieke waarde in patroon: '{n}'")
            v = int(n)
            if v <= 0:
                raise ValueError("Alle aantallen moeten > 0 zijn")
            row.append(v)
        pattern.append(row)
    return pattern

# -------------------------
# Helpers + session_state init
# -------------------------
def ensure_state():
    if "students" not in st.session_state:
        st.session_state.students = []  # list of dicts: name, pil (PIL Image), slot, font_size, source, pdf_index
    if "selected_slot" not in st.session_state:
        st.session_state.selected_slot = None
    if "layout_name" not in st.session_state:
        st.session_state.layout_name = list(LAYOUTS.keys())[0]
    if "base_slots" not in st.session_state:
        st.session_state.base_slots = []  # export geometry
    if "layout_custom" not in st.session_state:
        st.session_state.layout_custom = LAYOUTS.get("Eigen opstelling")
    if "page_size" not in st.session_state:
        st.session_state.page_size = portrait(A4)

ensure_state()

# -------------------------
# Compute base geometry (same math as desktop)
# Returns base_slots (list) and base_bank_rects
# -------------------------
def compute_base_geometry(layout_name):
    cfg = LAYOUTS.get(layout_name, LAYOUTS[list(LAYOUTS.keys())[0]])
    orient = cfg.get("orientation", "portrait")
    page = portrait(A4) if orient == "portrait" else landscape(A4)
    W, H = page

    regular = cfg.get("regular", True)
    if regular:
        rows = cfg["rows"]
        banks_per_row = [cfg["banks"]] * rows
        def seats_lookup(r, c): return cfg["seats"]
    else:
        pattern = cfg["pattern"]
        rows = len(pattern)
        banks_per_row = [len(row) for row in pattern]
        def seats_lookup(r, c): return pattern[r][c]

    max_banks = max(banks_per_row) if banks_per_row else 0

    max_seats_in_widest_row = 0
    for r in range(rows):
        seats_list = [seats_lookup(r, c) for c in range(banks_per_row[r])]
        max_seats_in_widest_row = max(max_seats_in_widest_row, max(seats_list) if seats_list else 0)

    avail_w = W - PAGE_MARGIN_LR*2 - (max_banks-1)*BANK_SPACING
    seats_per_bank_for_width = cfg["seats"] if regular else (max_seats_in_widest_row or 1)
    seat_by_w = (avail_w / max_banks - 2*INNER_PAD_X - (seats_per_bank_for_width-1)*SEAT_SPACING) / max(seats_per_bank_for_width,1)

    font_est = 14
    avail_h = H - PAGE_MARGIN_TOP - PAGE_MARGIN_BOTTOM - (rows-1)*ROW_SPACING
    seat_by_h = avail_h/rows - (INNER_PAD_TOP + CAPTION_GAP + font_est + INNER_PAD_BOTTOM)

    seat_size = int(max(SEAT_MIN, min(SEAT_MAX, seat_by_w, seat_by_h)))

    def bank_w_base(seats):
        return int(2*INNER_PAD_X + seats*seat_size + (seats-1)*SEAT_SPACING)
    bank_h_base = int(INNER_PAD_TOP + seat_size + CAPTION_GAP + font_est + INNER_PAD_BOTTOM)

    # compute base slots
    base_slots = []
    base_bank_rects = []
    y_base = PAGE_MARGIN_TOP + max(0, (H - PAGE_MARGIN_TOP - PAGE_MARGIN_BOTTOM - (rows*bank_h_base + (rows-1)*ROW_SPACING))//2)
    title_y = 24
    if y_base < title_y + 10:
        y_base = title_y + 16

    for r in range(rows):
        row_banks = banks_per_row[r]
        if regular:
            row_bank_widths = [bank_w_base(cfg["seats"]) for _ in range(row_banks)]
        else:
            row_bank_widths = [bank_w_base(seats_lookup(r, c)) for c in range(row_banks)]
        row_total_w = sum(row_bank_widths) + (row_banks-1)*BANK_SPACING
        x_base = PAGE_MARGIN_LR + (W - 2*PAGE_MARGIN_LR - row_total_w)//2
        for b in range(row_banks):
            seats = seats_lookup(r,b) if not regular else cfg["seats"]
            bw = row_bank_widths[b]
            x0b, y0b = x_base, y_base
            x1b, y1b = x0b + bw, y0b + bank_h_base
            base_bank_rects.append((x0b, y0b, x1b, y1b))
            sx = x0b + INNER_PAD_X
            sy = y0b + INNER_PAD_TOP
            for s in range(seats):
                base_slots.append({
                    "x": sx, "y": sy, "w": seat_size, "h": seat_size,
                    "cx": sx + seat_size/2, "cy": sy + seat_size/2
                })
                sx += seat_size + SEAT_SPACING
            x_base += bw + BANK_SPACING
        y_base += bank_h_base + ROW_SPACING

    return page, base_slots, base_bank_rects, seat_size

# -------------------------
# Utilities for students management
# -------------------------
def crop_square(pil_img):
    w,h = pil_img.size
    side = min(w,h)
    L = (w - side)//2
    T = (h - side)//2
    return pil_img.crop((L,T,L+side,T+side))

def add_student_from_pil(pil, name=""):
    pil_sq = crop_square(pil)
    st.session_state.students.append({
        "name": name or "leerling",
        "pil": pil_sq,
        "slot": None,
        "font_size": FONT_MAX,
        "source": None,
        "pdf_index": None,
        "img_filename": None
    })

def auto_assign_students_to_slots():
    # ensure base_slots exist
    if not st.session_state.base_slots:
        return
    used = set(s["slot"] for s in st.session_state.students if s.get("slot") is not None and isinstance(s.get("slot"), int) and s["slot"] < len(st.session_state.base_slots))
    free = [i for i in range(len(st.session_state.base_slots)) if i not in used]
    for s in st.session_state.students:
        if s.get("slot") is None or not isinstance(s.get("slot"), int) or s["slot"] >= len(st.session_state.base_slots):
            if free:
                s["slot"] = free.pop(0)
            else:
                s["slot"] = None

# -------------------------
# Sidebar UI: uploads, layout selection, actions
# -------------------------
st.sidebar.title("Zitplaatsen — instellingen")

# class & room
st.sidebar.text_input("Klas (bv. 3A):", key="class_name", value="klas")
st.sidebar.text_input("Lokaal (bv. T107):", key="room_name", value="lokaal")

# layout selector
layout_choice = st.sidebar.selectbox("Opstelling:", list(LAYOUTS.keys()), index=list(LAYOUTS.keys()).index(st.session_state.layout_name))
if layout_choice != st.session_state.layout_name:
    st.session_state.layout_name = layout_choice
    # recompute geometry
    page, base_slots, base_bank_rects, seat_size = compute_base_geometry(st.session_state.layout_name)
    st.session_state.page_size = page
    st.session_state.base_slots = base_slots
    st.session_state.base_bank_rects = base_bank_rects
    auto_assign_students_to_slots()

# Custom layout popup-like inputs
if st.sidebar.button("Eigen opstelling instellen"):
    st.sidebar.info("Vul hieronder je eigen opstelling (regelmatig of patroon).")
    # show fields
    regular = st.sidebar.radio("Regulier of onregelmatig?", ("Regulier", "Onregelmatig"))
    if regular == "Regulier":
        rows = st.sidebar.number_input("Rijen:", min_value=1, max_value=12, value=4, key="custom_rows")
        banks = st.sidebar.number_input("Banken per rij:", min_value=1, max_value=10, value=3, key="custom_banks")
        seats = st.sidebar.number_input("Zitplaatsen per bank:", min_value=1, max_value=8, value=2, key="custom_seats")
        orient = st.sidebar.selectbox("Oriëntatie:", ("portrait","landscape"), index=0, key="custom_orient")
        LAYOUTS["Eigen opstelling"] = {"regular": True, "rows": rows, "banks": banks, "seats": seats, "orientation": orient}
    else:
        raw = st.sidebar.text_area("Patroon (bv: [4],[3,3,3],[3,3,3]) of 4;3,3,3;3,3,3", value=" [4],[3,3,3],[3,3,3]")
        try:
            pattern = parse_pattern_text(raw)
            orient = st.sidebar.selectbox("Oriëntatie:", ("portrait","landscape"), index=0, key="custom_orient2")
            LAYOUTS["Eigen opstelling"] = {"regular": False, "pattern": pattern, "orientation": orient, "center_first_row": True}
        except Exception as e:
            st.sidebar.error(f"Patroon fout: {e}")
    st.experimental_rerun()

# Upload images (multiple) or PDF
st.sidebar.markdown("---")
st.sidebar.header("Importeer foto's")
import_method = st.sidebar.radio("Bron", ("Losse afbeeldingen", "Smartschool PDF"))
if import_method == "Losse afbeeldingen":
    uploaded = st.sidebar.file_uploader("Upload afbeeldingen (meerdere)", accept_multiple_files=True, type=["jpg","jpeg","png"])
    if uploaded:
        # prompt names
        default_names = [os.path.splitext(f.name)[0] for f in uploaded]
        names_text = st.sidebar.text_area("Namen (één per regel) — optioneel. Laat leeg = bestandsnamen", value="\n".join(default_names))
        names = [ln.strip() for ln in names_text.splitlines() if ln.strip()]
        for i, f in enumerate(uploaded):
            try:
                pil = Image.open(io.BytesIO(f.read())).convert("RGB")
            except Exception:
                continue
            name = names[i] if i < len(names) else os.path.splitext(f.name)[0]
            add_student_from_pil(pil, name=name)
        st.sidebar.success(f"{len(uploaded)} foto's toegevoegd.")
        st.experimental_rerun()
else:
    pdf_file = st.sidebar.file_uploader("Upload Smartschool PDF (1e pagina wordt gebruikt)", type=["pdf"])
    if pdf_file is not None:
        raw = pdf_file.read()
        try:
            pages = convert_from_bytes(raw, dpi=PDF_DPI)
            page = pages[0].convert("RGB")
            st.sidebar.image(page, caption="PDF preview (pagina 1)", use_column_width=True)
            N = st.sidebar.number_input("Aantal leerlingen op de PDF", min_value=1, max_value=200, value=10)
            names_text = st.sidebar.text_area("Namen (één per regel) — optioneel. Laat leeg = leerling_1..N", value="\n".join([f"leerling_{i+1}" for i in range(int(N))]))
            names = [ln.strip() for ln in names_text.splitlines() if ln.strip()]
            if st.sidebar.button("Knip foto's uit PDF en voeg toe"):
                for i in range(int(N)):
                    r = i // PDF_COLS
                    c = i % PDF_COLS
                    x1 = PDF_MARGIN_LEFT + c * (PDF_PHOTO_W + PDF_H_SPACING)
                    y1 = PDF_MARGIN_TOP + r * (PDF_PHOTO_H + PDF_V_SPACING)
                    x2 = x1 + PDF_PHOTO_W
                    y2 = y1 + PDF_PHOTO_H
                    crop = page.crop((x1, y1, x2, y2))
                    pil_sq = crop_square(crop)
                    name = names[i] if i < len(names) else f"leerling_{i+1}"
                    st.session_state.students.append({
                        "name": name,
                        "pil": pil_sq,
                        "slot": None,
                        "font_size": FONT_MAX,
                        "source": "uploaded_pdf",
                        "pdf_index": i,
                        "img_filename": None
                    })
                st.sidebar.success(f"{int(N)} foto's uit PDF toegevoegd.")
                st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"PDF lezen mislukt: {e}")

# Actions
st.sidebar.markdown("---")
if st.sidebar.button("Shuffle"):
    random.shuffle(st.session_state.students)
    # reassign
    for i, s in enumerate(st.session_state.students):
        s["slot"] = i if i < len(st.session_state.base_slots) else None
    st.experimental_rerun()

# Save / Load seating (JSON + zip of images)
def save_seating_to_file():
    if not st.session_state.students:
        st.sidebar.warning("Geen leerlingen om op te slaan.")
        return
    # create temp dir
    tmp = tempfile.mkdtemp()
    assets_dir = os.path.join(tmp, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    students_meta = []
    for i, s in enumerate(st.session_state.students):
        fname = f"{i}_{s['name'].replace(' ','_')}.png"
        p = os.path.join(assets_dir, fname)
        try:
            s["pil"].save(p, format="PNG")
        except Exception:
            Image.new("RGB",(100,100),(240,240,240)).save(p, format="PNG")
        students_meta.append({
            "name": s["name"],
            "slot": s.get("slot"),
            "font_size": s.get("font_size", FONT_MAX),
            "source": s.get("source"),
            "pdf_index": s.get("pdf_index"),
            "img_filename": fname
        })
    data = {
        "class": st.session_state.get("class_name","klas"),
        "room": st.session_state.get("room_name","lokaal"),
        "layout": st.session_state.layout_name,
        "custom_layout": LAYOUTS.get("Eigen opstelling"),
        "students": students_meta
    }
    # write json and zip assets
    json_path = os.path.join(tmp, "seating.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    zip_out = io.BytesIO()
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="seating.json")
        for fname in os.listdir(assets_dir):
            zf.write(os.path.join(assets_dir,fname), arcname=fname)
    st.sidebar.download_button("Download opstelling (.zip)", data=zip_out.getvalue(), file_name=f"{data['class']}_{data['room']}_seating.zip")

if st.sidebar.button("Opslaan opstelling"):
    save_seating_to_file()

# Load seating (upload zip)
upload_seating = st.sidebar.file_uploader("Open opstelling (.zip)", type=["zip"])
if upload_seating is not None:
    with tempfile.TemporaryDirectory() as td:
        z = zipfile.ZipFile(io.BytesIO(upload_seating.read()))
        z.extractall(td)
        jsonp = os.path.join(td, "seating.json")
        if not os.path.isfile(jsonp):
            st.sidebar.error("Zip bevat geen seating.json")
        else:
            with open(jsonp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # clear current
            st.session_state.students = []
            st.session_state.layout_name = data.get("layout", st.session_state.layout_name)
            if "custom_layout" in data and data["custom_layout"]:
                LAYOUTS["Eigen opstelling"] = data["custom_layout"]
            page, base_slots, base_bank_rects, seat_size = compute_base_geometry(st.session_state.layout_name)
            st.session_state.page_size = page
            st.session_state.base_slots = base_slots
            st.session_state.base_bank_rects = base_bank_rects
            for meta in data.get("students", []):
                imgfile = meta.get("img_filename")
                pil = None
                if imgfile:
                    p = os.path.join(td, imgfile)
                    if os.path.isfile(p):
                        try:
                            pil = Image.open(p).convert("RGB")
                        except Exception:
                            pil = None
                if pil is None:
                    pil = Image.new("RGB",(seat_size,seat_size),(240,240,240))
                st.session_state.students.append({
                    "name": meta.get("name","leerling"),
                    "pil": pil,
                    "slot": meta.get("slot"),
                    "font_size": meta.get("font_size", FONT_MAX),
                    "source": meta.get("source"),
                    "pdf_index": meta.get("pdf_index"),
                    "img_filename": imgfile
                })
            auto_assign_students_to_slots()
            st.experimental_rerun()

# -------------------------
# Main area: compute geometry if not present
# -------------------------
if not st.session_state.base_slots:
    page, base_slots, base_bank_rects, seat_size = compute_base_geometry(st.session_state.layout_name)
    st.session_state.page_size = page
    st.session_state.base_slots = base_slots
    st.session_state.base_bank_rects = base_bank_rects
else:
    page = st.session_state.page_size
    base_slots = st.session_state.base_slots
    base_bank_rects = st.session_state.base_bank_rects
    seat_size = base_slots[0]["w"] if base_slots else 100

auto_assign_students_to_slots()

# -------------------------
# Helper: find student by slot
# -------------------------
def student_at_slot(idx):
    for i, s in enumerate(st.session_state.students):
        if s.get("slot") == idx:
            return i, s
    return None, None

# -------------------------
# Seat interaction: select & swap
# -------------------------
def on_click_slot(idx):
    sel = st.session_state.selected_slot
    if sel is None:
        # select if occupied
        si, s = student_at_slot(idx)
        if s:
            st.session_state.selected_slot = idx
    else:
        # attempt swap (move or swap)
        if sel == idx:
            st.session_state.selected_slot = None
            return
        si_sel, s_sel = student_at_slot(sel)
        si_tgt, s_tgt = student_at_slot(idx)
        if s_sel is None:
            st.session_state.selected_slot = None
            return
        if s_tgt is None:
            # move
            s_sel["slot"] = idx
        else:
            # swap slots
            s_sel["slot"], s_tgt["slot"] = s_tgt["slot"], s_sel["slot"]
        st.session_state.selected_slot = None

# -------------------------
# Rendering the layout visually in columns
# We create rows; for each bank we create a mini-column with seats inline
# -------------------------
st.title("Zitplaatsen — Web (Streamlit)")

col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Opstelling")
    # we'll render rows using st.columns; create a visual representation
    cfg = LAYOUTS.get(st.session_state.layout_name)
    regular = cfg.get("regular", True)
    if regular:
        rows = cfg["rows"]
        banks_per_row = [cfg["banks"]] * rows
        def seats_lookup(r,c): return cfg["seats"]
    else:
        pattern = cfg["pattern"]
        rows = len(pattern)
        banks_per_row = [len(row) for row in pattern]
        def seats_lookup(r,c): return pattern[r][c]

    for r in range(rows):
        # determine widths: create column placeholders per bank with spaces between
        row_banks = banks_per_row[r]
        bank_contents = []
        for b in range(row_banks):
            seats = seats_lookup(r,b) if not regular else cfg["seats"]
            # build a small column for this bank: seats horizontally inside
            cols = st.columns(seats)
            for s_i in range(seats):
                # compute the global slot index: it's the same ordering as base_slots
                # find the slot index corresponding to (r,b,s_i)
                # we can compute it by iterating base_slots until we reach the match ordinal
                # simpler: keep a counter
                pass
    # to map row/bank/seat -> base_slots index, rebuild mapping
    mapping = []
    idx = 0
    for r in range(rows):
        for b in range(banks_per_row[r]):
            seats = seats_lookup(r,b) if not regular else cfg["seats"]
            for s_i in range(seats):
                mapping.append((r,b,s_i, idx))
                idx += 1

    # Render by rows using mapping
    current = 0
    for r in range(rows):
        banks = [m for m in mapping if m[0]==r]
        # create streamlit columns for the entire row: one column per seat but we want spacing between banks
        # we will create as many columns as number of seats in row, rendering separators by empty columns
        seats_in_row = sum(seats_lookup(r,c) if not regular else cfg["seats"] for c in range(banks_per_row[r]))
        cols = st.columns(seats_in_row + (banks_per_row[r]-1))  # rough: add separators slots
        col_idx = 0
        bank_counter = 0
        # iterate banks
        for b in range(banks_per_row[r]):
            seats = seats_lookup(r,b) if not regular else cfg["seats"]
            for s_i in range(seats):
                _,_,_, slot_index = mapping[current]
                si, s = student_at_slot(slot_index)
                with cols[col_idx]:
                    if s:
                        # highlight if selected
                        selected = (st.session_state.selected_slot == slot_index)
                        st.image(s["pil"], use_column_width=True)
                        if selected:
                            st.markdown(f"**{s['name']}** ✅")
                        else:
                            st.markdown(f"**{s['name']}**")
                        if st.button("Select / Swap", key=f"sel_{slot_index}"):
                            on_click_slot(slot_index)
                            st.experimental_rerun()
                        if st.button("Bewerk naam", key=f"edit_{slot_index}"):
                            new = st.text_input("Nieuwe naam", value=s["name"], key=f"ni_{slot_index}")
                            if st.button("OK", key=f"ok_{slot_index}"):
                                s["name"] = new
                                st.experimental_rerun()
                        if st.button("Verwijder", key=f"del_{slot_index}"):
                            st.session_state.students.pop(si)
                            st.experimental_rerun()
                    else:
                        st.image(Image.new("RGB",(100,100),(240,240,240)), use_column_width=True)
                        st.markdown("_leeg_")
                        if st.button("Select / Swap", key=f"sel_{slot_index}"):
                            on_click_slot(slot_index)
                            st.experimental_rerun()
                current += 1
                col_idx += 1
            # add separator column if there is space
            if bank_counter < banks_per_row[r]-1:
                # try to place an empty spacer column
                if col_idx < len(cols):
                    with cols[col_idx]:
                        st.write("")  # spacer
                    col_idx += 1
                bank_counter += 1

with col2:
    st.subheader("Acties")
    st.write("Geselecteerde plek:", st.session_state.selected_slot)
    if st.button("Reset selectie"):
        st.session_state.selected_slot = None
        st.experimental_rerun()
    st.markdown("---")
    st.write("Leerlingen aantal:", len(st.session_state.students))
    if st.session_state.students:
        sel_idx = None
        if st.button("Verplaats lege naar eerstvolgende lege plek"):
            # reassign sequentially based on students order
            free = [i for i in range(len(st.session_state.base_slots)) if all(s.get("slot") != i for s in st.session_state.students)]
            for s_i, s in enumerate(st.session_state.students):
                if s_i < len(st.session_state.base_slots):
                    s["slot"] = s_i
            st.experimental_rerun()

    st.markdown("---")
    if st.button("Exporteer naar PDF"):
        # create PDF in memory
        cfg = LAYOUTS.get(st.session_state.layout_name)
        page = portrait(A4) if cfg.get("orientation","portrait")=="portrait" else landscape(A4)
        W,H = page
        packet = io.BytesIO()
        c = pdfcanvas.Canvas(packet, pagesize=page)
        # title
        c.setFont("Helvetica-Bold", 20)
        title = f"Klas {st.session_state.get('class_name','klas')} — Lokaal {st.session_state.get('room_name','lokaal')}"
        c.drawCentredString(W/2, H-36, title)
        # banks
        c.setLineWidth(1)
        for (x0,y0,x1,y1) in st.session_state.base_bank_rects:
            c.rect(x0, H - y1, x1-x0, y1-y0, stroke=1, fill=0)
        # placeholders dotted
        c.setDash(2,2)
        for slot in st.session_state.base_slots:
            x,y,w,h = slot["x"], slot["y"], slot["w"], slot["h"]
            c.rect(x, H-(y+h), w, h, stroke=1, fill=0)
        c.setDash()
        oversample = 2
        for s in st.session_state.students:
            if s.get("slot") is None or s.get("slot") >= len(st.session_state.base_slots):
                continue
            slot = st.session_state.base_slots[s["slot"]]
            x,y,w,h = slot["x"], slot["y"], slot["w"], slot["h"]
            thumb = s["pil"].resize((max(1, w*oversample), max(1, h*oversample)), Image.LANCZOS)
            ir = ImageReader(thumb)
            c.drawImage(ir, x, H-(y+h), width=w, height=h, preserveAspectRatio=True, mask='auto')
            ui_font_size = int(s.get("font_size", FONT_MAX))
            ui_font_size = max(FONT_MIN, min(FONT_MAX, ui_font_size))
            c.setFont("Helvetica-Bold", ui_font_size)
            c.drawCentredString(x + w/2, H-(y+h+CAPTION_GAP+12), s["name"])
        c.showPage()
        c.save()
        packet.seek(0)
        st.download_button("Download PDF", data=packet.getvalue(), file_name=f"{st.session_state.get('class_name','klas')}_{st.session_state.get('room_name','lokaal')}.pdf", mime="application/pdf")

st.caption("Tip: klik op 'Select / Swap' op een leerling en daarna op de doelplek om te wisselen. Gebruik 'Eigen opstelling' om zelf een patroon in te voeren.")
