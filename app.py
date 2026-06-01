import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="TrustVision - Marquette AI Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --gold:    #F0A500;
    --cyan:    #00C9FF;
    --teal:    #00E5CC;
    --pink:    #FF6B9D;
    --bg:      #060B18;
    --card:    #111827;
    --border:  #1E2D4A;
    --text:    #E8EDF5;
    --muted:   #6B7FA3;
}

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(ellipse at 20% 50%, rgba(0,201,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(240,165,0,0.05) 0%, transparent 50%),
        var(--bg) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1020 0%, #060B18 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* Make the Streamlit top toolbar visible so toggle arrow works */
[data-testid="stToolbar"] { display: block !important; visibility: visible !important; }
.stAppHeader { display: block !important; visibility: visible !important; }

.hero {
    background: linear-gradient(135deg, #0D1B3E 0%, #0A2240 40%, #061830 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px 40px 30px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(240,165,0,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -20px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(0,201,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-dept {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 8px;
    animation: fadeSlideDown 0.6s ease both;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 6px;
    animation: fadeSlideDown 0.7s ease both;
}
.hero-title .t { color: #FFFFFF; }
.hero-title .v { color: var(--gold); }
.hero-tagline {
    font-size: 1rem;
    font-weight: 300;
    color: var(--cyan);
    font-style: italic;
    margin-bottom: 18px;
    animation: fadeSlideDown 0.8s ease both;
}
.hero-desc {
    font-size: 0.88rem;
    color: #8A9BBF;
    line-height: 1.7;
    max-width: 680px;
    margin-bottom: 20px;
    animation: fadeSlideDown 0.9s ease both;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    animation: fadeSlideDown 1s ease both;
}
.hero-tag {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid;
}
.tag-gold { color: var(--gold); border-color: rgba(240,165,0,0.4);   background: rgba(240,165,0,0.08); }
.tag-cyan { color: var(--cyan); border-color: rgba(0,201,255,0.4);   background: rgba(0,201,255,0.08); }
.tag-teal { color: var(--teal); border-color: rgba(0,229,204,0.4);   background: rgba(0,229,204,0.08); }
.tag-pink { color: var(--pink); border-color: rgba(255,107,157,0.4); background: rgba(255,107,157,0.08); }
.hero-location {
    position: absolute;
    top: 24px; right: 28px;
    font-size: 0.7rem;
    color: var(--muted);
    text-align: right;
    line-height: 1.6;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 20px 0 24px;
}
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    transition: transform 0.2s ease;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card.gold { border-top: 3px solid var(--gold); }
.stat-card.cyan { border-top: 3px solid var(--cyan); }
.stat-card.teal { border-top: 3px solid var(--teal); }
.stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 8px;
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
}
.stat-card.gold .stat-value { color: var(--gold); }
.stat-card.cyan .stat-value { color: var(--cyan); }
.stat-card.teal .stat-value { color: var(--teal); }

.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(0,201,255,0.15), rgba(0,229,204,0.15));
    border: 1px solid rgba(0,201,255,0.3);
    color: var(--cyan);
}

.sec-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin: 20px 0 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
}
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--border);
}

.sidebar-brand {
    background: linear-gradient(135deg, #0D1B3E, #0A2240);
    border-bottom: 1px solid var(--border);
    padding: 24px 20px 20px;
    margin: -1rem -1rem 20px;
}
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 900;
    color: #fff;
    margin-bottom: 4px;
}
.sidebar-logo span { color: var(--gold); }
.sidebar-sub {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    line-height: 1.5;
}

.processing-bar {
    height: 3px;
    background: linear-gradient(90deg, var(--cyan), var(--teal), var(--gold), var(--pink), var(--cyan));
    background-size: 200% 100%;
    border-radius: 2px;
    margin-bottom: 16px;
    animation: shimmer 1.5s linear infinite;
}

.empty-state {
    text-align: center;
    padding: 80px 0;
    animation: pulse 3s ease-in-out infinite;
}
.empty-icon { font-size: 3rem; margin-bottom: 14px; }
.empty-text {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    color: var(--border);
    letter-spacing: 0.08em;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.8); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes pulse {
    0%, 100% { opacity: 0.4; }
    50%       { opacity: 0.8; }
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">Trust<span>Vision</span></div>
        <div class="sidebar-sub">
            Marquette AI Hub<br>
            ECE Dept · Haggerty Hall 208
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">🤖 Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model", ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"], label_visibility="collapsed")

    st.markdown('<div class="sec-label">🎯 Confidence Threshold</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence", 0.1, 0.95, 0.40, 0.05, label_visibility="collapsed")

    st.markdown('<div class="sec-label">📂 Input Source</div>', unsafe_allow_html=True)
    source = st.radio("Source", ["Upload Image", "Upload Video"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#3A4A6B; line-height:1.8;'>
        🔬 YOLO11 · ByteTrack · Grad-CAM<br>
        📍 Haggerty Hall Room 208<br>
        🗓️ Summer Research 2025
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-location">
        Marquette University<br>
        Haggerty Hall · Room 208<br>
        Summer 2025
    </div>
    <div class="hero-dept">⚡ Electrical &amp; Computer Engineering Department</div>
    <div class="hero-title"><span class="t">Trust</span><span class="v">Vision</span></div>
    <div class="hero-tagline">"Where Ideas Meet Intelligence"</div>
    <div class="hero-desc">
        The Marquette AI Hub is a space for innovation, exploration, and collaboration in Artificial Intelligence:
        empowering minds to build smart solutions for a better tomorrow.<br><br>
        <strong style="color:#C8D8F0;">TrustVision</strong> is a real-time object detection workstation that goes beyond bounding boxes.
        It detects and tracks objects in live video, then explains <em>why</em> the model made each decision,
        how confident it is, and where it is likely to fail: making AI transparent and trustworthy
        for students, faculty, and visitors at the AI Hub.
    </div>
    <div class="hero-tags">
        <span class="hero-tag tag-gold">YOLO11 Detection</span>
        <span class="hero-tag tag-cyan">ByteTrack Tracking</span>
        <span class="hero-tag tag-teal">Grad-CAM Explainability</span>
        <span class="hero-tag tag-pink">Uncertainty Scoring</span>
        <span class="hero-tag tag-gold">FiftyOne Dataset QA</span>
        <span class="hero-tag tag-cyan">Real-Time Dashboard</span>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model(model_choice)

# ── Image mode ────────────────────────────────────────────────────────────────
if source == "Upload Image":
    file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if file:
        image = Image.open(file)
        st.markdown('<div class="processing-bar"></div>', unsafe_allow_html=True)

        with st.spinner("Running YOLO detection..."):
            results = model(image, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()
            detections = results[0].boxes

        num_det   = len(detections) if detections is not None else 0
        avg_conf  = float(detections.conf.mean()) if detections is not None and len(detections) > 0 else 0.0
        class_ids = detections.cls.tolist() if detections is not None and len(detections) > 0 else []
        classes   = list(set([model.names[int(c)] for c in class_ids]))

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card gold">
                <div class="stat-label">Objects Found</div>
                <div class="stat-value">{num_det}</div>
            </div>
            <div class="stat-card cyan">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{avg_conf:.0%}</div>
            </div>
            <div class="stat-card teal">
                <div class="stat-label">Unique Classes</div>
                <div class="stat-value">{len(classes)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if classes:
            badges = " ".join([f'<span class="badge">{c}</span>' for c in sorted(classes)])
            st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec-label">📷 Original</div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        with col2:
            st.markdown('<div class="sec-label">🎯 Detections</div>', unsafe_allow_html=True)
            st.image(annotated, channels="BGR", use_column_width=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🖼️</div>
            <div class="empty-text">Upload an image to begin detection</div>
        </div>
        """, unsafe_allow_html=True)

# ── Video mode ────────────────────────────────────────────────────────────────
elif source == "Upload Video":
    file = st.file_uploader("Drop a video here", type=["mp4", "mov", "avi"], label_visibility="collapsed")

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(file.read())
        tfile.close()

        st.markdown('<div class="processing-bar"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<div class="sec-label">🎬 Live Detections</div>', unsafe_allow_html=True)
            stframe = st.empty()
        with col2:
            st.markdown('<div class="sec-label">📊 Live Stats</div>', unsafe_allow_html=True)
            stat_box = st.empty()
            st.markdown('<div class="sec-label">🏷️ Classes</div>', unsafe_allow_html=True)
            class_box = st.empty()

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
            annotated = results[0].plot()
            detections = results[0].boxes

            stframe.image(annotated, channels="BGR", use_column_width=True)

            if frame_count % 10 == 0:
                num_det = len(detections) if detections is not None else 0
                avg_c   = float(detections.conf.mean()) if detections is not None and len(detections) > 0 else 0.0
                classes = []
                if detections is not None and len(detections) > 0:
                    classes = list(set([model.names[int(c)] for c in detections.cls.tolist()]))

                stat_box.markdown(f"""
                <div class="stat-card gold" style="margin-bottom:12px;">
                    <div class="stat-label">Objects</div>
                    <div class="stat-value">{num_det}</div>
                </div>
                <div class="stat-card cyan" style="margin-bottom:12px;">
                    <div class="stat-label">Avg Conf</div>
                    <div class="stat-value">{avg_c:.0%}</div>
                </div>
                <div class="stat-card teal">
                    <div class="stat-label">Frame</div>
                    <div class="stat-value" style="font-size:1.2rem;">{frame_count}</div>
                </div>
                """, unsafe_allow_html=True)

                if classes:
                    badges = "".join([
                        f'<div class="badge" style="display:block;margin:5px 0;">{c}</div>'
                        for c in sorted(classes)
                    ])
                    class_box.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)

            frame_count += 1

        cap.release()
        os.unlink(tfile.name)
        st.success(f"✅ Done — processed {frame_count} frames.")

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <div class="empty-text">Upload a video to begin detection</div>
        </div>
        """, unsafe_allow_html=True)
