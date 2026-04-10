"""
Smart City Complaint Intelligence System — Flask Backend
Supports: image upload, voice notes, navigation, Leaflet map
"""
import uuid, sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

from flask import (Flask, request, jsonify, render_template,
                   session, redirect, url_for, send_from_directory)
from functools import wraps
from werkzeug.utils import secure_filename
from utils.db import get_db, init_db, dept_for_category
from utils.ml import predict
from utils.red_alert import detect_red_alert_zones

app = Flask(__name__)
app.secret_key = 'civicIntel-secret-2025-xk9'

# ── FILE UPLOAD CONFIG ─────────────────────────────────────────────────────
UPLOAD_FOLDER   = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXT     = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_UPLOAD_MB   = 8
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# ── INIT DB ────────────────────────────────────────────────────────────────
with app.app_context():
    init_db()

# ── AUTH DECORATOR ──────────────────────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map-view')
def map_view():
    return render_template('map.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        conn = get_db()
        user = conn.execute(
            'SELECT * FROM admin_users WHERE username=? AND password=?',
            (username, password)
        ).fetchone()
        conn.close()
        if user:
            session['admin_logged_in'] = True
            session['admin_user'] = username
            return redirect(url_for('admin_dashboard'))
        error = 'Invalid credentials. Try admin / admin123'
    return render_template('login.html', error=error)

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    return render_template('admin.html')

# ── Serve uploaded images ────────────────────────────────────────────────────
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ══════════════════════════════════════════════════════════════════════════════
# API — PREDICT (no save)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict_complaint():
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()
    if len(text) < 10:
        return jsonify({'error': 'Complaint text too short'}), 400
    try:
        result = predict(text)
        result['department'] = dept_for_category(result['category'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ══════════════════════════════════════════════════════════════════════════════
# API — SUBMIT (multipart form with optional image)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/submit', methods=['POST'])
def submit_complaint():
    """
    Accepts multipart/form-data (for image upload) OR JSON.
    Fields: text, latitude, longitude, area_name, submitter,
            voice_note, image (file)
    """
    # Support both multipart and JSON
    if request.content_type and 'multipart' in request.content_type:
        text       = (request.form.get('text') or '').strip()
        lat_raw    = request.form.get('latitude')
        lng_raw    = request.form.get('longitude')
        area_name  = (request.form.get('area_name') or '').strip()
        submitter  = (request.form.get('submitter') or 'Citizen').strip()
        voice_note = (request.form.get('voice_note') or '').strip()
    else:
        data       = request.get_json(force=True)
        text       = (data.get('text') or '').strip()
        lat_raw    = data.get('latitude')
        lng_raw    = data.get('longitude')
        area_name  = (data.get('area_name') or '').strip()
        submitter  = (data.get('submitter') or 'Citizen').strip()
        voice_note = (data.get('voice_note') or '').strip()

    # Validate
    if len(text) < 10:
        return jsonify({'error': 'Complaint text too short (min 10 chars)'}), 400
    if lat_raw is None or lng_raw is None:
        return jsonify({'error': 'Location coordinates are required'}), 400
    try:
        lat, lng = float(lat_raw), float(lng_raw)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid coordinates'}), 400

    # Handle image upload
    image_path = ''
    if 'image' in request.files:
        img = request.files['image']
        if img and img.filename and allowed_file(img.filename):
            ext       = img.filename.rsplit('.', 1)[1].lower()
            filename  = f"{uuid.uuid4().hex}.{ext}"
            img.save(os.path.join(UPLOAD_FOLDER, filename))
            image_path = f"/uploads/{filename}"

    # AI prediction
    try:
        ml = predict(text)
    except Exception as e:
        return jsonify({'error': f'AI prediction failed: {e}'}), 500

    complaint_id = 'SC-' + uuid.uuid4().hex[:8].upper()
    dept = dept_for_category(ml['category'])

    conn = get_db()
    conn.execute("""
        INSERT INTO complaints
          (complaint_id, text, area_name, latitude, longitude, category,
           priority, is_fake, department, status, submitter, image, voice_note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (complaint_id, text, area_name, lat, lng,
          ml['category'], ml['priority'], ml['is_fake'],
          dept, 'Open', submitter, image_path, voice_note))
    conn.commit()
    conn.close()

    return jsonify({
        'complaint_id':          complaint_id,
        'category':              ml['category'],
        'priority':              ml['priority'],
        'is_fake':               ml['is_fake'],
        'department':            dept,
        'message':               'Complaint submitted successfully',
        'category_confidence':   ml['category_confidence'],
        'priority_confidence':   ml['priority_confidence'],
        'image':                 image_path,
    })

# ══════════════════════════════════════════════════════════════════════════════
# API — MAP DATA
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/map-data')
def map_data():
    conn = get_db()
    rows = conn.execute("""
        SELECT complaint_id, text, area_name, latitude, longitude,
               category, priority, is_fake, department, status,
               image, voice_note, created_at
        FROM complaints ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    complaints = [dict(r) for r in rows]
    zones = detect_red_alert_zones(complaints)
    return jsonify({'complaints': complaints, 'red_alert_zones': zones})

# ══════════════════════════════════════════════════════════════════════════════
# API — RED ALERT
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/red-alert')
def red_alert():
    conn = get_db()
    rows = conn.execute("SELECT * FROM complaints").fetchall()
    conn.close()
    complaints = [dict(r) for r in rows]
    zones = detect_red_alert_zones(complaints)
    return jsonify({'red_alert_zones': zones, 'count': len(zones)})

# ══════════════════════════════════════════════════════════════════════════════
# ADMIN APIs
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/admin/complaints')
@admin_required
def admin_complaints():
    conn     = get_db()
    category = request.args.get('category', '')
    priority = request.args.get('priority', '')
    area     = request.args.get('area', '')
    status   = request.args.get('status', '')
    is_fake  = request.args.get('is_fake', '')

    q = "SELECT * FROM complaints WHERE 1=1"
    params = []
    if category: q += " AND category=?";        params.append(category)
    if priority: q += " AND priority=?";        params.append(priority)
    if area:     q += " AND area_name LIKE ?";  params.append(f'%{area}%')
    if status:   q += " AND status=?";          params.append(status)
    if is_fake != '': q += " AND is_fake=?";   params.append(int(is_fake))
    q += " ORDER BY created_at DESC"

    rows = conn.execute(q, params).fetchall()
    conn.close()
    return jsonify({'complaints': [dict(r) for r in rows]})


@app.route('/admin/stats')
@admin_required
def admin_stats():
    conn     = get_db()
    total    = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    genuine  = conn.execute("SELECT COUNT(*) FROM complaints WHERE is_fake=0").fetchone()[0]
    fake     = conn.execute("SELECT COUNT(*) FROM complaints WHERE is_fake=1").fetchone()[0]
    critical = conn.execute("SELECT COUNT(*) FROM complaints WHERE priority='Critical' AND is_fake=0").fetchone()[0]
    open_c   = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Open'").fetchone()[0]
    by_cat   = conn.execute("SELECT category, COUNT(*) as cnt FROM complaints WHERE is_fake=0 GROUP BY category").fetchall()
    by_pri   = conn.execute("SELECT priority, COUNT(*) as cnt FROM complaints WHERE is_fake=0 GROUP BY priority").fetchall()
    recent   = conn.execute("SELECT * FROM complaints ORDER BY created_at DESC LIMIT 10").fetchall()
    zones    = detect_red_alert_zones([dict(r) for r in conn.execute("SELECT * FROM complaints").fetchall()])
    conn.close()
    return jsonify({
        'total': total, 'genuine': genuine, 'fake': fake,
        'critical': critical, 'open': open_c,
        'by_category': [dict(r) for r in by_cat],
        'by_priority': [dict(r) for r in by_pri],
        'recent':      [dict(r) for r in recent],
        'red_alert_zones': len(zones),
    })


@app.route('/admin/update-status', methods=['POST'])
@admin_required
def update_status():
    data = request.get_json(force=True)
    conn = get_db()
    conn.execute("UPDATE complaints SET status=? WHERE complaint_id=?",
                 (data['status'], data['complaint_id']))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/admin/seed-demo', methods=['POST'])
@admin_required
def seed_demo():
    import random, datetime
    demo = [
        (22.7196, 75.8577, "Huge pothole on Vijay Nagar road causing accidents daily", "Vijay Nagar"),
    (22.7226, 75.8627, "Road completely broken near Palasia bus stop emergency", "Palasia"),
    (22.7176, 75.8527, "Massive crater on road near Treasure Island mall dangerous", "Vijay Nagar"),
    (22.7156, 75.8607, "Road collapsed near Vijay Nagar square very dangerous", "Vijay Nagar"),
    (22.7146, 75.8557, "Bridge crack on main road emergency situation", "Vijay Nagar"),


    # 🟡 MODERATE
    (22.7196, 75.8877, "Water supply cut for 5 days in Scheme 54 families facing difficulty", "Scheme 54"),
    (22.7176, 75.8827, "Water pipe burst flooding entire sector 4 road", "Scheme 54"),
    (22.7396, 75.8677, "Power outage 12 hours Bhawarkua area no proper response", "Bhawarkua"),
    (22.7296, 75.8477, "Drug addicts gathering near Nehru Park creating unsafe environment", "Nehru Park"),
    (22.7316, 75.8457, "Chain snatching incidents reported frequently near Nehru Park market", "Nehru Park"),
    (22.7116, 75.8797, "Drainage overflow flooding streets near Rajwada old area", "Rajwada"),

    # 🟢 NORMAL
    (22.7096, 75.8777, "Garbage not collected regularly in Rajwada area causing inconvenience", "Rajwada"),
    (22.7196, 75.8577, "Streetlight not working properly in Vijay Nagar area", "Vijay Nagar"),
    (22.7226, 75.8627, "Minor road cracks near Palasia bus stop", "Palasia"),
    (22.7396, 75.8677, "Occasional voltage fluctuation in Bhawarkua area", "Bhawarkua"),
    (22.7176, 75.8827, "Low water pressure in some houses of Scheme 54", "Scheme 54"),
    ]
    conn  = get_db()
    added = 0
    for lat, lng, text, area in demo:
        try:
            ml   = predict(text)
            cid  = 'SC-' + uuid.uuid4().hex[:8].upper()
            dept = dept_for_category(ml['category'])
            days = random.randint(0, 2)
            ts   = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            conn.execute("""
                INSERT INTO complaints
                  (complaint_id,text,area_name,latitude,longitude,category,
                   priority,is_fake,department,status,submitter,image,voice_note,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (cid, text, area,
                  lat + random.uniform(-0.001, 0.001),
                  lng + random.uniform(-0.001, 0.001),
                  ml['category'], ml['priority'], ml['is_fake'],
                  dept, 'Open', 'DemoUser', '', '', ts))
            added += 1
        except:
            pass
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'added': added})



@app.route('/admin/clear-data', methods=['POST'])
@admin_required
def clear_data():
    """Delete ALL complaints — resets database for fresh start"""
    try:
        conn = get_db()
        # Count before delete
        count = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        conn.execute("DELETE FROM complaints")
        # Reset auto-increment counter
        conn.execute("DELETE FROM sqlite_sequence WHERE name='complaints'")
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'deleted': count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("  Civic Intel — Smart City Intelligence System")
    print("  Visit: http://127.0.0.1:5000")
    print("  Admin: http://127.0.0.1:5000/admin")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)