# 🏙️ CityPulse — AI-Based Smart City Complaint Intelligence System

A full-stack web application that uses AI/ML to classify city complaints, predict priority, detect fake submissions, identify red alert zones, and visualize emergencies on an interactive heatmap.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train AI Models (REQUIRED first time)
```bash
python train_model.py
```
This trains and saves 3 models to `/models/`:
- `category_model.pkl` — TF-IDF + Naive Bayes (5 categories)
- `priority_model.pkl` — TF-IDF + Logistic Regression (Critical/Moderate/Normal)
- `fake_model.pkl` — TF-IDF + Logistic Regression (Genuine/Fake)

### 3. Start the Server
```bash
python app.py
```

### 4. Open in Browser
| URL | Description |
|-----|-------------|
| http://localhost:5000 | Citizen complaint portal |
| http://localhost:5000/map-view | Live heatmap & visualization |
| http://localhost:5000/admin | Admin dashboard (login required) |

**Admin Credentials:** `admin` / `admin123`

---

## 📁 Project Structure
```
smartcity/
├── app.py                    # Flask backend + all API routes
├── train_model.py            # ML model training script
├── requirements.txt
├── README.md
├── models/                   # Saved ML models (auto-created)
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── fake_model.pkl
├── database/
│   └── complaints.db         # SQLite database (auto-created)
├── templates/
│   ├── index.html            # Citizen complaint page
│   ├── map.html              # Live map visualization
│   ├── admin.html            # Admin dashboard
│   └── login.html            # Admin login
├── utils/
│   ├── db.py                 # Database helpers
│   ├── ml.py                 # Model loader & prediction
│   └── red_alert.py          # Red alert zone detection
└── static/                   # CSS/JS/images
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Citizen complaint page |
| GET | `/map-view` | Live map page |
| POST | `/predict` | Live AI prediction (no save) |
| POST | `/submit` | Submit + save complaint |
| GET | `/map-data` | All complaint data for map |
| GET | `/red-alert` | Current red alert zones |
| GET | `/admin` | Admin dashboard |
| GET | `/admin/complaints` | Filtered complaints list |
| GET | `/admin/stats` | Dashboard statistics |
| POST | `/admin/update-status` | Update complaint status |
| POST | `/admin/seed-demo` | Seed demo data |

---

## 🧠 AI/ML Details

### Category Classification
- **Algorithm:** TF-IDF (bigrams, 8000 features) + Multinomial Naive Bayes
- **Categories:** Road, Water, Electricity, Sanitation, PublicSafety
- **Training:** 82 labeled samples with preprocessing

### Priority Prediction
- **Algorithm:** TF-IDF + Logistic Regression (multinomial)
- **Classes:** Critical, Moderate, Normal
- **Keywords:** Detects urgency signals in complaint text

### Fake Detection
- **Algorithm:** TF-IDF + Logistic Regression (class_weight=balanced)
- **Classes:** 0=Genuine, 1=Fake
- **Handles:** Gibberish, personal attacks, conspiracy content

### Red Alert Zone Detection
- **Logic:** Pandas spatial clustering
- **Trigger:** 3+ Critical complaints within ~2km radius AND 72h window
- **Output:** Zone centroid, complaint count, affected categories

---

## 🗺️ Map Features (Leaflet.js)
- **Dark tile layer** from CartoDB
- **Heatmap** using `leaflet.heat` — density of all complaints
- **Priority markers** — Critical (red glow), Moderate (amber), Normal (teal)
- **Red alert circles** — 1.5km dashed border around alert zones
- **Interactive popups** — complaint text, category, priority, department, time
- **Auto-fit bounds** on load

---

## 🔴 Red Alert Logic
```python
THRESHOLD     = 3       # min critical complaints
RADIUS_DEG    = 0.02    # ~2km
TIME_WINDOW_H = 72      # hours
```
When conditions met: zone appears on map, sidebar alert, banner notification.

---

## 🏢 Department Auto-Routing
| Category | Department |
|----------|-----------|
| Road | PWD (Public Works Dept) |
| Water | Water Supply Board |
| Electricity | Electric Department |
| Sanitation | Municipal Corporation |
| PublicSafety | Police Department |
