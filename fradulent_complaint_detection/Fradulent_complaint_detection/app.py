import os
import pickle
import re
import sqlite3
import tempfile
import warnings
from datetime import datetime
from functools import wraps
from uuid import uuid4
from urllib.parse import quote

import pandas as pd
from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = "mini_project_secret"

BASE_DIR = app.root_path
DATABASE = os.environ.get(
    "FRAUD_APP_DB",
    os.path.join(tempfile.gettempdir(), "fraudulent_complaints.db"),
)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg"}
MAX_IMAGE_SIZE = 2 * 1024 * 1024
ADMIN_REVIEW_STATUSES = {"Pending", "Verified", "Rejected"}
FRAUD_ALERT_THRESHOLD = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE + (512 * 1024)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS complaints(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            complaint TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT,
            image_path TEXT,
            review_status TEXT NOT NULL DEFAULT 'Pending',
            FOREIGN KEY(username) REFERENCES users(username)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unread',
            created_at TEXT NOT NULL,
            FOREIGN KEY(complaint_id) REFERENCES complaints(id)
        )
        """
    )

    columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "created_at" not in columns:
        cursor.execute("ALTER TABLE complaints ADD COLUMN created_at TEXT")
    if "image_path" not in columns:
        cursor.execute("ALTER TABLE complaints ADD COLUMN image_path TEXT")
    if "review_status" not in columns:
        cursor.execute("ALTER TABLE complaints ADD COLUMN review_status TEXT DEFAULT 'Pending'")

    notification_columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(notifications)").fetchall()
    }

    user_columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(users)").fetchall()
    }
    if "name" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN name TEXT")
    if "email" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)"
    )
    if "status" not in notification_columns and notification_columns:
        cursor.execute("ALTER TABLE notifications ADD COLUMN status TEXT DEFAULT 'unread'")

    cursor.execute(
        "UPDATE complaints SET result = 'Fraudulent' WHERE LOWER(result) = 'fake'"
    )
    cursor.execute(
        """
        UPDATE complaints
        SET review_status = 'Pending'
        WHERE review_status IS NULL
            OR TRIM(review_status) = ''
            OR review_status NOT IN ('Pending', 'Verified', 'Rejected')
        """
    )
    if notification_columns:
        cursor.execute(
            "UPDATE notifications SET status = 'unread' WHERE status IS NULL OR TRIM(status) = ''"
        )

    admin_password = generate_password_hash("admin123")
    cursor.execute(
        """
        INSERT INTO users(username, name, email, password, role)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(username) DO NOTHING
        """,
        ("admin", "Administrator", "admin@frauddetection.local", admin_password, "admin"),
    )

    conn.commit()
    conn.close()


def train_and_store_model():
    data = pd.read_csv(os.path.join(BASE_DIR, "complaints.csv"))
    data["label"] = data["label"].str.lower().str.strip()

    trained_vectorizer = TfidfVectorizer(stop_words="english")
    features = trained_vectorizer.fit_transform(data["complaint"])

    trained_model = LogisticRegression(max_iter=1000)
    trained_model.fit(features, data["label"])

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(trained_model, model_file)
    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(trained_vectorizer, vectorizer_file)

    return trained_model, trained_vectorizer


def load_model_assets():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            with open(MODEL_PATH, "rb") as model_file:
                loaded_model = pickle.load(model_file)
            with open(VECTORIZER_PATH, "rb") as vectorizer_file:
                loaded_vectorizer = pickle.load(vectorizer_file)
        return loaded_model, loaded_vectorizer
    except (FileNotFoundError, pickle.PickleError, EOFError, AttributeError, InconsistentVersionWarning):
        return train_and_store_model()


model, vectorizer = load_model_assets()
init_db()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("home"))
        return view(*args, **kwargs)

    return wrapped_view


def admin_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if session.get("role") != "admin":
            return redirect(url_for("home"))
        return view(*args, **kwargs)

    return wrapped_view


def normalize_result(label):
    return "Fraudulent" if str(label).strip().lower() == "fake" else "Genuine"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_image(upload):
    if not upload or not upload.filename:
        return None

    if not allowed_file(upload.filename):
        raise ValueError("Only JPG, JPEG, and PNG image files are allowed.")
    if upload.mimetype not in ALLOWED_MIME_TYPES:
        raise ValueError("Invalid image format. Please upload a JPG, JPEG, or PNG file.")

    upload.stream.seek(0, os.SEEK_END)
    file_size = upload.stream.tell()
    upload.stream.seek(0)

    if file_size > MAX_IMAGE_SIZE:
        raise ValueError("Image size must be 2MB or less.")

    filename = secure_filename(upload.filename)
    unique_name = f"{uuid4().hex}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    upload.save(save_path)
    return f"uploads/{unique_name}"


def predict_complaint(complaint_text):
    vectorized_text = vectorizer.transform([complaint_text])
    prediction = model.predict(vectorized_text)[0]
    confidence = round(float(model.predict_proba(vectorized_text).max()), 4)
    return normalize_result(prediction), confidence


def build_notification_message(complaint_text):
    summary = " ".join(complaint_text.split()[:8]).strip()
    if len(complaint_text.split()) > 8:
        summary = f"{summary}..."
    return f"New complaint submitted: {summary}"


def format_timestamp(value):
    if not value:
        return "Earlier record"
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return value


def normalize_filter(value):
    return value if value in {"all", "fraud", "genuine"} else "all"


def serialize_complaint(complaint):
    image_url = None
    if complaint["image_path"]:
        encoded_path = quote(complaint["image_path"])
        image_url = url_for("static", filename=encoded_path)

    return {
        "id": complaint["id"],
        "username": complaint["username"],
        "complaint": complaint["complaint"],
        "short_complaint": complaint["complaint"][:72] + ("..." if len(complaint["complaint"]) > 72 else ""),
        "result": complaint["result"],
        "confidence": round(float(complaint["confidence"]) * 100, 2),
        "created_at": complaint["created_at"] or "",
        "created_at_display": format_timestamp(complaint["created_at"]),
        "image_path": complaint["image_path"],
        "image_url": image_url,
        "review_status": complaint["review_status"] or "Pending",
    }


def fetch_admin_summary():
    db = get_db()
    summary = db.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN result = 'Fraudulent' THEN 1 ELSE 0 END) AS fraudulent,
            SUM(CASE WHEN result = 'Genuine' THEN 1 ELSE 0 END) AS genuine,
            SUM(CASE WHEN date(created_at) = date('now', 'localtime') THEN 1 ELSE 0 END) AS today_count
        FROM complaints
        """
    ).fetchone()

    total = summary["total"] or 0
    fraudulent = summary["fraudulent"] or 0
    genuine = summary["genuine"] or 0
    today_count = summary["today_count"] or 0

    fraudulent_pct = round((fraudulent / total) * 100, 1) if total else 0
    genuine_pct = round((genuine / total) * 100, 1) if total else 0

    return {
        "total": total,
        "fraudulent": fraudulent,
        "genuine": genuine,
        "today_count": today_count,
        "fraudulent_pct": fraudulent_pct,
        "genuine_pct": genuine_pct,
        "alert": fraudulent >= FRAUD_ALERT_THRESHOLD,
        "alert_message": "High Fraud Activity Detected",
    }


def fetch_admin_complaints(result_filter="all", search_query="", limit=None):
    db = get_db()
    clauses = []
    params = []

    if result_filter == "fraud":
        clauses.append("result = ?")
        params.append("Fraudulent")
    elif result_filter == "genuine":
        clauses.append("result = ?")
        params.append("Genuine")

    search_query = (search_query or "").strip()
    if search_query:
        clauses.append("(complaint LIKE ? OR username LIKE ? OR CAST(id AS TEXT) LIKE ?)")
        term = f"%{search_query}%"
        params.extend([term, term, term])

    query = """
        SELECT id, username, complaint, result, confidence, created_at, image_path, review_status
        FROM complaints
    """
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    return db.execute(query, params).fetchall()


def get_admin_dashboard_payload(result_filter="all", search_query=""):
    complaints = fetch_admin_complaints(result_filter=result_filter, search_query=search_query)
    recent_complaints = fetch_admin_complaints(limit=6)
    return {
        "summary": fetch_admin_summary(),
        "complaints": [serialize_complaint(row) for row in complaints],
        "recent_complaints": [serialize_complaint(row) for row in recent_complaints],
        "active_filter": result_filter,
        "search_query": search_query,
        "review_statuses": sorted(ADMIN_REVIEW_STATUSES),
    }


def create_admin_notification(complaint_id, complaint_text, created_at):
    db = get_db()
    db.execute(
        """
        INSERT INTO notifications(complaint_id, message, status, created_at)
        VALUES (?, ?, 'unread', ?)
        """,
        (complaint_id, build_notification_message(complaint_text), created_at),
    )
    db.commit()


def fetch_notifications(limit=6):
    notifications = get_db().execute(
        """
        SELECT id, complaint_id, message, status, created_at
        FROM notifications
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    unread_count = get_db().execute(
        "SELECT COUNT(*) AS count FROM notifications WHERE status = 'unread'"
    ).fetchone()["count"]
    return notifications, unread_count


def serialize_notifications(notifications):
    return [
        {
            "id": notification["id"],
            "complaint_id": notification["complaint_id"],
            "message": notification["message"],
            "status": notification["status"],
            "created_at": notification["created_at"],
        }
        for notification in notifications
    ]


def get_dashboard_metrics():
    db = get_db()
    total = db.execute("SELECT COUNT(*) AS count FROM complaints").fetchone()["count"]
    fraudulent = db.execute(
        "SELECT COUNT(*) AS count FROM complaints WHERE result = ?",
        ("Fraudulent",),
    ).fetchone()["count"]
    genuine = db.execute(
        "SELECT COUNT(*) AS count FROM complaints WHERE result = ?",
        ("Genuine",),
    ).fetchone()["count"]
    recent = db.execute(
        """
        SELECT id, username, complaint, result, confidence, created_at, image_path, review_status
        FROM complaints
        ORDER BY id DESC
        LIMIT 8
        """
    ).fetchall()
    return {
        "total": total,
        "fraudulent": fraudulent,
        "genuine": genuine,
        "recent_complaints": recent,
    }


def verify_password(stored_password, provided_password):
    if stored_password.startswith("pbkdf2:") or stored_password.startswith("scrypt:"):
        return check_password_hash(stored_password, provided_password)
    return stored_password == provided_password


def render_auth_page(active_tab="user-login"):
    return render_template("login.html", active_tab=active_tab)


@app.context_processor
def inject_admin_notifications():
    if session.get("role") != "admin":
        return {"admin_notifications": [], "unread_notifications": 0}

    notifications, unread_count = fetch_notifications()
    return {
        "admin_notifications": notifications,
        "unread_notifications": unread_count,
    }


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    flash("Uploaded file is too large. Please upload an image up to 2MB.", "danger")
    if session.get("user"):
        return redirect(url_for("submit_complaint"))
    return redirect(url_for("home"))


@app.route("/")
def home():
    if session.get("user"):
        return redirect(url_for("admin" if session.get("role") == "admin" else "dashboard"))
    return render_auth_page(request.args.get("tab", "user-login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_auth_page("user-signup")

    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if len(name) < 2:
            flash("Name must be at least 2 characters long.", "danger")
            return render_auth_page("user-signup")
        if not EMAIL_PATTERN.match(email):
            flash("Please enter a valid email address.", "danger")
            return render_auth_page("user-signup")
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_auth_page("user-signup")
        if password != confirm_password:
            flash("Password and confirm password do not match.", "danger")
            return render_auth_page("user-signup")

        try:
            db = get_db()
            db.execute(
                "INSERT INTO users(username, name, email, password, role) VALUES (?, ?, ?, ?, ?)",
                (email, name, email, generate_password_hash(password), "user"),
            )
            db.commit()
            flash("Account created successfully. Please log in.", "success")
            return render_auth_page("user-login")
        except sqlite3.IntegrityError:
            flash("That email is already registered. Please log in instead.", "danger")

    return render_auth_page("user-signup")


@app.route("/login", methods=["POST"])
def login():
    auth_mode = request.form.get("auth_mode", "user")
    username = request.form["username"].strip()
    password = request.form["password"]

    db = get_db()
    if auth_mode == "admin":
        user = db.execute(
            "SELECT username, name, email, password, role FROM users WHERE username = ? AND role = 'admin'",
            (username,),
        ).fetchone()
        active_tab = "admin-login"
    else:
        user = db.execute(
            """
            SELECT username, name, email, password, role
            FROM users
            WHERE role = 'user' AND (username = ? OR email = ?)
            """,
            (username, username.lower()),
        ).fetchone()
        active_tab = "user-login"

    if not user or not verify_password(user["password"], password):
        flash("Invalid username or password.", "danger")
        return render_auth_page(active_tab)

    session["user"] = user["username"]
    session["role"] = user["role"]
    display_name = user["name"] or user["username"]
    flash(f"Welcome back, {display_name}.", "success")
    if user["role"] == "admin":
        return redirect(url_for("admin"))
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    metrics = get_dashboard_metrics()
    return render_template("dashboard.html", metrics=metrics)


@app.route("/submit-complaint", methods=["GET", "POST"])
@login_required
def submit_complaint():
    prediction = None

    if request.method == "POST":
        complaint_text = request.form["complaint"].strip()
        if not complaint_text:
            flash("Please enter complaint text before submitting.", "danger")
            return render_template("submit_complaint.html", prediction=prediction)

        image_path = None
        image_file = request.files.get("image")
        try:
            image_path = save_uploaded_image(image_file)
        except ValueError as exc:
            flash(str(exc), "danger")
            return render_template("submit_complaint.html", prediction=prediction)

        result, confidence = predict_complaint(complaint_text)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        db = get_db()
        cursor = db.execute(
            """
            INSERT INTO complaints(username, complaint, result, confidence, created_at, image_path, review_status)
            VALUES (?, ?, ?, ?, ?, ?, 'Pending')
            """,
            (session["user"], complaint_text, result, confidence, created_at, image_path),
        )
        db.commit()
        create_admin_notification(cursor.lastrowid, complaint_text, created_at)

        prediction = {
            "complaint": complaint_text,
            "result": result,
            "confidence": round(confidence * 100, 2),
            "image_path": image_path,
        }
        flash("Complaint analyzed and stored successfully.", "success")

    return render_template("submit_complaint.html", prediction=prediction)


@app.route("/admin")
@login_required
@admin_required
def admin():
    result_filter = normalize_filter(request.args.get("filter", "all"))
    search_query = request.args.get("q", "").strip()
    dashboard_data = get_admin_dashboard_payload(result_filter=result_filter, search_query=search_query)
    return render_template("admin.html", **dashboard_data)


@app.route("/admin/complaints-data")
@login_required
@admin_required
def admin_complaints_data():
    result_filter = normalize_filter(request.args.get("filter", "all"))
    search_query = request.args.get("q", "").strip()
    return jsonify(get_admin_dashboard_payload(result_filter=result_filter, search_query=search_query))


@app.route("/admin/complaints/<int:complaint_id>")
@login_required
@admin_required
def admin_complaint_detail(complaint_id):
    complaint = get_db().execute(
        """
        SELECT id, username, complaint, result, confidence, created_at, image_path, review_status
        FROM complaints
        WHERE id = ?
        """,
        (complaint_id,),
    ).fetchone()
    if not complaint:
        return jsonify({"ok": False, "message": "Complaint not found."}), 404
    return jsonify({"ok": True, "complaint": serialize_complaint(complaint)})


@app.route("/admin/complaints/<int:complaint_id>/review-status", methods=["POST"])
@login_required
@admin_required
def update_review_status(complaint_id):
    payload = request.get_json(silent=True) or request.form
    review_status = (payload.get("review_status") or "").strip().title()
    if review_status not in ADMIN_REVIEW_STATUSES:
        return jsonify({"ok": False, "message": "Invalid review status."}), 400

    db = get_db()
    cursor = db.execute(
        "UPDATE complaints SET review_status = ? WHERE id = ?",
        (review_status, complaint_id),
    )
    db.commit()
    if cursor.rowcount == 0:
        return jsonify({"ok": False, "message": "Complaint not found."}), 404

    complaint = db.execute(
        """
        SELECT id, username, complaint, result, confidence, created_at, image_path, review_status
        FROM complaints
        WHERE id = ?
        """,
        (complaint_id,),
    ).fetchone()
    return jsonify({"ok": True, "complaint": serialize_complaint(complaint)})


@app.route("/admin/notifications")
@login_required
@admin_required
def admin_notifications():
    notifications, unread_count = fetch_notifications(limit=8)
    return jsonify(
        {
            "notifications": serialize_notifications(notifications),
            "unread_count": unread_count,
        }
    )


@app.route("/admin/notifications/<int:notification_id>/read", methods=["POST"])
@login_required
@admin_required
def mark_notification_read(notification_id):
    db = get_db()
    db.execute(
        "UPDATE notifications SET status = 'read' WHERE id = ?",
        (notification_id,),
    )
    db.commit()
    notifications, unread_count = fetch_notifications(limit=8)
    return jsonify(
        {
            "ok": True,
            "notifications": serialize_notifications(notifications),
            "unread_count": unread_count,
        }
    )


@app.route("/delete/<int:complaint_id>", methods=["POST"])
@login_required
@admin_required
def delete_complaint(complaint_id):
    db = get_db()
    complaint = db.execute(
        "SELECT image_path FROM complaints WHERE id = ?",
        (complaint_id,),
    ).fetchone()
    db.execute("DELETE FROM notifications WHERE complaint_id = ?", (complaint_id,))
    db.execute("DELETE FROM complaints WHERE id = ?", (complaint_id,))
    db.commit()

    if complaint and complaint["image_path"]:
        file_path = os.path.join(BASE_DIR, "static", complaint["image_path"])
        if os.path.exists(file_path):
            os.remove(file_path)

    flash("Complaint deleted successfully.", "success")
    return redirect(url_for("admin"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
