import os
from datetime import datetime
import config # Import file config.py

from flask import Flask, url_for, session, redirect, render_template, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from flask_babel import Babel, _

from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename

# --- 1. KHỞI TẠO VÀ CẤU HÌNH ---
app = Flask(__name__)

# Cấu hình từ file config.py
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAIL_USERNAME'] = config.EMAIL_USER
app.config['MAIL_PASSWORD'] = config.EMAIL_PASS

# Cấu hình chung
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Cấu hình Mail
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True

# --- CẤU HÌNH BABEL ---
app.config['LANGUAGES'] = {
    'en': 'English',
    'vi': 'Tiếng Việt'
}

def get_locale():
    # Lấy ngôn ngữ từ session, nếu không có thì dùng ngôn ngữ của trình duyệt
    return session.get('language', request.accept_languages.best_match(app.config['LANGUAGES'].keys()))

babel = Babel(app, locale_selector=get_locale)
# --------------------

# Khởi tạo các extension
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
oauth = OAuth(app)

# --- Cấu hình Google Login ---
google = oauth.register(
    name='google',
    client_id='564904327189-4gsii5kfkht070218tsjqu8amnstc7o1.apps.googleusercontent.com',
    client_secret='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',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- 2. ĐỊNH NGHĨA MODEL ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)
    picture = db.Column(db.String(200), nullable=True)
    recordings = db.relationship('Recording', backref='user', lazy=True)

class Recording(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 3. ĐỊNH NGHĨA ROUTE ---

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/language/<language>')
def set_language(language=None):
    session['language'] = language
    return redirect(request.referrer)

# ... (Các route còn lại của bạn giữ nguyên không đổi) ...
# ... (login, register, logout, authorize, diagnose, contact, history, ...)
# ... (delete_recording, profile, edit_profile, upload_audio) ...

# --- 4. CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Tạo database nếu chưa có
    app.run(debug=True)