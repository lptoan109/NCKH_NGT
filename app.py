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

# TẠO DATABASE (CHẠY MỘT LẦN)
# Lần đầu tiên chạy trên server, bạn cần mở Bash console,
# kích hoạt venv và chạy các lệnh sau trong python shell:
# from app import app, db
# with app.app_context():
#     db.create_all()

# --- 3. ĐỊNH NGHĨA ROUTE ---

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/language/<language>')
def set_language(language=None):
    session['language'] = language
    return redirect(request.referrer)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.password_hash and bcrypt.check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user)
            return redirect(url_for('diagnose'))
        else:
            flash(_('Invalid username or password.'), 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')

        existing_user_email = User.query.filter_by(email=email).first()
        if existing_user_email:
            flash(_('This email address is already in use.'), 'danger')
            return redirect(url_for('register'))

        existing_user_username = User.query.filter_by(username=username).first()
        if existing_user_username:
            flash(_('This username already exists.'), 'danger')
            return redirect(url_for('register'))

        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash(_('Account created successfully! Please log in.'), 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('user_info', None)
    session.pop('language', None)
    return redirect(url_for('homepage'))

@app.route('/login/google')
def login_google():
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    token = google.authorize_access_token()
    user_info = google.userinfo()
    user = User.query.filter_by(email=user_info['email']).first()
    if not user:
        user = User(
            email=user_info['email'],
            username=user_info['name'],
            picture=user_info['picture']
        )
        db.session.add(user)
        db.session.commit()
    login_user(user)
    return redirect(url_for('diagnose'))

@app.route('/diagnose')
@login_required
def diagnose():
    return render_template('diagnose.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            sender_email_from_form = request.form.get('email')
            subject = request.form.get('subject')
            message_body = request.form.get('message')

            msg = Message(
                subject=f"Tin nhắn từ Web NGT Cough: {subject}",
                sender=("Website NGT Cough", app.config['MAIL_USERNAME']),
                recipients=[app.config['MAIL_USERNAME']]
            )
            msg.body = f"Message from: {name} <{sender_email_from_form}>\n\n{message_body}"
            mail.send(msg)
            flash(_('Thank you for your message! We will get back to you shortly.'), 'success')
        except Exception as e:
            flash(_('An error occurred while sending the message. Please try again.'), 'danger')
            print(e)
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/history')
@login_required
def history():
    recordings = Recording.query.filter_by(user_id=current_user.id).order_by(Recording.timestamp.desc()).all()
    return render_template('history.html', recordings=recordings)

@app.route('/delete_recording/<int:recording_id>', methods=['POST'])
@login_required
def delete_recording(recording_id):
    recording = Recording.query.get_or_404(recording_id)
    if recording.user_id != current_user.id:
        return {"error": "Unauthorized"}, 403
    try:
        filepath = os.path.join(app.root_path, 'uploads', recording.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        db.session.delete(recording)
        db.session.commit()
        flash(_('Recording deleted successfully.'), 'success')
    except Exception as e:
        flash(_('An error occurred while deleting the file.'), 'danger')
        print(f"Error deleting file: {e}")
    return redirect(url_for('history'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        new_username = request.form.get('username')
        current_user.username = new_username
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
                os.makedirs(upload_path, exist_ok=True)
                file.save(os.path.join(upload_path, filename))
                current_user.picture = f"/{app.config['UPLOAD_FOLDER']}/{filename}"
        db.session.commit()
        flash(_('Profile updated successfully!'), 'success')
        return redirect(url_for('profile'))
    return render_template('edit_profile.html')

@app.route('/upload_audio', methods=['POST'])
@login_required
def upload_audio():
    audio_file = request.files.get('audio_data')
    if not audio_file:
        return {"error": "No audio file"}, 400
    
    upload_folder = os.path.join(app.root_path, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = secure_filename(f"user_{current_user.id}_{timestamp_str}.wav")
    filepath = os.path.join(upload_folder, filename)
    audio_file.save(filepath)
    
    new_recording = Recording(filename=filename, user_id=current_user.id)
    db.session.add(new_recording)
    db.session.commit()
    
    return {"success": True, "filename": filename}

# --- 4. CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    app.run(debug=True)