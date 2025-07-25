import os
from flask_babel import Babel, _
from flask import session
from flask_mail import Mail, Message
import config
from flask import Flask, url_for, session, redirect, render_template, request, flash
from authlib.integrations.flask_client import OAuth
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

# --- 1. Cài đặt và Cấu hình ---
app = Flask(__name__)

app.config['SECRET_KEY'] = 'your-very-secret-key'

# --- CẤU HÌNH BABEL ---
app.config['LANGUAGES'] = {
    'en': 'English',
    'vi': 'Tiếng Việt'
}
# 1. Định nghĩa hàm get_locale một cách bình thường (không có @ ở trên)
def get_locale():
    if 'language' in session:
        return session['language']
    return request.accept_languages.best_match(app.config['LANGUAGES'].keys())

# 2. Truyền thẳng hàm vào khi khởi tạo Babel
babel = Babel(app, locale_selector=get_locale)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Cấu hình cho Flask-Mail ---
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = config.EMAIL_USER
app.config['MAIL_PASSWORD'] = config.EMAIL_PASS
mail = Mail(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_super_secret_key_change_this'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Cấu hình Google Login ---
GOOGLE_CLIENT_ID = '564904327189-4gsii5kfkht070218tsjqu8amnstc7o1.apps.googleusercontent.com'
GOOGLE_CLIENT_SECRET = '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'
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- 2. Định nghĩa các Model cho Database ---
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

# --- 3. Định nghĩa các trang (Routes) ---

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
        if user and bcrypt.check_password_hash(user.password_hash, request.form.get('password')):
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
            name = request.form.get('ho_ten')
            sender_email_from_form = request.form.get('email')
            subject = request.form.get('tieu_de')
            message_body = request.form.get('noi_dung')

            msg = Message(
                subject=f"Tin nhắn từ Web NGT Cough: {subject}",
                sender=("Website NGT Cough", config.EMAIL_USER),
                recipients=[config.EMAIL_USER]
            )
            msg.body = f"""
Bạn đã nhận được một tin nhắn mới từ:

Tên: {name}
Email: {sender_email_from_form}

Nội dung:
{message_body}
"""
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
        return {"error": "Không có quyền truy cập"}, 403
        
    try:
        filepath = os.path.join('uploads', recording.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            
        db.session.delete(recording)
        db.session.commit()
        
        flash(_('Recording deleted successfully.'), 'success')
    except Exception as e:
        flash(_('An error occurred while deleting the file.'), 'danger')
        print(f"Lỗi khi xóa file: {e}")
    
    return redirect(url_for('history'))
    
if not os.path.exists('uploads'):
    os.makedirs('uploads')

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
        return {"error": "Không có file âm thanh"}, 400
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = secure_filename(f"user_{current_user.id}_{timestamp_str}.wav")
    filepath = os.path.join('uploads', filename)
    audio_file.save(filepath)
    
    new_recording = Recording(filename=filename, user_id=current_user.id)
    db.session.add(new_recording)
    db.session.commit()
    
    return {"success": True, "filename": filename}

# --- 4. Chạy ứng dụng ---
if __name__ == '__main__':
    app.run(debug=True)