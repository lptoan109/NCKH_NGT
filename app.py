import os
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
# --- Cấu hình cho Flask-Mail ---
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = config.EMAIL_USER # Đọc từ file config.py
app.config['MAIL_PASSWORD'] = config.EMAIL_PASS # Đọc từ file config.py
mail = Mail(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_super_secret_key_change_this' # GIỮ NGUYÊN SECRET KEY CŨ CỦA BẠN

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

# Model User được nâng cấp
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True) # Cho phép null cho user Google
    picture = db.Column(db.String(200), nullable=True) # Lưu link avatar Google
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and bcrypt.check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user)
            return redirect(url_for('diagnose'))
        else:
            flash('Sai tên đăng nhập hoặc mật khẩu.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')

        # --- PHẦN NÂNG CẤP ---
        # KIỂM TRA XEM EMAIL HOẶC USERNAME ĐÃ TỒN TẠI CHƯA
        existing_user_email = User.query.filter_by(email=email).first()
        if existing_user_email:
            flash('Địa chỉ email này đã được sử dụng.', 'danger')
            return redirect(url_for('register'))

        existing_user_username = User.query.filter_by(username=username).first()
        if existing_user_username:
            flash('Tên đăng nhập này đã tồn tại.', 'danger')
            return redirect(url_for('register'))
        # ---------------------

        # Nếu chưa tồn tại, tiếp tục tạo user mới
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Tạo tài khoản thành công! Vui lòng đăng nhập.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('user_info', None) # Xóa session của Google nếu có
    return redirect(url_for('homepage'))

@app.route('/login/google')
def login_google():
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

# Hàm authorize được nâng cấp để hợp nhất user
@app.route('/authorize')
def authorize():
    token = google.authorize_access_token()
    user_info = google.userinfo()
    
    # Tìm user trong DB bằng email từ Google
    user = User.query.filter_by(email=user_info['email']).first()
    
    # Nếu user chưa có, tạo mới trong DB
    if not user:
        user = User(
            email=user_info['email'],
            username=user_info['name'],
            picture=user_info['picture']
            # Không cần password_hash cho user Google
        )
        db.session.add(user)
        db.session.commit()
    
    # Đăng nhập người dùng bằng Flask-Login
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
            # Lấy dữ liệu từ form
            name = request.form.get('ho_ten')
            sender_email_from_form = request.form.get('email')
            subject = request.form.get('tieu_de')
            message_body = request.form.get('noi_dung')

            # Tạo email
            msg = Message(
                subject=f"Tin nhắn từ Web NGT Cough: {subject}",
                # NGƯỜI GỬI (sender) PHẢI LÀ EMAIL CỦA BẠN
                sender=("Website NGT Cough", config.EMAIL_USER),
                recipients=[config.EMAIL_USER] # Gửi về email của chính bạn
            )

            # Đưa thông tin người liên hệ vào nội dung thư
            msg.body = f"""
Bạn đã nhận được một tin nhắn mới từ:

Tên: {name}
Email: {sender_email_from_form}

Nội dung:
{message_body}
"""

            # Gửi đi
            mail.send(msg)

            flash('Cảm ơn bạn đã gửi tin nhắn! Chúng tôi sẽ phản hồi sớm.', 'success')
        except Exception as e:
            flash('Đã có lỗi xảy ra khi gửi tin nhắn. Vui lòng thử lại.', 'danger')
            print(e) # In lỗi ra server log để debug

        return redirect(url_for('contact'))

    return render_template('contact.html')

# (Các route khác như history, upload_audio giữ nguyên)
@app.route('/history')
@login_required
def history():
    recordings = Recording.query.filter_by(user_id=current_user.id).order_by(Recording.timestamp.desc()).all()
    return render_template('history.html', recordings=recordings)

@app.route('/delete_recording/<int:recording_id>', methods=['POST'])
@login_required
def delete_recording(recording_id):
    # Tìm bản ghi trong database
    recording = Recording.query.get_or_404(recording_id)
    
    # Đảm bảo người dùng chỉ có thể xóa bản ghi của chính mình
    if recording.user_id != current_user.id:
        return {"error": "Không có quyền truy cập"}, 403
        
    try:
        # Xóa file trên máy chủ
        filepath = os.path.join('uploads', recording.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            
        # Xóa bản ghi trong database
        db.session.delete(recording)
        db.session.commit()
        
        flash('Đã xóa bản ghi thành công.', 'success')
    except Exception as e:
        flash('Đã có lỗi xảy ra khi xóa file.', 'danger')
        print(f"Lỗi khi xóa file: {e}") # Ghi log lỗi ra console
    
    return redirect(url_for('history'))
    
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

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
# --- 4. Chạy ứng dụng ---
if __name__ == '__main__':
    app.run(debug=True)