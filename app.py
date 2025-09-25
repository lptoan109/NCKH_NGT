import os
from datetime import datetime
import config # Import file config.py
import random

from itsdangerous import URLSafeTimedSerializer
from flask import Flask, url_for, session, redirect, render_template, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from ai_predictor import CoughPredictor
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

# Khởi tạo các extension
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
oauth = OAuth(app)

# --- Cấu hình Google Login ---
# !!! THAY THẾ BẰNG CLIENT ID VÀ SECRET ĐÚNG TỪ "Web client 1" !!!
google = oauth.register(
    name='google',
    client_id='564904327189-4gsii5kfkht070218tsjqu8amnstc7o1.apps.googleusercontent.com', # <-- GIÁ TRỊ ĐÚNG
    client_secret='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', # <-- THAY BẰNG MÃ BÍ MẬT ĐÚNG CỦA BẠN
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- TẢI MODEL AI MỘT LẦN KHI SERVER KHỞI ĐỘNG ---
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'student_model.tflite')
ai_model = CoughPredictor(model_path=MODEL_FILE_PATH)

# --- 2. ĐỊNH NGHĨA MODEL ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)
    picture = db.Column(db.String(200), nullable=True)
    recordings = db.relationship('Recording', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    # Thêm các cột để lưu kết quả
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 3. ĐỊNH NGHĨA ROUTE ---

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.password_hash and bcrypt.check_password_hash(user.password_hash, request.form.get('password')):
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

        existing_user_email = User.query.filter_by(email=email).first()
        if existing_user_email:
            flash('Địa chỉ email này đã được sử dụng.', 'danger')
            return redirect(url_for('register'))

        existing_user_username = User.query.filter_by(username=username).first()
        if existing_user_username:
            flash('Tên đăng nhập này đã tồn tại.', 'danger')
            return redirect(url_for('register'))

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
    session.pop('user_info', None)
    return redirect(url_for('homepage'))

@app.route('/login/google')
def login_google():
    # Sửa http thành https ở dòng dưới
    redirect_uri = 'https://ngt.pythonanywhere.com/authorize'
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
def diagnose():
    # Route này không cần login, ai cũng có thể truy cập trang
    return render_template('diagnose.html')

@app.route('/about')
def about():
    return render_template('about.html')

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
            flash('Cảm ơn bạn đã gửi tin nhắn! Chúng tôi sẽ phản hồi sớm.', 'success')
        except Exception as e:
            flash('Đã có lỗi xảy ra khi gửi tin nhắn. Vui lòng thử lại.', 'danger')
            print(e)
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    pagination = Recording.query.filter_by(user_id=current_user.id)\
                                .order_by(Recording.timestamp.desc())\
                                .paginate(page=page, per_page=10, error_out=False)
    recordings = pagination.items
    return render_template('history.html', recordings=recordings, pagination=pagination)

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
        flash('Đã xóa bản ghi thành công.', 'success')
    except Exception as e:
        flash('Đã có lỗi xảy ra khi xóa file.', 'danger')
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
        flash('Cập nhật thông tin thành công!', 'success')
        return redirect(url_for('profile'))
    return render_template('edit_profile.html')

# @app.route('/upload_audio', methods=['POST'])
# def upload_audio():
#     """
#     Xử lý file âm thanh được tải lên, dự đoán và lưu kết quả.
#     Route này không yêu cầu đăng nhập.
#     """
#     audio_file = request.files.get('audio_data')
#     if not audio_file:
#         return jsonify({"error": "Không có file âm thanh"}), 400

#     # --- Logic lưu file và dự đoán được đơn giản hóa ---
    
#     # 1. Tạo một tên file duy nhất
#     timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#     # Phân biệt file của user và guest
#     user_prefix = f"user_{current_user.id}" if current_user.is_authenticated else "guest"
#     filename = secure_filename(f"{user_prefix}_{timestamp_str}.wav")
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#     # 2. Lưu file MỘT LẦN DUY NHẤT
#     audio_file.save(filepath)
    
#     try:
#         # 3. Gọi model AI để xử lý file đã lưu
#         ai_result = ai_model.predict(filepath)

#         # 4. Nếu người dùng đã đăng nhập, lưu kết quả vào lịch sử
#         if current_user.is_authenticated:
#             if 'error' not in ai_result:
#                 new_prediction = Prediction(
#                     filename=filename,
#                     user_id=current_user.id,
#                     result=ai_result.get('predicted_class', 'N/A'),
#                     confidence=ai_result.get('confidence', '0%')
#                 )
#                 db.session.add(new_prediction)
#                 db.session.commit()
#         else:
#             # Nếu là khách, chúng ta có thể xóa file tạm sau khi dự đoán để tiết kiệm dung lượng
#             os.remove(filepath) # Bỏ comment dòng này nếu bạn muốn xóa file của guest
#             pass

#         # 5. Trả về kết quả chẩn đoán cho frontend
#         return jsonify({"success": True, "diagnosis_result": ai_result})

#     except Exception as e:
#         # Xử lý nếu có lỗi từ model AI
#         print(f"Lỗi khi dự đoán AI: {e}")
#         return jsonify({"error": "Lỗi máy chủ trong quá trình phân tích"}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_file = request.files.get('audio_data')
    if not audio_file:
        return {"error": "No audio file"}, 400

    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- LOGIC CHẨN ĐOÁN GIẢ TẠM THỜI ---
    diseases = ["Bệnh lao", "Hen suyễn", "Covid"]
    roll = random.randint(1, 100)
    diagnosis_result = ""

    if roll <= 79:
        diagnosis_result = "Khỏe mạnh"
    else:
        # 21% còn lại chia đều cho 3 bệnh
        diagnosis_result = random.choice(diseases)
    # --- KẾT THÚC LOGIC TẠM THỜI ---

    filename = ""
    # Kiểm tra nếu người dùng đã đăng nhập để lưu lịch sử
    if current_user.is_authenticated:
        filename = secure_filename(f"user_{current_user.id}_{timestamp_str}.wav")
        filepath = os.path.join(upload_folder, filename)
        audio_file.save(filepath)

        new_recording = Recording(filename=filename, user_id=current_user.id)
        db.session.add(new_recording)
        db.session.commit()
    else:
        # Nếu là khách, chỉ lưu file tạm và không lưu vào database
        filename = secure_filename(f"guest_{timestamp_str}.wav")
        filepath = os.path.join(upload_folder, filename)
        audio_file.save(filepath)

    # Trả về kết quả chẩn đoán trong JSON response
    return {"success": True, "filename": f"/static/uploads/{filename}", "diagnosis_result": diagnosis_result}


# --- 4. CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

# app.py

# --- HÀM TẠO TOKEN BẢO MẬT ---
def generate_reset_token(email):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    return serializer.dumps(email, salt='password-reset-salt')

# --- HÀM XÁC THỰC TOKEN ---
def confirm_reset_token(token, expiration=3600): # Hết hạn sau 1 giờ
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = serializer.loads(
            token,
            salt='password-reset-salt',
            max_age=expiration
        )
        return email
    except Exception:
        return False

# --- ROUTE XỬ LÝ QUÊN MẬT KHẨU ---
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            token = generate_reset_token(user.email)
            reset_url = url_for('reset_password', token=token, _external=True)
            
            # Soạn và gửi email
            msg = Message('Yêu cầu đặt lại mật khẩu - NGT Cough',
                          sender=("Website NGT Cough", app.config['MAIL_USERNAME']),
                          recipients=[user.email])
            msg.body = f'''Chào {user.username},

Để đặt lại mật khẩu của bạn, vui lòng nhấn vào đường link sau:
{reset_url}

Nếu bạn không phải là người yêu cầu, vui lòng bỏ qua email này. Link sẽ hết hạn sau 1 giờ.

Trân trọng,
Đội ngũ NGT Cough'''
            mail.send(msg)
            
        # Dù email có tồn tại hay không, vẫn hiện thông báo này để bảo mật
        flash('Nếu email của bạn tồn tại trong hệ thống, một hướng dẫn đặt lại mật khẩu đã được gửi đến.', 'success')
        return redirect(url_for('login'))
        
    return render_template('forgot_password.html')


# --- ROUTE XỬ LÝ ĐẶT LẠI MẬT KHẨU ---
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = confirm_reset_token(token)
    if not email:
        flash('Đường link đặt lại mật khẩu không hợp lệ hoặc đã hết hạn.', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        
        user = User.query.filter_by(email=email).first()
        user.password_hash = hashed_password
        db.session.commit()
        
        flash('Mật khẩu của bạn đã được cập nhật thành công! Vui lòng đăng nhập.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)