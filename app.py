import os
import re
import json
import random
import string
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory, redirect, render_template, make_response, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, verify_jwt_in_request
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer, util
import jieba.analyse
import pdfplumber
from dotenv import load_dotenv
from sqlalchemy.exc import OperationalError
from functools import wraps
from flask_babel import Babel

# --- Flask-Admin 依赖 ---
from flask_admin import Admin, AdminIndexView, expose, BaseView
from flask_admin.contrib.sqla import ModelView
from flask_admin.actions import action
from flask_admin.menu import MenuLink
from markupsafe import Markup

# ==========================================
# 0. 配置与初始化
# ==========================================
load_dotenv()

app = Flask(__name__, static_folder='.', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app)

# --- 数据库配置 ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- 安全配置 ---
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'default-unsafe-jwt-secret')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-unsafe-flask-secret')

# --- 汉化配置 ---
app.config['BABEL_DEFAULT_LOCALE'] = 'zh_CN'

db = SQLAlchemy(app)
jwt = JWTManager(app)


# --- Babel 初始化 (修复版) ---
def get_locale():
    return 'zh_CN'


babel = Babel(app, locale_selector=get_locale)

# ==========================================
# AI 模型管理 (懒加载池)
# ==========================================
# 定义支持的模型列表
MODEL_OPTIONS = {
    # 默认：多语言通用 (平衡)
    'default': 'paraphrase-multilingual-MiniLM-L12-v2',

    # 中文高精：BGE v1.5 (智源出品，目前中文语义匹配 SOTA)
    'chinese_acc': 'BAAI/bge-base-zh-v1.5',

    # 中文长文本：Jina v2 (支持 8k 长度，适合超长简历)
    'chinese_long': 'jinaai/jina-embeddings-v2-base-zh'
}

# 全局模型缓存池
model_cache = {}


def get_model(model_key='default'):
    """按需懒加载模型，避免启动时撑爆内存"""
    model_name = MODEL_OPTIONS.get(model_key, MODEL_OPTIONS['default'])

    if model_name in model_cache:
        return model_cache[model_name]

    print(f">>> [系统] 正在动态加载模型: {model_name} (请耐心等待下载...)")
    try:
        # trust_remote_code=True 是为了 Jina/BGE 等自定义模型
        new_model = SentenceTransformer(model_name, trust_remote_code=True)
        model_cache[model_name] = new_model
        print(f">>> [系统] 模型 {model_name} 加载就绪！")
        return new_model
    except Exception as e:
        print(f">>> [错误] 模型加载失败: {e}")
        # 如果加载失败，尝试回退到默认
        if model_key != 'default':
            print(">>> [系统] 尝试回退到默认模型...")
            return get_model('default')
        return None


# ==========================================
# 1. 数据库模型 (Schema)
# ==========================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # user, admin, super_admin
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime)

    def __str__(self):
        return self.username


class InviteCode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False)
    is_used = db.Column(db.Boolean, default=False)
    created_by = db.Column(db.String(80))
    used_by = db.Column(db.String(80))
    used_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.now)


class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(20), default='info')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)


class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    owner = db.relationship('User', backref=db.backref('resumes', lazy=True))

    name = db.Column(db.String(100))
    role = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(50))
    edu = db.Column(db.String(100))
    loc = db.Column(db.String(100))

    skills_json = db.Column(db.Text)
    summary = db.Column(db.Text)
    experience_json = db.Column(db.Text)

    is_public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "email": self.email,
            "phone": self.phone,
            "edu": self.edu,
            "location": self.loc,
            "skills": json.loads(self.skills_json) if self.skills_json else [],
            "summary": self.summary,
            "experience": json.loads(self.experience_json) if self.experience_json else [],
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "matchScore": 0
        }

    def __str__(self):
        return f"{self.name} - {self.role}"


# ==========================================
# 2. Flask-Admin 后台配置
# ==========================================

class DashboardView(AdminIndexView):
    @expose('/')
    def index(self):
        user_count = User.query.count()
        resume_count = Resume.query.count()
        invite_unused = InviteCode.query.filter_by(is_used=False).count()
        active_announcements = Announcement.query.filter_by(is_active=True).count()

        return self.render(
            'admin/dashboard_index.html',
            user_count=user_count,
            resume_count=resume_count,
            invite_unused=invite_unused,
            active_announcements=active_announcements
        )

    def is_accessible(self):
        return True


class MyModelView(ModelView):
    can_view_details = True
    page_size = 20

    def is_accessible(self):
        return True


class UserModelView(MyModelView):
    can_create = False
    column_list = ('username', 'role', 'is_active', 'created_at', 'last_login')
    column_searchable_list = ['username']
    column_filters = ['role', 'is_active']
    column_labels = dict(username='用户名', role='角色', is_active='账号状态', created_at='注册时间',
                         last_login='最近登录', password_hash='密码哈希')
    form_choices = {'role': [('user', '普通用户'), ('admin', '管理员'), ('super_admin', '超级管理员')]}

    def on_model_change(self, form, model, is_created):
        if 'password_hash' in form and form.password_hash.data:
            if not form.password_hash.data.startswith('scrypt:'):
                model.password_hash = generate_password_hash(form.password_hash.data)


class InviteCodeModelView(MyModelView):
    column_list = ('code', 'is_used', 'used_by', 'used_at', 'created_at')
    column_filters = ['is_used']
    column_labels = dict(code='邀请码', is_used='是否已用', used_by='使用者', used_at='使用时间', created_at='生成时间',
                         created_by='创建人')


class AnnouncementModelView(MyModelView):
    column_list = ('title', 'type', 'is_active', 'created_at')
    column_labels = dict(title='公告标题', content='公告内容', type='类型', is_active='是否显示', created_at='发布时间')
    form_choices = {'type': [('info', '通知 (蓝色)'), ('warning', '警告 (橙色)'), ('success', '更新 (绿色)')]}


class ResumeModelView(MyModelView):
    column_list = ('owner', 'name', 'role', 'matchScore', 'created_at')
    column_filters = ['owner.username', 'role']
    column_labels = dict(owner='归属用户', name='候选人姓名', role='求职意向', email='邮箱', phone='电话', edu='学历',
                         loc='地点', summary='简介', is_public='是否公开', created_at='上传时间',
                         matchScore='最近匹配分(缓存)')


# 初始化后台 (移除了 template_mode)
admin = Admin(app, name='招聘助手后台', index_view=DashboardView())

admin.add_view(UserModelView(User, db.session, name='用户管理'))
admin.add_view(InviteCodeModelView(InviteCode, db.session, name='邀请码'))
admin.add_view(AnnouncementModelView(Announcement, db.session, name='系统公告'))
admin.add_view(ResumeModelView(Resume, db.session, name='简历库'))
admin.add_link(MenuLink(name='⬅️ 返回前台', url='/'))


# ==========================================
# 3. 工具函数
# ==========================================
def normalize_score(raw_score):
    threshold = 0.2
    max_score_anchor = 0.65
    if raw_score < threshold: return 0
    score = (raw_score - threshold) / (max_score_anchor - threshold) * 100
    return int(min(max(score, 0), 99))


def calculate_keyword_score(text, keywords):
    if not keywords: return 0
    text_lower = text.lower()
    match_count = sum(1 for kw in keywords if kw.lower() in text_lower)
    return (match_count / len(keywords)) * 100


def parse_resume_text(text):
    data = {"name": "", "email": "", "phone": "", "edu": "", "role": "", "loc": "", "skills": "", "summary": "",
            "exp_company": "", "exp_title": "", "exp_desc": ""}
    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    phone_match = re.search(r'1[3-9]\d{9}', text)
    if email_match: data['email'] = email_match.group()
    if phone_match: data['phone'] = phone_match.group()

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if re.search(r'姓\s*名', line) and not data['name']:
            data['name'] = re.sub(r'姓\s*名[:：]\s*', '', line)
        elif re.search(r'学\s*历', line) and not data['edu']:
            data['edu'] = re.sub(r'学\s*历[:：]\s*', '', line)
        elif re.search(r'求\s*职\s*意\s*向', line) and not data['role']:
            data['role'] = re.sub(r'求\s*职\s*意\s*向[:：]\s*', '', line)
        elif re.search(r'(坐\s*标|地\s*点)', line):
            data['loc'] = re.sub(r'(坐\s*标|地\s*点)[:：]\s*', '', line)

    blocks = re.split(r'【|\[', text)
    for block in blocks:
        if "技能" in block or "Skills" in block:
            content = re.sub(r'.*(技能|Skills).*', '', block, 1).strip().replace(']', '').replace('】', '')
            data['skills'] = content.replace('\n', ', ').replace('- ', '')
        elif "简介" in block or "Summary" in block:
            content = re.sub(r'.*(简介|Summary).*', '', block, 1).strip().replace(']', '').replace('】', '')
            data['summary'] = content
        elif "经历" in block or "Experience" in block:
            content = re.sub(r'.*(经历|Experience).*', '', block, 1).strip().replace(']', '').replace('】', '')
            exp_lines = [l for l in content.split('\n') if l.strip()]
            if exp_lines:
                first = exp_lines[0]
                if '|' in first:
                    parts = first.split('|')
                    data['exp_title'] = parts[0].strip()
                    data['exp_company'] = parts[1].strip() if len(parts) > 1 else ""
                elif '-' in first:
                    parts = first.split('-')
                    data['exp_company'] = parts[0].strip()
                    data['exp_title'] = parts[1].strip() if len(parts) > 1 else ""
                else:
                    data['exp_title'] = first
                data['exp_desc'] = "\n".join(exp_lines[1:])

    if not data['name'] and lines:
        pot = lines[0].strip()
        if len(pot) < 10: data['name'] = pot
    return data


# ==========================================
# 4. API 接口
# ==========================================
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    code_str = data.get('invitation_code')
    if not username or not password: return jsonify({"error": "账号密码不能为空"}), 400
    invite = InviteCode.query.filter_by(code=code_str, is_used=False).first()
    if not invite: return jsonify({"error": "邀请码无效或已被使用"}), 403
    if len(password) < 8 or not re.search(r"[a-zA-Z]", password) or not re.search(r"\d", password): return jsonify(
        {"error": "密码过弱"}), 400
    if User.query.filter_by(username=username).first(): return jsonify({"error": "用户名已存在"}), 400
    new_user = User(username=username, password_hash=generate_password_hash(password), role='user')
    invite.is_used = True
    invite.used_by = username
    invite.used_at = datetime.now()
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "注册成功"}), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password): return jsonify(
        {"error": "用户名或密码错误"}), 401
    if not user.is_active: return jsonify({"error": "账号已被封禁"}), 403
    user.last_login = datetime.now()
    db.session.commit()
    return jsonify({"token": create_access_token(identity=str(user.id)), "username": username, "role": user.role}), 200


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        user = User.query.get(int(get_jwt_identity()))
        if not user or user.role not in ['admin', 'super_admin']: return jsonify({"error": "需要管理员权限"}), 403
        return fn(*args, **kwargs)

    return wrapper


@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    return jsonify({
        "user_count": User.query.count(),
        "resume_count": Resume.query.count(),
        "invite_unused": InviteCode.query.filter_by(is_used=False).count(),
        "active_announcements": Announcement.query.filter_by(is_active=True).count()
    })


@app.route('/api/admin/invite_codes', methods=['GET', 'POST'])
@admin_required
def admin_invite_codes():
    if request.method == 'POST':
        count = request.json.get('count', 1)
        new_codes = []
        creator = User.query.get(int(get_jwt_identity())).username
        for _ in range(count):
            code_str = 'HR-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            new_code = InviteCode(code=code_str, created_by=creator)
            db.session.add(new_code)
            new_codes.append(code_str)
        db.session.commit()
        return jsonify({"message": "success", "codes": new_codes})
    codes = InviteCode.query.filter_by(is_used=False).order_by(InviteCode.created_at.desc()).limit(10).all()
    return jsonify([{"code": c.code, "created_at": c.created_at} for c in codes])


@app.route('/api/admin/announcement', methods=['POST'])
@admin_required
def admin_publish_announcement():
    data = request.json
    if data.get('overwrite', False): Announcement.query.update({Announcement.is_active: False})
    new_ann = Announcement(title=data.get('title'), content=data.get('content'), type=data.get('type', 'info'),
                           is_active=True)
    db.session.add(new_ann)
    db.session.commit()
    return jsonify({"message": "公告发布成功"})


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/announcement', methods=['GET'])
def get_announcement():
    ann = Announcement.query.filter_by(is_active=True).order_by(Announcement.created_at.desc()).first()
    if ann: return jsonify({"title": ann.title, "content": ann.content, "type": ann.type})
    return jsonify({})


@app.route('/api/match', methods=['POST'])
@jwt_required()
def match_resumes():
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        data = request.json
        current_jd = data.get('jd_text', "")

        # === 懒加载模型 ===
        selected_model_key = data.get('model_type', 'default')
        current_model = get_model(selected_model_key)

        if current_model is None:
            return jsonify({"error": "AI 模型加载失败"}), 500

        if user.role in ['admin', 'super_admin']:
            resumes = Resume.query.all()
        else:
            resumes = Resume.query.filter_by(user_id=user_id).all()

        if not resumes: return jsonify([])
        resumes_list = [r.to_dict() for r in resumes]

        # 关键词提取
        jd_keywords = jieba.analyse.extract_tags(current_jd, topK=10)
        # 向量编码
        jd_emb = current_model.encode(current_jd, convert_to_tensor=True)

        for r in resumes_list:
            skills = " ".join(r.get('skills', []))
            exps = " ".join([f"{e.get('title', '')} {e.get('desc', '')}" for e in r.get('experience', [])])
            full_txt = f"{r.get('role', '')} {skills} {r.get('summary', '')} {r.get('edu', '')} {exps}"

            s_skills = normalize_score(
                util.cos_sim(current_model.encode(skills, convert_to_tensor=True), jd_emb).item())
            s_exp = normalize_score(util.cos_sim(current_model.encode(exps, convert_to_tensor=True), jd_emb).item())
            s_prof = normalize_score(
                util.cos_sim(current_model.encode(f"{r.get('summary', '')} {r.get('edu', '')}", convert_to_tensor=True),
                             jd_emb).item())
            s_kw = calculate_keyword_score(full_txt, jd_keywords)

            weighted = (s_skills * 0.4) + (s_exp * 0.4) + (s_prof * 0.2)
            r['matchScore'] = int((weighted * 0.5) + (s_kw * 0.5))
            r['radar'] = {'skills': s_skills, 'experience': s_exp, 'profile': s_prof}
            r['keywords'] = jd_keywords

        resumes_list.sort(key=lambda x: x['matchScore'], reverse=True)
        return jsonify(resumes_list)

    except Exception as e:
        print(f"Match error: {e}")
        return jsonify([]), 500


@app.route('/api/add_resume', methods=['POST'])
@jwt_required()
def add_resume():
    try:
        user_id = int(get_jwt_identity())
        data = request.json
        skills = data.get('skills', [])
        if isinstance(skills, str): skills = [s.strip() for s in skills.replace('，', ',').split(',') if s.strip()]
        new_resume = Resume(user_id=user_id, name=data.get('name', '未命名'), role=data.get('role', ''),
                            email=data.get('email', ''), phone=data.get('phone', ''), edu=data.get('edu', ''),
                            loc=data.get('location', ''), skills_json=json.dumps(skills),
                            summary=data.get('summary', ''), experience_json=json.dumps(data.get('experience', [])),
                            is_public=data.get('is_public', False))
        db.session.add(new_resume)
        db.session.commit()
        return jsonify({"message": "success", "id": new_resume.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/delete_resume/<int:resume_id>', methods=['DELETE'])
@jwt_required()
def delete_resume(resume_id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    resume = Resume.query.get(resume_id)
    if not resume: return jsonify({"error": "不存在"}), 404
    if resume.user_id != user_id and user.role not in ['admin', 'super_admin']: return jsonify(
        {"error": "无权删除"}), 403
    db.session.delete(resume)
    db.session.commit()
    return jsonify({"message": "success"})


@app.route('/api/parse_resume', methods=['POST'])
def parse_resume():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    try:
        text = ""
        with pdfplumber.open(request.files['file']) as pdf:
            for page in pdf.pages: text += page.extract_text() + "\n"
        return jsonify({"message": "success", "data": parse_resume_text(text)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            if InviteCode.query.count() == 0:
                db.session.add(InviteCode(code='HR-START-2025', created_by='System'))
                db.session.commit()
                print(f">>> 邀请码初始化: HR-START-2025")
            if not User.query.filter_by(role='super_admin').first():
                db.session.add(
                    User(username='admin', password_hash=generate_password_hash('123456'), role='super_admin'))
                db.session.commit()
                print(">>> 管理员: admin / 123456")
        except OperationalError as e:
            print(f"❌ 数据库错误: {e}")
            exit(1)
    print(">>> 启动 Flask 服务器...")
    app.run(host='0.0.0.0', port=7860)