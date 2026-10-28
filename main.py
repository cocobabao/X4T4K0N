import asyncio
import sqlite3
import uuid
import re
import hashlib
import subprocess
import json
import requests
import numpy as np
import jwt
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
from faster_whisper import WhisperModel

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer
import imageio_ffmpeg

BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "uploads"
DB_PATH = BASE / "videos.db"
HTML_PATH = BASE / "index.html"

# Очищаем папку uploads при старте, чтобы не было мусора от старых запусков
UPLOAD_DIR.mkdir(exist_ok=True)


FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

GROQ_API_KEY = "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"
JWT_SECRET = "cyberryvok_super_secret_jwt_key_2026"
JWT_ALGORITHM = "HS256"

print("=== ЗАГРУЗКА МОДЕЛИ WHISPER (small) ===")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

print("=== ЗАГРУЗКА МОДЕЛИ ЭМБЕДДИНГОВ (E5) ===")
embed_model = SentenceTransformer('intfloat/multilingual-e5-small')
print("=== ВСЕ МОДЕЛИ ГОТОВЫ! СЕРВЕР ЗАПУЩЕН ===")

app = FastAPI(title="Video -> Text Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

def db():
    con = sqlite3.connect(DB_PATH, timeout=20, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
                fio TEXT NOT NULL, is_admin INTEGER DEFAULT 0, created_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS registrations (
                id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
                fio TEXT NOT NULL, created_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS stats_logs (
                id TEXT PRIMARY KEY, action_type TEXT NOT NULL, created_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id TEXT PRIMARY KEY, user_id TEXT, cache_key TEXT UNIQUE, source TEXT NOT NULL, 
                original TEXT, thumbnail TEXT, audio_path TEXT, text TEXT, summary TEXT, 
                method TEXT, status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, video_id TEXT, chunk_text TEXT, 
                start_sec REAL, end_sec REAL, embedding TEXT,
                FOREIGN KEY(video_id) REFERENCES transcriptions(id)
            )
        """)
        admin = con.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
        if not admin:
            con.execute(
                "INSERT INTO users (id, username, password, fio, is_admin, created_at) VALUES (?, ?, ?, ?, 1, ?)",
                ("admin_id_001", "admin", "admin123", "Главный Администратор Системы", datetime.now(timezone.utc).isoformat())
            )

init_db()

def create_jwt_token(user_data: dict) -> str:
    payload = {
        "user_id": user_data["id"], "username": user_data["username"],
        "fio": user_data["fio"], "is_admin": bool(user_data["is_admin"]),
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Отсутствует токен авторизации")
    token = authorization.split(" ")[1]
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        raise HTTPException(401, "Токен недействителен или истёк")

def log_stat(action_type: str):
    with db() as con:
        con.execute("INSERT INTO stats_logs (id, action_type, created_at) VALUES (?, ?, ?)",
            (uuid.uuid4().hex, action_type, datetime.now(timezone.utc).isoformat()))

def get_cache_key(url: str) -> str:
    if "youtube.com" in url or "youtu.be" in url:
        parsed = urlparse(url)
        q = parse_qs(parsed.query)
        if "v" in q: return q['v'][0]
        if "youtu.be" in parsed.netloc: return parsed.path.lstrip('/')
    return url

def parse_vtt(vtt_content: str) -> str:
    content = re.sub(r'^WEBVTT.*?\n\n', '', vtt_content, flags=re.DOTALL)
    content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n', '', content)
    content = re.sub(r'<[^>]+>', '', content)
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    result = []
    for line in lines:
        if not result or result[-1] != line: result.append(line)
    return " ".join(result)

def update_status(vid_id, status, error=None, text=None, method=None, audio_path=None):
    with db() as con:
        if status == "ready":
            con.execute("UPDATE transcriptions SET status=?, text=?, error=?, method=?, audio_path=? WHERE id=?",
                (status, text, error, method, audio_path, vid_id))
        else:
            con.execute("UPDATE transcriptions SET status=?, error=? WHERE id=?", (status, error, vid_id))

def format_time(seconds: float) -> str:
    if seconds is None: return "00:00"
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = int(m // 60)
    m = m % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def index_segments(video_id: str, segments_list):
    if not segments_list: return
    chunks, curr_text = [], ""
    curr_start, curr_end = segments_list[0].start, segments_list[0].end

    for seg in segments_list:
        text = seg.text.strip()
        if not text: continue
        if len(curr_text) + len(text) < 400:
            curr_text += " " + text
            curr_end = seg.end
        else:
            if curr_text: chunks.append({"text": curr_text.strip(), "start": curr_start, "end": curr_end})
            curr_text, curr_start, curr_end = text, seg.start, seg.end

    if curr_text: chunks.append({"text": curr_text.strip(), "start": curr_start, "end": curr_end})
    if not chunks: return

    texts_to_embed = ["passage: " + c["text"] for c in chunks]
    embeddings = embed_model.encode(texts_to_embed)

    with db() as con:
        con.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
        for c, emb in zip(chunks, embeddings):
            con.execute("""INSERT INTO chunks (id, video_id, chunk_text, start_sec, end_sec, embedding) 
                VALUES (?, ?, ?, ?, ?, ?)""", (uuid.uuid4().hex, video_id, c["text"], c["start"], c["end"], json.dumps(emb.tolist())))

def get_cached(cache_key: str):
    with db() as con:
        row = con.execute("SELECT * FROM transcriptions WHERE cache_key = ? AND status = 'ready'", (cache_key,)).fetchone()
        if row: return {"id": row["id"], "status": "ready", "text": row["text"], "error": None, "cached": True}
    return None

@app.get("/", response_class=HTMLResponse)
def index():
    if not HTML_PATH.exists(): raise HTTPException(404, "Файл index.html не найден.")
    return HTML_PATH.read_text(encoding="utf-8")

class AuthIn(BaseModel):
    username: str
    password: str

class RegisterIn(BaseModel):
    username: str
    password: str
    fio: str

@app.post("/api/auth/login")
def login(payload: AuthIn):
    with db() as con:
        user = con.execute("SELECT * FROM users WHERE username = ? AND password = ?", (payload.username.strip(), payload.password.strip())).fetchone()
        if not user: raise HTTPException(401, "Неверный логин или пароль")
        log_stat("auth")
        token = create_jwt_token(dict(user))
        return {"status": "success", "token": token, "user": {"id": user["id"], "username": user["username"], "fio": user["fio"], "is_admin": bool(user["is_admin"])}}

@app.get("/api/auth/me")
def verify_me(current_user: dict = Depends(decode_jwt_token)):
    with db() as con:
        user = con.execute("SELECT id, username, fio, is_admin FROM users WHERE id = ?", (current_user["user_id"],)).fetchone()
        if not user: raise HTTPException(401, "Пользователь удален")
        return {"user": {"id": user["id"], "username": user["username"], "fio": user["fio"], "is_admin": bool(user["is_admin"])}}

@app.post("/api/auth/register")
def register(payload: RegisterIn):
    uname, pwd, fio = payload.username.strip(), payload.password.strip(), payload.fio.strip()
    if not uname or not pwd or not fio: raise HTTPException(400, "Все поля обязательны")
    if fio.count(' ') < 2: raise HTTPException(400, "В ФИО должно быть минимум 2 пробела")
    with db() as con:
        if con.execute("SELECT id FROM users WHERE username = ?", (uname,)).fetchone() or con.execute("SELECT id FROM registrations WHERE username = ?", (uname,)).fetchone():
            raise HTTPException(400, "Пользователь уже зарегистрирован или ожидает заявку")
        con.execute("INSERT INTO registrations (id, username, password, fio, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, uname, pwd, fio, datetime.now(timezone.utc).isoformat()))
    return {"status": "pending", "message": "Заявка отправлена администратору!"}

@app.get("/api/admin/requests")
def get_registration_requests(current_user: dict = Depends(decode_jwt_token)):
    if not current_user.get("is_admin"): raise HTTPException(403, "Доступ запрещен")
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM registrations ORDER BY created_at DESC").fetchall()]

@app.post("/api/admin/requests/{req_id}/decision")
def request_decision(req_id: str, action: dict, current_user: dict = Depends(decode_jwt_token)):
    if not current_user.get("is_admin"): raise HTTPException(403, "Доступ запрещен")
    with db() as con:
        req = con.execute("SELECT * FROM registrations WHERE id = ?", (req_id,)).fetchone()
        if not req: raise HTTPException(404, "Заявка не найдена")
        if action.get("approve", False):
            con.execute("INSERT INTO users (id, username, password, fio, is_admin, created_at) VALUES (?, ?, ?, ?, 0, ?)",
                (uuid.uuid4().hex, req["username"], req["password"], req["fio"], datetime.now(timezone.utc).isoformat()))
        con.execute("DELETE FROM registrations WHERE id = ?", (req_id,))
    return {"status": "ok"}

@app.get("/api/admin/users")
def get_all_users(current_user: dict = Depends(decode_jwt_token)):
    if not current_user.get("is_admin"): raise HTTPException(403, "Доступ запрещен")
    with db() as con:
        users = con.execute("SELECT id, username, password, fio, created_at, is_admin FROM users WHERE is_admin = 0 ORDER BY created_at DESC").fetchall()
        result = []
        for u in users:
            u_dict = dict(u)
            lectures = con.execute("SELECT * FROM transcriptions WHERE user_id = ? ORDER BY created_at DESC", (u["id"],)).fetchall()
            formatted = []
            for l in lectures:
                l_dict = dict(l)
                ck = l_dict.get("cache_key", "")
                orig = l_dict.get("original", "")
                real_url = orig if orig.startswith("http") else (f"https://youtu.be/{ck}" if len(ck) == 11 else "https://youtube.com")
                l_dict["display_source"] = "файл" if l_dict["source"] == "file" else real_url
                l_dict["watch_url"] = None if l_dict["source"] == "file" else real_url
                formatted.append(l_dict)
            u_dict["lectures"] = formatted
            result.append(u_dict)
        return result

@app.delete("/api/admin/users/{user_id}")
def delete_user(user_id: str, current_user: dict = Depends(decode_jwt_token)):
    if not current_user.get("is_admin"): raise HTTPException(403, "Доступ запрещен")
    with db() as con:
        if not con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone(): raise HTTPException(404, "Пользователь не найден")
        lectures = con.execute("SELECT id FROM transcriptions WHERE user_id = ?", (user_id,)).fetchall()
        for l in lectures:
            vid_id = l["id"]
            con.execute("DELETE FROM chunks WHERE video_id = ?", (vid_id,))
            for file_to_del in UPLOAD_DIR.glob(f"{vid_id}*"):
                try: file_to_del.unlink(missing_ok=True)
                except Exception: pass
        con.execute("DELETE FROM transcriptions WHERE user_id = ?", (user_id,))
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return {"status": "deleted"}

@app.get("/api/admin/stats")
def get_statistics(current_user: dict = Depends(decode_jwt_token)):
    if not current_user.get("is_admin"): raise HTTPException(403, "Доступ запрещен")
    with db() as con:
        return {
            "auths": [r["created_at"] for r in con.execute("SELECT created_at FROM stats_logs WHERE action_type = 'auth'").fetchall()],
            "transcriptions": [r["created_at"] for r in con.execute("SELECT created_at FROM stats_logs WHERE action_type = 'transcription'").fetchall()],
            "urls": [r["created_at"] for r in con.execute("SELECT created_at FROM stats_logs WHERE action_type = 'url_submit'").fetchall()]
        }

def process_media_task(vid_id: str, source_target: str, is_url: bool = True):
    media_path = ""
    temp_audio_path = str(UPLOAD_DIR / f"{vid_id}_temp_audio.mp3")
    downloaded_file = None
    
    try:
        print(f"\n[FSTEPS] Стартовала фоновая задача для ID: {vid_id}, is_url={is_url}")
        if is_url:
            original_url = source_target
            def progress_hook(d):
                if d['status'] == 'downloading':
                    p = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_percent_str', '0.0%').strip())
                    speed = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_speed_str', '').strip())
                    update_status(vid_id, f"downloading ({p} | {speed})")
                elif d['status'] == 'finished':
                    update_status(vid_id, "extracting")

            common_opts = {
                "quiet": True, "noplaylist": True, "ffmpeg_location": FFMPEG_EXE,
                "progress_hooks": [progress_hook], "nocheckcertificate": True, "geo_bypass": True
            }

            ydl_opts_info = {**common_opts, "skip_download": True}
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                info = ydl.extract_info(original_url, download=False)
                title, thumbnail = info.get("title", original_url), info.get("thumbnail", "")
            
            with db() as con:
                con.execute("UPDATE transcriptions SET original=?, thumbnail=? WHERE id=?", (title, thumbnail, vid_id))

            text_subs = ""
            subs, auto_subs = info.get("subtitles", {}), info.get("automatic_captions", {})
            target_lang = "ru" if ("ru" in subs or "ru" in auto_subs) else ("en" if ("en" in subs or "en" in auto_subs) else None)

            if target_lang:
                ydl_opts_sub = {
                    **common_opts, "skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
                    "subtitleslangs": [target_lang], "subtitlesformat": "vtt", "outtmpl": str(UPLOAD_DIR / f"{vid_id}.%(ext)s")
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts_sub) as ydl: ydl.extract_info(original_url, download=True)
                    vtt_files = list(UPLOAD_DIR.glob(f"{vid_id}*.vtt"))
                    if vtt_files: 
                        text_subs = parse_vtt(vtt_files[0].read_text(encoding="utf-8"))
                        for vf in vtt_files: vf.unlink(missing_ok=True)
                except Exception: pass
            
            update_status(vid_id, "downloading (video...)")
            ydl_opts_media = {
                **common_opts, 
                "outtmpl": str(UPLOAD_DIR / f"{vid_id}.%(ext)s"), 
                "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best[height<=720]",
                "merge_output_format": "mp4"
            }
            with yt_dlp.YoutubeDL(ydl_opts_media) as ydl:
                info = ydl.extract_info(original_url, download=True)
                downloaded_file = ydl.prepare_filename(info)
                if not downloaded_file.endswith('.mp4'):
                    base_name = downloaded_file.rsplit('.', 1)[0]
                    if Path(base_name + '.mp4').exists():
                        downloaded_file = base_name + '.mp4'
                media_path = downloaded_file
            
            subprocess.run([FFMPEG_EXE, "-y", "-i", media_path, "-vn", "-acodec", "libmp3lame", temp_audio_path], check=True, capture_output=True)
            
            if text_subs:
                with db() as con:
                    con.execute("UPDATE transcriptions SET text=?, method=? WHERE id=?", (text_subs, f"subtitles ({target_lang})", vid_id))
        else:
            update_status(vid_id, "extracting")
            media_path = source_target
            
            thumb_path = UPLOAD_DIR / f"{vid_id}_thumb.jpg"
            try:
                subprocess.run([FFMPEG_EXE, "-y", "-ss", "00:00:01", "-i", media_path, "-vframes", "1", "-q:v", "2", str(thumb_path)], check=True, capture_output=True)
                thumb_url = f"/uploads/{vid_id}_thumb.jpg"
            except Exception: thumb_url = ""

            with db() as con:
                con.execute("UPDATE transcriptions SET thumbnail=? WHERE id=?", (thumb_url, vid_id))

            subprocess.run([FFMPEG_EXE, "-y", "-i", media_path, "-vn", "-acodec", "libmp3lame", temp_audio_path], check=True, capture_output=True)

        print("[WHISPER] Запуск распознавания речи...")
        update_status(vid_id, "transcribing")
        segments_gen, info = whisper_model.transcribe(temp_audio_path, beam_size=5, vad_filter=True, language="ru", initial_prompt="Транскрибация подкаста или видео. Используй правильную пунктуацию.")
        segments_list = list(segments_gen)
        text_whisper = " ".join([seg.text for seg in segments_list]).strip()

        with db() as con:
            row = con.execute("SELECT text, method FROM transcriptions WHERE id = ?", (vid_id,)).fetchone()
            existing_text, existing_method = (row["text"], row["method"]) if row else ("", "")

        if len(text_whisper) > 10: final_text, method = text_whisper, "whisper (small)"
        elif existing_text: final_text, method = existing_text, existing_method
        else: raise Exception("Не удалось распознать текст.")

        update_status(vid_id, "ready", text=final_text, method=method, audio_path=media_path)
        index_segments(vid_id, segments_list)
        log_stat("transcription")
        print(f"[SUCCESS] Задача {vid_id} успешно завершена!")

    except Exception as e:
        print(f"[ERROR] Ошибка в задаче {vid_id}: {str(e)}")
        update_status(vid_id, "error", error=str(e))
    finally:
        if Path(temp_audio_path).exists():
            try: Path(temp_audio_path).unlink(missing_ok=True)
            except Exception: pass

class UrlIn(BaseModel): url: str
class SearchIn(BaseModel): query: str; video_ids: list[str] = []

@app.post("/api/url")
async def from_url(payload: UrlIn, background_tasks: BackgroundTasks, current_user: dict = Depends(decode_jwt_token)):
    original_url = payload.url.strip()
    if not original_url: raise HTTPException(400, "Пустая ссылка")
    if "vimeo.com" in original_url: raise HTTPException(status_code=400, detail="Данный этап пока что в разработке.")

    cache_key = get_cache_key(original_url)
    with db() as con:
        if con.execute("SELECT id FROM transcriptions WHERE user_id = ? AND cache_key = ? LIMIT 1", (current_user["user_id"], cache_key)).fetchone():
            raise HTTPException(400, "Этот ролик уже добавлен в вашу историю!")
        existing_ready = con.execute("SELECT * FROM transcriptions WHERE cache_key = ? AND status = 'ready' LIMIT 1", (cache_key,)).fetchone()

    vid_id = uuid.uuid4().hex
    if existing_ready:
        unique_cache_key = f"{cache_key}_{current_user['user_id']}"
        with db() as con:
            con.execute("""INSERT INTO transcriptions (id, user_id, cache_key, source, original, thumbnail, text, summary, method, status, audio_path, created_at)
                VALUES (?, ?, ?, 'url', ?, ?, ?, ?, ?, 'ready', ?, ?)""",
                (vid_id, current_user["user_id"], unique_cache_key, existing_ready["original"], existing_ready["thumbnail"], existing_ready["text"], existing_ready["summary"], existing_ready["method"], existing_ready["audio_path"], datetime.now(timezone.utc).isoformat()))
        log_stat("url_submit")
        return {"id": vid_id, "status": "ready", "processing": False}

    with db() as con:
        try:
            con.execute("""INSERT INTO transcriptions (id, user_id, cache_key, source, original, status, created_at)
                VALUES (?, ?, ?, 'url', ?, 'queued', ?)""", (vid_id, current_user["user_id"], cache_key, original_url, datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError:
            row = con.execute("SELECT id, status FROM transcriptions WHERE cache_key = ? LIMIT 1", (cache_key,)).fetchone()
            if row: return {"id": row["id"], "status": row["status"], "processing": row["status"] != "ready"}
            raise
    
    log_stat("url_submit")
    background_tasks.add_task(process_media_task, vid_id, original_url, True)
    return {"id": vid_id, "status": "queued", "processing": True}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, current_user: dict = Depends(decode_jwt_token)):
    vid_id = uuid.uuid4().hex
    ext = Path(file.filename or "").suffix.lower()
    file_path = UPLOAD_DIR / f"{vid_id}{ext}"
    
    if ext not in {".mp4", ".webm", ".mp3", ".wav", ".m4a"}: raise HTTPException(400, "Неподдерживаемый формат")

    hasher = hashlib.sha256()
    with file_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            hasher.update(chunk); out.write(chunk)
    
    file_hash = hasher.hexdigest()
    cached_res = get_cached(file_hash)
    if cached_res:
        file_path.unlink(missing_ok=True)
        return cached_res

    with db() as con:
        con.execute("""INSERT INTO transcriptions (id, user_id, cache_key, source, original, status, created_at)
            VALUES (?, ?, ?, 'file', ?, 'queued', ?)""", (vid_id, current_user["user_id"], file_hash, file.filename, datetime.now(timezone.utc).isoformat()))

    log_stat("transcription")
    background_tasks.add_task(process_media_task, vid_id, str(file_path), False)
    return {"id": vid_id, "status": "queued", "processing": True}

@app.post("/api/search")
def semantic_search(payload: SearchIn):
    query = payload.query.strip()
    if not query: return []
    query_emb = embed_model.encode(["query: " + query])[0]

    with db() as con:
        v_ids = payload.video_ids
        if v_ids and "all" not in v_ids:
            placeholders = ','.join(['?'] * len(v_ids))
            query_sql = f"""SELECT c.chunk_text, c.start_sec, c.end_sec, c.embedding, t.original, t.id as video_id, t.source, t.audio_path 
                FROM chunks c JOIN transcriptions t ON c.video_id = t.id WHERE t.id IN ({placeholders})"""
            rows = con.execute(query_sql, v_ids).fetchall()
        else:
            rows = con.execute("""SELECT c.chunk_text, c.start_sec, c.end_sec, c.embedding, t.original, t.id as video_id, t.source, t.audio_path 
                FROM chunks c JOIN transcriptions t ON c.video_id = t.id""").fetchall()

    results = []
    for row in rows:
        sim = float(np.dot(query_emb, np.array(json.loads(row["embedding"]))) / (np.linalg.norm(query_emb) * np.linalg.norm(np.array(json.loads(row["embedding"])))))
        if sim < 0.75: # Я немного снизил порог до 75%, чтобы поиск находил больше совпадений
            continue

        start_sec = row["start_sec"] or 0
        
        # ВОТ ТУТ БЫЛА ОШИБКА, ИСПРАВЛЕНО:
        media_url = f"/uploads/{Path(row['audio_path']).name}" if row["audio_path"] else None
        
        watch_url = None
        if "youtube.com" in row["original"] or "youtu.be" in row["original"]:
            cache_k = get_cache_key(row["original"])
            if len(cache_k) == 11: watch_url = f"https://youtu.be/{cache_k}?t={int(start_sec)}"

        results.append({
            "text": row["chunk_text"], "video_title": row["original"], "video_id": row["video_id"],
            "score": sim, "start_sec": start_sec, "time_str": format_time(start_sec),
            "watch_url": watch_url, "media_url": media_url
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]

@app.post("/api/summary/{vid_id}")
def generate_summary(vid_id: str):
    with db() as con:
        row = con.execute("SELECT text, summary FROM transcriptions WHERE id = ?", (vid_id,)).fetchone()
        if not row: raise HTTPException(404, "Видео не найдено")
        if row["summary"] and not row["summary"].startswith("### Ошибка") and "Прежде чем приступить" not in row["summary"]: return {"summary": row["summary"]}
        full_text = row["text"]
        if not full_text or not full_text.strip(): raise HTTPException(400, "Текст пустой")

    try:
        headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + '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',
}

        json_data = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": f"Ты профессиональный академический помощник. Выдай готовый, подробный, структурированный конспект лекции по тексту. Обязательно выдели оглавление, ключевые разделы, подразделы и главные мысли на русском языке.\n\n{full_text[:12000]}"}]
        }
        res_json = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=json_data).json()
        if "error" in res_json: raise Exception(res_json["error"].get("message", "Ошибка API"))
        summary_text = res_json["choices"][0]["message"]["content"]
        with db() as con: con.execute("UPDATE transcriptions SET summary = ? WHERE id = ?", (summary_text, vid_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось получить ответ от Groq: {str(e)}")
    return {"summary": summary_text}

@app.delete("/api/videos/{vid_id}")
def delete_video(vid_id: str):
    with db() as con:
        row = con.execute("SELECT * FROM transcriptions WHERE id = ?", (vid_id,)).fetchone()
        if not row: raise HTTPException(404, "Запись не найдена")
        if row["source"] != "file": raise HTTPException(400, "Удалять можно только файлы, загруженные с ПК.")
        con.execute("DELETE FROM chunks WHERE video_id = ?", (vid_id,))
        con.execute("DELETE FROM transcriptions WHERE id = ?", (vid_id,))
    for file_to_del in UPLOAD_DIR.glob(f"{vid_id}*"):
        try: file_to_del.unlink(missing_ok=True)
        except Exception: pass
    return {"status": "deleted"}

@app.get("/api/status/{vid_id}")
def check_status(vid_id: str):
    with db() as con:
        row = con.execute("SELECT id, status, text, error, method, original FROM transcriptions WHERE id = ?", (vid_id,)).fetchone()
        if not row: raise HTTPException(404, "Запись не найдена")
        return dict(row)

@app.get("/api/videos")
def list_videos(current_user: dict = Depends(decode_jwt_token)):
    with db() as con:
        rows = con.execute("SELECT * FROM transcriptions WHERE user_id = ? ORDER BY created_at DESC", (current_user["user_id"],)).fetchall()
    
    res = []
    for r in rows:
        item = dict(r)
        if not item.get("thumbnail") and item.get("cache_key") and len(item["cache_key"]) == 11:
            item["thumbnail"] = f"https://img.youtube.com/vi/{item['cache_key']}/hqdefault.jpg"
        
        item["media_url"] = None
        
        # 1. Проверяем путь из базы
        if item.get("audio_path") and Path(item["audio_path"]).exists():
            item["media_url"] = f"/uploads/{Path(item['audio_path']).name}"
        else:
            # 2. Если по старому пути нет, ищем файл по ID в папке uploads (вдруг расширение сменилось)
            for ext in ['.mp4', '.webm', '.mkv', '.m4a', '.mp3']:
                p = UPLOAD_DIR / f"{item['id']}{ext}"
                if p.exists():
                    item["media_url"] = f"/uploads/{p.name}"
                    # Обновляем путь в базе, чтобы больше не искать
                    with db() as con:
                        con.execute("UPDATE transcriptions SET audio_path=? WHERE id=?", (str(p), item['id']))
                    break
                    
        res.append(item)
    return res