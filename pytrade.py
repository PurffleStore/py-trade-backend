# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import json
import datetime
from typing import List, Dict, Optional, Tuple, Callable

# Load .env file automatically for local development
from dotenv import load_dotenv
load_dotenv()

import pyodbc
import jwt  # PyJWT
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS
import pyodbc



# --- Your modules ---
from analysestock import analysestock
from list import (
    build_companies_payload,
    MARKETS,              # kept for backward compat (not mutated)
    get_markets,          # merged filters (MARKETS + extras)
    search_companies,     # global search helper
)
from assistant import get_answer
from signin import get_db_connection, ensure_user_table_exists  # <- NOTE: from db.py
import bcrypt  # pip install bcrypt
from typing import Tuple
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
from market_data import download as market_download, get_ohlcv as market_get_ohlcv

# ------------------------------------------------------------------------------
# App, ENV, CORS
# ------------------------------------------------------------------------------
app = Flask(__name__)

# IMPORTANT: keep this secret strong. You may move it to .env (JWT_SECRET)
app.config["JWT_SECRET"] = os.environ.get(
    "JWT_SECRET",
    "96c63da06374c1bde332516f3acbd23c84f35f90d8a6321a25d790a0a451af32"
)

# Token lifetimes (override in .env if you like)
ACCESS_MINUTES  = int(os.environ.get("ACCESS_MINUTES", "15"))   # 15 minutes
REFRESH_DAYS    = int(os.environ.get("REFRESH_DAYS", "7"))      # 7 days

# CORS — allow all origins so any frontend domain (Hostinger, HF Space, localhost) works.
# Credentials (cookies) are sent via Authorization header instead of cookies in production.
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    expose_headers=["Authorization"],
)

# Ensure table exists at startup — wrapped so a missing DB never crashes the app
try:
    ensure_user_table_exists()
except Exception as _db_err:
    import logging
    logging.warning("DB init skipped at startup (will retry on first request): %s", _db_err)

# register AI TA routes blueprint
try:
    from ai_ta_routes import bp as ai_ta_bp
    app.register_blueprint(ai_ta_bp)
except Exception as ex:
    # Log the import/registration error so it's visible in logs and provide a minimal
    # fallback endpoint so the frontend preflight (OPTIONS) and GET won't return 404.
    app.logger.exception('Failed to register ai_ta blueprint: %s', ex)

    @app.route('/ai_ta', methods=['GET', 'OPTIONS'])
    def ai_ta_fallback():
        # Respond to CORS preflight immediately
        if request.method == 'OPTIONS':
            return ('', 200)
        symbol = (request.args.get('symbol') or '').strip()
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400
        try:
            # Try to lazy-import the predictor; if that fails, return a clear 503
            from ai_ta_model import predict_score_for_symbol
            # Call with default model path (may return heuristic if model missing)
            res = predict_score_for_symbol(symbol)
            return jsonify(res), 200
        except Exception as e:
            app.logger.exception('ai_ta_fallback failed: %s', e)
            return jsonify({'error': 'AI TA unavailable', 'detail': str(e)}), 503

# ------------------------------------------------------------------------------
# Helpers: JWT
# ------------------------------------------------------------------------------
def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)

def _encode_jwt(payload: dict, minutes: Optional[int] = None, days: Optional[int] = None) -> str:
    iat = _now_utc()
    exp = iat
    if minutes:
        exp = iat + datetime.timedelta(minutes=minutes)
    if days:
        exp = iat + datetime.timedelta(days=days)
    to_encode = {
        "iat": int(iat.timestamp()),
        "exp": int(exp.timestamp()),
        **payload,
    }
    return jwt.encode(to_encode, app.config["JWT_SECRET"], algorithm="HS256")

def create_access_token(user_id: int, email: str, name: Optional[str] = None) -> str:
    payload = {"sub": str(user_id), "email": email, "type": "access"}
    if name:
        payload["name"] = name  # optional convenience for UI
    return _encode_jwt(payload, minutes=ACCESS_MINUTES)

def create_refresh_token(user_id: int, email: str) -> str:
    return _encode_jwt({"sub": str(user_id), "email": email, "type": "refresh"}, days=REFRESH_DAYS)

def _decode_jwt(token: str) -> dict:
    return jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])

def set_token_cookies(resp, access_token: str, refresh_token: str):
    """
    Sets HttpOnly cookies. For local dev we use SameSite=None + Secure=False if http.
    In production, set Secure=True and run on HTTPS.
    """
    secure = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
    # Access cookie ~ session/short-lived
    resp.set_cookie(
        "access_token",
        access_token,
        httponly=True,
        secure=secure,
        samesite="None" if secure else "Lax",
        max_age=ACCESS_MINUTES * 60,
        path="/",
    )
    # Refresh cookie ~ longer-lived
    resp.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        secure=secure,
        samesite="None" if secure else "Lax",
        max_age=REFRESH_DAYS * 24 * 3600,
        path="/refresh",  # scope refresh cookie to the refresh endpoint
    )

def clear_token_cookies(resp):
    resp.delete_cookie("access_token", path="/")
    resp.delete_cookie("refresh_token", path="/refresh")

def extract_bearer_token() -> Optional[str]:
    """
    Reads 'Authorization: Bearer <token>' header if present.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None

def get_current_user_from_token(require_access: bool = True) -> Optional[dict]:
    """
    Validates either:
      - Bearer access token (preferred), or
      - access_token cookie (fallback).
    Returns dict {userId, email} or None on failure.
    """
    token = extract_bearer_token()
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    try:
        decoded = _decode_jwt(token)
        if require_access and decoded.get("type") != "access":
            return None
        user_id = decoded.get("sub")
        email = decoded.get("email")
        if not user_id or not email:
            return None
        return {"userId": int(user_id), "email": email}
    except Exception:
        return None

def jwt_required(func: Callable) -> Callable:
    """
    Decorator to protect routes with access token.
    Accepts either Bearer access token or access_token cookie.
    """
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        user = get_current_user_from_token(require_access=True)
        if not user:
            return jsonify({"message": "Unauthorised"}), 401
        # attach to request context if needed
        request.user = user  # type: ignore
        return func(*args, **kwargs)
    return wrapper


def _is_bcrypt_hash(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(("$2a$", "$2b$", "$2y$"))

def _safe_check_password(stored_hash: str, password: str) -> Tuple[bool, str]:
    """
    Tries Werkzeug hash first; if it fails, tries bcrypt; then a last-resort
    plain-text compare (for legacy/dev data only).
    Returns (ok, scheme) where scheme is one of "werkzeug", "bcrypt", "plain", "unknown".
    """
    # 1) Werkzeug format (pbkdf2:sha256:...)
    try:
        if check_password_hash(stored_hash, password):
            return True, "werkzeug"
    except Exception:
        pass

    # 2) bcrypt format ($2a/$2b/$2y)
    if _is_bcrypt_hash(stored_hash):
        try:
            if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                return True, "bcrypt"
        except Exception:
            pass

    # 3) plain text (legacy/dev only; not recommended)
    try:
        if stored_hash == password:
            return True, "plain"
    except Exception:
        pass

    return False, "unknown"

def _rehash_and_update_if_needed(user_id: int, email: str, scheme: str, conn) -> None:
    """
    If the user logged in with a legacy hash (bcrypt or plain), immediately
    rehash to the canonical Werkzeug PBKDF2 format and update the DB.
    """
    # Only migrate non-Werkzeug schemes
    if scheme in ("bcrypt", "plain"):
        new_hash = generate_password_hash(email + ":", method="pbkdf2:sha256", salt_length=16)  # temp
        # Above dummy line only to get the *format*; we now re-create with the actual password in _do_signin
        # This function just exists for structure; actual rehash done in _do_signin (see below).
        pass

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}, 200

# ------------------------------------------------------------------------------
# Public APIs
# ------------------------------------------------------------------------------
@app.get("/getfilters")
def get_filters():
    """
    Returns UI filter tree.
    Uses get_markets() so you get MARKETS + extra markets (NASDAQ-100, DAX-40, OMXS-30)
    without changing your original MARKETS object.
    """
    return jsonify({
        "asOf": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "markets": get_markets()
    })

@app.get("/getcompanies")
def get_companies():
    """
    /getcompanies?code=NIFTY50 (or ?index=...)
    """
    code = (request.args.get("code") or request.args.get("index") or "").upper()
    if not code:
        return jsonify({"error": "Missing ?code=<INDEXCODE>"}), 400
    try:
        payload = build_companies_payload(code)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/search_companies")
def route_search_companies():
    """
    /search_companies?q=INFY&indices=NIFTY50,NIFTY100&limit=50
    - q: search term (symbol or company substring)
    - indices: optional CSV of index codes; defaults to all supported
    - limit: 1..200 (default 50)
    """
    q = request.args.get("q", "")
    indices_csv = request.args.get("indices", "")
    limit_raw = request.args.get("limit", "50")

    try:
        limit_i = max(1, min(200, int(limit_raw)))
    except Exception:
        limit_i = 50

    indices = None
    if indices_csv.strip():
        indices = [c.strip().upper() for c in indices_csv.split(",") if c.strip()]

    try:
        results = search_companies(q, indices=indices, limit=limit_i)
        return jsonify({
            "query": q,
            "count": len(results),
            "results": results
        })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

@app.route('/analysestock', methods=['POST'])
def analyze_all():
    """
    Analyse stocks for the given ticker symbols.
    """
    try:
        data = request.get_json(force=True)
        tickersSymbol = data['ticker']
        results = []
        for ticker in tickersSymbol:
            try:
                results.append(analysestock(ticker))
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})
        return Response(json.dumps(results, indent=2), mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        reply = get_answer(user_message)
        return jsonify({"answer": reply})
    except Exception as e:
        app.logger.exception("chat error: %s", e)
        return jsonify({
            "answer": "I encountered an error while processing your question. Please try again.",
            "error": str(e)
        }), 200  # return 200 so frontend shows the message instead of error state

# ------------------------------------------------------------------------------
# Auth: Sign-up / Sign-in with JWT (no Google)
# ------------------------------------------------------------------------------
@app.route('/signup', methods=['POST'])
def sign_up():
    data = request.json or {}
    name = (data.get('name') or "").strip()
    phone = (data.get('phone') or "").strip()
    email = (data.get('email') or "").strip().lower()
    password = data.get('password') or ""

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM Users WHERE email = ?', (email,))
        if cursor.fetchone():
            return jsonify({"message": "Email already in use"}), 400

        # create Werkzeug PBKDF2 hash explicitly
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)

        cursor.execute(
            'INSERT INTO Users (name, phone, email, password) VALUES (?, ?, ?, ?)',
            (name, phone, email, hashed_password)
        )
        conn.commit()
        return jsonify({"message": "User created successfully"}), 201
    finally:
        try: cursor.close()
        except: pass
        conn.close()

def _do_signin(email: str, password: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, email, password FROM Users WHERE email = ?', (email,))
        row = cursor.fetchone()
        if not row:
            return None

        user_id, name, email_db, stored_pw = row[0], row[1], row[2], row[3] or ""

        ok, scheme = _safe_check_password(stored_pw, password)
        if not ok:
            return None

        # If the stored hash was legacy (bcrypt/plain), migrate it now
        if scheme != "werkzeug":
            new_hash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)
            try:
                cursor.execute('UPDATE Users SET password = ? WHERE id = ?', (new_hash, user_id))
                conn.commit()
            except Exception:
                # If migration fails, still allow login this time
                pass

        return {"userId": int(user_id), "name": name, "email": email_db}
    finally:
        try:
            cursor.close()
        except:
            pass
        conn.close()


@app.route('/signin', methods=['POST'])
def sign_in():
    """
    Sign in with email + password.
    Returns access & refresh tokens in both:
      - HttpOnly cookies (recommended), and
      - JSON body (for SPA storage if you choose).
    """
    data = request.json or {}
    email = (data.get('email') or "").strip().lower()
    password = data.get('password') or ""

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    user = _do_signin(email, password)
    if not user:
        return jsonify({"message": "Invalid email or password"}), 401

    access_token = create_access_token(user_id=user["userId"], email=user["email"], name=user["name"])
    refresh_token = create_refresh_token(user_id=user["userId"], email=user["email"])

    payload = {
        "message": "Signed in successfully",
        "userId": user["userId"],
        "name": user["name"],
        "email": user["email"],
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "accessTokenExpiresIn": ACCESS_MINUTES * 60,
        "refreshTokenExpiresIn": REFRESH_DAYS * 24 * 3600,
    }
    resp = make_response(jsonify(payload), 200)
    set_token_cookies(resp, access_token, refresh_token)
    resp.headers["Authorization"] = f"Bearer {access_token}"  # optional
    return resp

@app.post("/refresh")
def refresh():
    """
    Rotate access token using refresh_token cookie (HttpOnly).
    """
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        return jsonify({"message": "Refresh token missing"}), 401

    try:
        decoded = _decode_jwt(refresh_token)
        if decoded.get("type") != "refresh":
            return jsonify({"message": "Invalid token type"}), 401
        user_id = int(decoded.get("sub"))
        email = decoded.get("email")
        if not user_id or not email:
            return jsonify({"message": "Invalid token"}), 401

        new_access = create_access_token(user_id=user_id, email=email)
        payload = {"accessToken": new_access, "accessTokenExpiresIn": ACCESS_MINUTES * 60}
        resp = make_response(jsonify(payload), 200)
        # update access_token cookie only
        secure = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
        resp.set_cookie(
            "access_token",
            new_access,
            httponly=True,
            secure=secure,
            samesite="None" if secure else "Lax",
            max_age=ACCESS_MINUTES * 60,
            path="/",
        )
        resp.headers["Authorization"] = f"Bearer {new_access}"  # optional
        return resp
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Refresh token expired"}), 401
    except Exception:
        return jsonify({"message": "Invalid refresh token"}), 401

@app.get("/me")
@jwt_required
def me():
    """
    Returns the current user's profile if the access token is valid.
    """
    user = getattr(request, "user", None)
    return jsonify({"userId": user["userId"], "email": user["email"]})

@app.post("/logout")
def logout():
    """
    Clears auth cookies. The client should also forget any in-memory tokens.
    """
    resp = make_response(jsonify({"message": "Logged out"}), 200)
    clear_token_cookies(resp)
    return resp


#community forum to post the data

# --- Add this API anywhere below other routes ---
@app.post("/posts")
@jwt_required
def create_community_post():
    """
    TEMP: Open endpoint. Expects JSON:
    { userId?, userName?, title?, category?, tags?, body }
    """
    data = request.get_json(silent=True) or {}

    user_id = int((data.get("userId") or 0))
    user_name = (data.get("userName") or "").strip() or "Guest"
    title = (data.get("title") or "").strip()
    category = (data.get("category") or "").strip()
    tags = (data.get("tags") or "").strip()
    body = (data.get("body") or "").strip()

    if not body:
        return jsonify({"message": "body is required"}), 400

    conn = get_db_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        # Return the new identity in the same statement (more reliable than SCOPE_IDENTITY())
        cursor.execute("""
            INSERT INTO Community (user_id, user_name, title, category, tags, body)
            OUTPUT INSERTED.id
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, user_name, title, category, tags, body))

        row = cursor.fetchone()
        conn.commit()

        if not row or row[0] is None:
            return jsonify({"error": "Failed to retrieve new post id"}), 500

        new_id = int(row[0]);

        return jsonify({
            "id": new_id,
            "message": "Post created",
            "userId": user_id,
            "userName": user_name
        }), 201
    except Exception as e:
        app.logger.exception("create_community_post failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cursor: cursor.close()
        except:
            pass
        conn.close()


@app.get("/posts")
def list_community_posts():
    """
    List community posts (public). Supports paging:
      GET /posts?limit=50&offset=0
    Returns: { total, offset, limit, count, results: Post[] }
    """
    limit_raw = request.args.get("limit", "50")
    offset_raw = request.args.get("offset", "0")
    try:
        limit = max(1, min(200, int(limit_raw)))
    except Exception:
        limit = 50
    try:
        offset = max(0, int(offset_raw))
    except Exception:
        offset = 0

    conn = get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        # data page
        cur.execute("""
            SELECT
                id,
                user_id      AS userId,
                user_name    AS userName,
                title,
                category,
                tags,
                body,
                created_at   AS createdAt
            FROM Community
            ORDER BY created_at DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
        """, (offset, limit))
        rows = cur.fetchall()

        posts = [
            {
                "id": int(r[0]),
                "userId": int(r[1]),
                "userName": r[2],
                "title": r[3],
                "category": r[4],
                "tags": r[5],
                "body": r[6],
                "createdAt": r[7].isoformat() if hasattr(r[7], "isoformat") else r[7],
            }
            for r in rows
        ]

        # total count
        cur.execute("SELECT COUNT(*) FROM Community")
        total = int(cur.fetchone()[0])

        return jsonify({
            "total": total,
            "offset": offset,
            "limit": limit,
            "count": len(posts),
            "results": posts
        }), 200
    except Exception as e:
        app.logger.exception("list_community_posts failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cur: cur.close()
        except:
            pass
        conn.close()


# --- Community: like a post ---
@app.post("/posts/<int:post_id>/like")
def like_post(post_id: int):
    conn = get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("UPDATE Community SET like_count = ISNULL(like_count,0) + 1 WHERE id = ?", (post_id,))
        conn.commit()
        cur.execute("SELECT ISNULL(like_count,0) FROM Community WHERE id = ?", (post_id,))
        row = cur.fetchone()
        return jsonify({"likeCount": int(row[0]) if row else 0}), 200
    except Exception as e:
        app.logger.exception("like_post failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cur: cur.close()
        except: pass
        conn.close()


# --- Community: dislike a post ---
@app.post("/posts/<int:post_id>/dislike")
def dislike_post(post_id: int):
    conn = get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("UPDATE Community SET dislike_count = ISNULL(dislike_count,0) + 1 WHERE id = ?", (post_id,))
        conn.commit()
        cur.execute("SELECT ISNULL(dislike_count,0) FROM Community WHERE id = ?", (post_id,))
        row = cur.fetchone()
        return jsonify({"dislikeCount": int(row[0]) if row else 0}), 200
    except Exception as e:
        app.logger.exception("dislike_post failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cur: cur.close()
        except: pass
        conn.close()


# --- Community: add comment to a post ---
@app.post("/posts/<int:post_id>/comments")
@jwt_required
def add_comment(post_id: int):
    data = request.get_json(silent=True) or {}
    body = (data.get("body") or "").strip()
    if not body:
        return jsonify({"error": "body is required"}), 400

    # derive author from JWT
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user_name = "Guest"
    try:
        import jwt as pyjwt
        payload = pyjwt.decode(token, options={"verify_signature": False})
        user_name = (payload.get("name") or payload.get("preferred_username") or
                     " ".join(filter(None, [payload.get("given_name"), payload.get("family_name")])) or "Guest")
    except Exception:
        pass

    conn = get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        # ensure CommunityComments table exists
        cur.execute("""
            IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME='CommunityComments')
            CREATE TABLE CommunityComments (
                id         INT IDENTITY PRIMARY KEY,
                post_id    INT NOT NULL,
                user_name  NVARCHAR(120),
                body       NVARCHAR(MAX),
                created_at DATETIME2 DEFAULT GETDATE()
            )
        """)
        cur.execute("""
            INSERT INTO CommunityComments (post_id, user_name, body)
            OUTPUT INSERTED.id
            VALUES (?, ?, ?)
        """, (post_id, user_name, body))
        row = cur.fetchone()
        cur.execute("UPDATE Community SET comment_count = ISNULL(comment_count,0) + 1 WHERE id = ?", (post_id,))
        conn.commit()
        return jsonify({"id": int(row[0]) if row else 0, "message": "Comment added"}), 201
    except Exception as e:
        app.logger.exception("add_comment failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cur: cur.close()
        except: pass
        conn.close()


# --- Community: get comments for a post ---
@app.get("/posts/<int:post_id>/comments")
def get_comments(post_id: int):
    conn = get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("""
            IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME='CommunityComments')
            CREATE TABLE CommunityComments (
                id         INT IDENTITY PRIMARY KEY,
                post_id    INT NOT NULL,
                user_name  NVARCHAR(120),
                body       NVARCHAR(MAX),
                created_at DATETIME2 DEFAULT GETDATE()
            )
        """)
        cur.execute("""
            SELECT id, user_name, body, created_at
            FROM CommunityComments WHERE post_id = ?
            ORDER BY created_at ASC
        """, (post_id,))
        rows = cur.fetchall()
        comments = [
            {"id": int(r[0]), "userName": r[1], "body": r[2],
             "createdAt": r[3].isoformat() if hasattr(r[3], "isoformat") else str(r[3])}
            for r in rows
        ]
        return jsonify({"postId": post_id, "comments": comments}), 200
    except Exception as e:
        app.logger.exception("get_comments failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cur: cur.close()
        except: pass
        conn.close()


# --- Community: stats ---
@app.get("/community/stats")
def community_stats():
    conn = get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Community")
        total_posts = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(DISTINCT user_name) FROM Community")
        total_members = int(cur.fetchone()[0])
        cur.execute("""
            SELECT COUNT(*) FROM Community
            WHERE created_at >= CAST(GETDATE() AS DATE)
        """)
        today_posts = int(cur.fetchone()[0])
        cur.execute("""
            SELECT TOP 5 category, COUNT(*) AS cnt
            FROM Community WHERE category IS NOT NULL AND category <> ''
            GROUP BY category ORDER BY cnt DESC
        """)
        top_cats = [{"category": r[0], "count": int(r[1])} for r in cur.fetchall()]
        cur.execute("""
            SELECT TOP 5 tags FROM Community
            WHERE tags IS NOT NULL AND tags <> ''
            ORDER BY created_at DESC
        """)
        tag_rows = cur.fetchall()
        all_tags: list[str] = []
        for rw in tag_rows:
            for t in str(rw[0]).split(","):
                t = t.strip().lstrip("#")
                if t and t not in all_tags:
                    all_tags.append(t)
        return jsonify({
            "totalPosts": total_posts,
            "totalMembers": total_members,
            "todayPosts": today_posts,
            "topCategories": top_cats,
            "popularTags": all_tags[:10],
        }), 200
    except Exception as e:
        app.logger.exception("community_stats failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cur: cur.close()
        except: pass
        conn.close()


# ------------------------------------------------------------------------------
# Market overview values (uses yfinance)
# ------------------------------------------------------------------------------
@app.get("/getmarketcards")
def get_market_cards():
    """
    Returns a list of market overview items grouped by category with live prices.
    Each item: { title, symbol, category, price, chg, chgPct }
    """
    # category -> list of (title, yfinance_symbol)
    ticker_groups = {
        'Indices': [
            ('S&P 500',     '^GSPC'),
            ('NASDAQ',      '^IXIC'),
            ('Dow Jones',   '^DJI'),
            ('FTSE 100',    '^FTSE'),
            ('DAX',         '^GDAXI'),
            ('CAC 40',      '^FCHI'),
            ('Nikkei 225',  '^N225'),
            ('Hang Seng',   '^HSI'),
            ('ASX 200',     '^AXJO'),
            ('Nifty 50',    '^NSEI'),
            ('SENSEX',      '^BSESN'),
        ],
        'Commodities': [
            ('Gold',              'GC=F'),
            ('Silver',            'SI=F'),
            ('Crude Oil (Brent)', 'BZ=F'),
            ('Crude Oil (WTI)',   'CL=F'),
            ('Natural Gas',       'NG=F'),
            ('Copper',            'HG=F'),
            ('Platinum',          'PL=F'),
            ('Palladium',         'PA=F'),
            ('Wheat',             'ZW=F'),
            ('Corn',              'ZC=F'),
        ],
        'Forex': [
            ('USD/INR',  'INR=X'),
            ('EUR/USD',  'EURUSD=X'),
            ('GBP/USD',  'GBPUSD=X'),
            ('USD/JPY',  'JPY=X'),
            ('USD/CHF',  'CHF=X'),
            ('AUD/USD',  'AUDUSD=X'),
            ('USD/CAD',  'CAD=X'),
            ('USD/CNY',  'CNY=X'),
            ('EUR/GBP',  'EURGBP=X'),
            ('EUR/JPY',  'EURJPY=X'),
        ],
        'Crypto': [
            ('Bitcoin',   'BTC-USD'),
            ('Ethereum',  'ETH-USD'),
            ('BNB',       'BNB-USD'),
            ('Solana',    'SOL-USD'),
            ('XRP',       'XRP-USD'),
            ('Cardano',   'ADA-USD'),
            ('Dogecoin',  'DOGE-USD'),
            ('Polygon',   'MATIC-USD'),
            ('Avalanche', 'AVAX-USD'),
            ('Chainlink', 'LINK-USD'),
        ],
        'Bonds': [
            ('US 10Y Yield',   '^TNX'),
            ('US 2Y Yield',    '^IRX'),
            ('US 30Y Yield',   '^TYX'),
            ('Germany 10Y',    '^IRDE'),
            ('UK 10Y Gilt',    '^TNX'),
        ],
    }

    def fetch_quote(symbol):
        try:
            t = yf.Ticker(symbol)
            hist = None
            try:
                hist = t.history(period="2d")
            except Exception:
                hist = None
            if hist is not None and hasattr(hist, 'empty') and not hist.empty:
                closes = list(hist['Close'].values)
                last = float(closes[-1])
                prev = float(closes[-2]) if len(closes) >= 2 else last
                chg = last - prev
                chg_pct = (chg / prev * 100) if prev != 0 else 0.0
                return last, chg, chg_pct
            # fallback to ticker.info
            info = {}
            try:
                info = t.info or {}
            except Exception:
                info = {}
            last = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
            prev = info.get('previousClose')
            if last is not None:
                price = float(last)
                chg = float(last) - float(prev) if prev else 0.0
                chg_pct = (chg / float(prev) * 100) if prev and float(prev) != 0 else 0.0
                return price, chg, chg_pct
        except Exception:
            app.logger.exception('getmarketcards: failed for %s', symbol)
        return None, None, None

    out = []
    for category, items in ticker_groups.items():
        for title, symbol in items:
            price, chg, chg_pct = fetch_quote(symbol)
            out.append({
                'title': title,
                'symbol': symbol,
                'category': category,
                'price': price,
                'chg': chg,
                'chgPct': chg_pct,
            })

    return jsonify(out), 200


# ------------------------------------------------------------------------------
# Live batch quotes for individual stocks  (NEW)
# ------------------------------------------------------------------------------
@app.get("/getquotes")
def get_quotes():
    """
    GET /getquotes?tickers=RELIANCE.NS,TCS.NS,INFY.NS
    Returns live price, change and day high/low for each ticker via yfinance.
    """
    raw = request.args.get('tickers', '')
    if not raw:
        return jsonify({'error': 'Missing ?tickers= parameter'}), 400

    symbols = [s.strip().upper() for s in raw.split(',') if s.strip()]
    if not symbols:
        return jsonify([]), 200

    out = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = None
            try:
                hist = t.history(period='2d')
            except Exception:
                hist = None

            price = chg = chg_pct = high = low = None

            if hist is not None and hasattr(hist, 'empty') and not hist.empty:
                closes = list(hist['Close'].values)
                highs  = list(hist['High'].values)
                lows   = list(hist['Low'].values)
                last = float(closes[-1])
                prev = float(closes[-2]) if len(closes) >= 2 else last
                price   = last
                chg     = last - prev
                chg_pct = (chg / prev * 100) if prev != 0 else 0.0
                high    = float(highs[-1])
                low     = float(lows[-1])
            else:
                info = {}
                try:
                    info = t.fast_info or {}
                    price   = float(info.get('last_price') or info.get('regularMarketPrice') or 0) or None
                    high    = float(info.get('day_high')   or 0) or None
                    low     = float(info.get('day_low')    or 0) or None
                    prev    = float(info.get('previous_close') or price or 0)
                    chg     = (price - prev) if price and prev else None
                    chg_pct = (chg / prev * 100) if chg and prev and prev != 0 else None
                except Exception:
                    pass

            out.append({
                'symbol':   sym,
                'price':    price,
                'chg':      chg,
                'chgPct':   chg_pct,
                'high':     high,
                'low':      low,
            })
        except Exception:
            app.logger.exception('getquotes: failed for %s', sym)
            out.append({'symbol': sym, 'price': None, 'chg': None, 'chgPct': None, 'high': None, 'low': None})

    return jsonify(out), 200


# ------------------------------------------------------------------------------
# Intraday price series for chart  (NEW)
# ------------------------------------------------------------------------------
@app.get("/getintraday")
def get_intraday():
    """
    GET /getintraday?symbol=RELIANCE.NS&range=1d&interval=1m
    Returns close prices with ISO timestamps for intraday area chart.
    range:    1d | 5d | 1mo
    interval: 1m | 5m | 15m | 1h
    """
    symbol   = request.args.get('symbol', '').strip()
    range_p  = request.args.get('range',    '1d')
    interval = request.args.get('interval', '5m')

    if not symbol:
        return jsonify({'error': 'Missing ?symbol= parameter'}), 400

    # safe interval/range combinations yfinance supports
    safe = {
        '1d':  ['1m', '2m', '5m', '15m', '30m', '60m', '90m'],
        '5d':  ['5m', '15m', '30m', '60m'],
        '1mo': ['30m', '60m', '90m', '1d'],
    }
    if range_p not in safe:
        range_p = '1d'
    if interval not in safe[range_p]:
        interval = safe[range_p][0]

    try:
        t = yf.Ticker(symbol.upper())
        hist = t.history(period=range_p, interval=interval)
        if hist is None or hist.empty:
            return jsonify({'symbol': symbol, 'timestamps': [], 'closes': []}), 200

        import pandas as pd
        timestamps = []
        for idx in hist.index:
            try:
                ts = pd.Timestamp(idx)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert('UTC').tz_localize(None)
                timestamps.append(ts.strftime('%Y-%m-%dT%H:%M'))
            except Exception:
                timestamps.append(str(idx))

        closes = [float(v) for v in hist['Close'].values]
        highs  = [float(v) for v in hist['High'].values]
        lows   = [float(v) for v in hist['Low'].values]

        return jsonify({
            'symbol':     symbol.upper(),
            'range':      range_p,
            'interval':   interval,
            'timestamps': timestamps,
            'closes':     closes,
            'highs':      highs,
            'lows':       lows,
        }), 200

    except Exception:
        app.logger.exception('getintraday: failed for %s', symbol)
        return jsonify({'symbol': symbol, 'timestamps': [], 'closes': []}), 200


# ------------------------------------------------------------------------------
# Global indices (uses yfinance)
# ------------------------------------------------------------------------------
@app.get("/getglobalindices")
def get_global_indices():
    """
    Returns a flat list of global indices for a set of countries. Each item:
    { id, name, country, region, price, change, changePct, sparkline }
    """
    # Define representative indices per country with yfinance symbols
    indices_map = {
        'India': [
            ('Nifty 50',         '^NSEI'),
            ('SENSEX',           '^BSESN'),
            ('Nifty Bank',       '^NSEBANK'),
            ('Nifty IT',         None),
            ('Nifty Midcap 100', None),
            ('Nifty Smallcap',   None),
        ],
        'United States': [
            ('S&P 500',           '^GSPC'),
            ('Dow Jones',         '^DJI'),
            ('Nasdaq Composite',  '^IXIC'),
            ('Russell 2000',      '^RUT'),
            ('S&P MidCap 400',    '^MID'),
            ('VIX',               '^VIX'),
        ],
        'United Kingdom': [
            ('FTSE 100',  '^FTSE'),
            ('FTSE 250',  '^FTMC'),
        ],
        'Germany': [
            ('DAX',    '^GDAXI'),
            ('MDAX',   '^MDAXI'),
            ('TecDAX', '^TECDAX'),
        ],
        'France': [
            ('CAC 40', '^FCHI'),
            ('SBF 120', None),
        ],
        'Japan': [
            ('Nikkei 225',  '^N225'),
            ('TOPIX',       '^TOPX'),
        ],
        'China': [
            ('Shanghai Composite', '000001.SS'),
            ('CSI 300',            '000300.SS'),
            ('Shenzhen',           '399001.SZ'),
        ],
        'Hong Kong': [
            ('Hang Seng',      '^HSI'),
            ('Hang Seng Tech', '^HSTECH'),
        ],
        'Australia': [
            ('ASX 200',    '^AXJO'),
            ('All Ords',   '^AORD'),
        ],
        'Canada': [
            ('TSX Composite', '^GSPTSE'),
            ('TSX 60',        '^TX60'),
        ],
        'Brazil': [
            ('Bovespa', '^BVSP'),
        ],
        'Sweden': [
            ('OMX Stockholm 30', '^OMXSPI'),
        ],
        'Russia': [
            ('MOEX Russia', 'IMOEX.ME'),
            ('RTS Index',   'RTSI.ME'),
        ],
    }

    def build_spark_from_hist(close_vals, pts=26):
        # normalize to length pts by sampling or padding
        arr = []
        if not close_vals:
            # synthetic series
            base = 100
            from random import random
            arr = [max(1, base + (random() - 0.5) * base * 0.02) for _ in range(pts)]
            return arr
        vals = list(close_vals)
        # if too short, pad with last
        while len(vals) < pts:
            vals.insert(0, vals[0])
        if len(vals) > pts:
            # sample evenly
            step = len(vals) / pts
            sampled = [vals[int(i * step)] for i in range(pts)]
            return [float(v) for v in sampled]
        return [float(v) for v in vals]

    out = []
    for country, items in indices_map.items():
        for name, symbol in items:
            price = None
            change = None
            change_pct = None
            spark = []
            try:
                if symbol:
                    t = yf.Ticker(symbol)
                    hist = None
                    try:
                        hist = t.history(period='30d')
                    except Exception:
                        hist = None
                    if hist is not None and hasattr(hist, 'empty') and not hist.empty:
                        closes = list(hist['Close'].values)
                        # create spark from most recent values
                        spark = build_spark_from_hist(closes[-26:], pts=26)
                        last = float(closes[-1])
                        prev = float(closes[-2]) if len(closes) >= 2 else last
                        price = last
                        change = last - prev
                        change_pct = (change / prev * 100) if prev != 0 else 0.0
                    else:
                        # try single-value info
                        info = {}
                        try:
                            info = t.info or {}
                        except Exception:
                            info = {}
                        last = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
                        prev = info.get('previousClose')
                        if last is not None:
                            try:
                                price = float(last)
                            except Exception:
                                price = None
                        if price is not None and prev is not None:
                            try:
                                p = float(prev)
                                change = float(price) - p
                                if p != 0:
                                    change_pct = (change / p) * 100
                                else:
                                    change_pct = 0.0
                            except Exception:
                                change_pct = None
            except Exception:
                app.logger.exception("get_global_indices failed for %s", name)
            out.append({
                "id": name,
                "name": name,
                "country": country,
                "region": None,  # region???
                "price": price,
                "change": change,
                "changePct": change_pct,
                "sparkline": spark,
            })

    return jsonify(out), 200

if __name__ == "__main__":
    # When invoked directly, start the Flask development server.
    # Use environment variables to control port and debug mode.
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting pytrade backend on {host}:{port} (debug={debug})")
    # Avoid using the reloader when running inside some environments unless debug is explicitly set
    app.run(host=host, port=port, debug=debug, use_reloader=debug)
