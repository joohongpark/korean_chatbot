"""SQLite 영속 저장 모듈.

chatbot.db 파일에 users / conversations / turns 테이블을 관리한다.
모든 공개 함수는 threading.Lock()으로 보호된다.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def init_db(db_path: str | Path) -> None:
    """DB 초기화: 테이블 생성 및 WAL 모드 활성화."""
    global _conn
    _conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                username    TEXT UNIQUE NOT NULL,
                created_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT '새 대화',
                task_topic  TEXT DEFAULT '',
                created_at  REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS turns (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id     TEXT NOT NULL,
                role                TEXT NOT NULL,
                learner_text        TEXT,
                rag_examples        TEXT,
                text                TEXT NOT NULL,
                created_at          REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """)
        _conn.commit()


def _db() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("DB not initialized. Call init_db() first.")
    return _conn


# ---------------------------------------------------------------------------
# 사용자
# ---------------------------------------------------------------------------

def get_or_create_user(username: str) -> str:
    """username으로 user_id(UUID) 반환. 없으면 생성."""
    with _lock:
        row = _db().execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row:
            return row["id"]
        uid = str(uuid.uuid4())
        _db().execute(
            "INSERT INTO users (id, username, created_at) VALUES (?, ?, ?)",
            (uid, username, time.time()),
        )
        _db().commit()
        return uid


# ---------------------------------------------------------------------------
# 대화
# ---------------------------------------------------------------------------

def create_conversation(user_id: str, task_topic: str = "") -> str:
    """새 대화를 생성하고 conversation_id(UUID) 반환."""
    cid = str(uuid.uuid4())
    with _lock:
        _db().execute(
            "INSERT INTO conversations (id, user_id, title, task_topic, created_at) VALUES (?, ?, ?, ?, ?)",
            (cid, user_id, "새 대화", task_topic, time.time()),
        )
        _db().commit()
    return cid


def update_conversation_title(cid: str, title: str) -> None:
    with _lock:
        _db().execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, cid)
        )
        _db().commit()


def list_conversations(user_id: str) -> list[dict]:
    """사용자 소유 대화 목록을 최신순으로 반환."""
    with _lock:
        rows = _db().execute(
            "SELECT id, title, task_topic, created_at FROM conversations "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid: str, user_id: str) -> Optional[dict]:
    """대화 상세 조회 (소유자 검증 포함). 없거나 다른 소유자면 None."""
    with _lock:
        row = _db().execute(
            "SELECT id, title, task_topic, created_at FROM conversations "
            "WHERE id = ? AND user_id = ?",
            (cid, user_id),
        ).fetchone()
    return dict(row) if row else None


def delete_conversation(cid: str, user_id: str) -> bool:
    """대화 삭제 (소유자 검증). 삭제됐으면 True."""
    with _lock:
        cur = _db().execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (cid, user_id),
        )
        _db().commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# 턴
# ---------------------------------------------------------------------------

def append_turn(
    cid: str,
    role: str,
    text: str,
    learner_text: Optional[str] = None,
    rag_examples: Optional[str] = None,
) -> None:
    with _lock:
        _db().execute(
            "INSERT INTO turns (conversation_id, role, learner_text, rag_examples, text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, role, learner_text, rag_examples or "", text, time.time()),
        )
        _db().commit()


def get_turns(cid: str) -> list[dict]:
    with _lock:
        rows = _db().execute(
            "SELECT id, role, learner_text, rag_examples, text, created_at "
            "FROM turns WHERE conversation_id = ? ORDER BY id ASC",
            (cid,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 어드민 전용
# ---------------------------------------------------------------------------

def admin_list_users() -> list[dict]:
    """전체 사용자 + 대화 수, 메시지 수, 마지막 활동 포함."""
    with _lock:
        rows = _db().execute("""
            SELECT
                u.id,
                u.username,
                u.created_at,
                COUNT(DISTINCT c.id)  AS conversation_count,
                COUNT(t.id)           AS message_count,
                MAX(t.created_at)     AS last_active
            FROM users u
            LEFT JOIN conversations c ON c.user_id = u.id
            LEFT JOIN turns t         ON t.conversation_id = c.id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def admin_list_conversations(user_id: Optional[str] = None) -> list[dict]:
    """전체(또는 특정 사용자) 대화 목록 + username 포함."""
    with _lock:
        if user_id:
            rows = _db().execute("""
                SELECT c.id, c.title, c.task_topic, c.created_at,
                       u.username, u.id AS user_id
                FROM conversations c
                JOIN users u ON u.id = c.user_id
                WHERE c.user_id = ?
                ORDER BY c.created_at DESC
            """, (user_id,)).fetchall()
        else:
            rows = _db().execute("""
                SELECT c.id, c.title, c.task_topic, c.created_at,
                       u.username, u.id AS user_id
                FROM conversations c
                JOIN users u ON u.id = c.user_id
                ORDER BY c.created_at DESC
            """).fetchall()
    return [dict(r) for r in rows]


def admin_get_turns(cid: str) -> list[dict]:
    """특정 대화의 전체 턴 반환 (어드민용, 소유자 검증 없음)."""
    with _lock:
        rows = _db().execute(
            "SELECT id, role, learner_text, rag_examples, text, created_at "
            "FROM turns WHERE conversation_id = ? ORDER BY id ASC",
            (cid,),
        ).fetchall()
    return [dict(r) for r in rows]
