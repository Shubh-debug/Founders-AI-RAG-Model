import uuid
import logging
from typing import Any, Dict, List, Optional

from ..core.database import get_database_manager
from ..core.security import hash_password, verify_password

logger = logging.getLogger(__name__)


db = get_database_manager()


async def ensure_user(user_id: str) -> None:
    if not user_id:
        raise ValueError("user_id is required")
    try:
        # Upsert-like behavior: insert if not exists
        await db.execute_command(
            """
            INSERT INTO users (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO UPDATE SET updated_at = NOW()
            """,
            user_id,
        )
    except Exception as e:
        logger.error(f"ensure_user failed for {user_id}: {e}")
        raise


async def create_user(user_id: str, email: str, password: str) -> bool:
    if not (user_id and email and password):
        raise ValueError("user_id, email and password are required")
    await ensure_user(user_id)
    pwd_hash = hash_password(password)
    try:
        await db.execute_command(
            """
            UPDATE users SET email=$2, password_hash=$3, updated_at=NOW() WHERE user_id=$1
            """,
            user_id, email, pwd_hash,
        )
        return True
    except Exception as e:
        logger.error(f"create_user failed: {e}")
        return False


async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    try:
        row = await db.execute_one(
            "SELECT user_id, email, password_hash, created_at FROM users WHERE email=$1",
            email,
        )
        return row
    except Exception as e:
        logger.error(f"get_user_by_email failed: {e}")
        return None


async def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    user = await get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user.get("password_hash") or ""):
        return None
    return user


async def save_user_form_responses(user_id: str, responses: List[Dict[str, Any]]) -> str:
    if not user_id:
        raise ValueError("user_id is required")
    if responses is None:
        raise ValueError("responses is required")

    await ensure_user(user_id)

    assessment_id = str(uuid.uuid4())
    try:
        await db.execute_command(
            """
            INSERT INTO user_form_responses (assessment_id, user_id, responses)
            VALUES ($1, $2, $3::jsonb)
            """,
            assessment_id,
            user_id,
            responses,
        )
        return assessment_id
    except Exception as e:
        logger.error(f"save_user_form_responses failed: {e}")
        raise


async def get_latest_user_form_responses(user_id: str) -> Optional[Dict[str, Any]]:
    if not user_id:
        raise ValueError("user_id is required")
    try:
        row = await db.execute_one(
            """
            SELECT assessment_id, responses, created_at
            FROM user_form_responses
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            user_id,
        )
        return row
    except Exception as e:
        logger.error(f"get_latest_user_form_responses failed: {e}")
        raise


def format_user_response_guideline(responses: List[Dict[str, Any]]) -> str:
    """
    Convert structured form responses into a concise guideline string
    suitable for inclusion in LLM prompts.
    """
    if not responses:
        return ""

    lines: List[str] = ["User Profile Responses:"]
    for item in responses:
        # Prefer question text if present; fallback to id/type
        question = item.get("question")
        ans = item.get("answer")
        if isinstance(ans, list):
            value = ", ".join(str(x) for x in ans if x)
        else:
            value = str(ans) if ans is not None else ""
        if question:
            lines.append(f"- {question}: {value}")
        else:
            qid = item.get("id")
            lines.append(f"- Q{qid}: {value}")

    guideline = "\n".join(lines)
    # Truncate if overly long to keep prompts efficient
    if len(guideline) > 4000:
        guideline = guideline[:3800] + "\n...[truncated user responses]..."
    return guideline
