from fastapi import HTTPException, Header
from supabase_client import supabase


def verify_jwt(authorization: str = Header(None)) -> str:
    """
    Validates 'Authorization: Bearer <jwt>' header via Supabase.
    Returns the user_id (UUID string) or raises 401.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Kein gültiger Authorization-Header")
    token = authorization[7:].strip()
    try:
        resp = supabase.auth.get_user(token)
        if not resp.user:
            raise HTTPException(status_code=401, detail="Token ungültig oder abgelaufen")
        return str(resp.user.id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token ungültig oder abgelaufen")
