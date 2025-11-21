"""
Twilio API routes for WebSocket endpoints.
"""
from fastapi import APIRouter, WebSocket, Query, Request, Response, status
from typing import Optional
import logging

from app.websocket.twilio_websocket import twilio_websocket_handler
from app.utils.context_manager import context_manager

router = APIRouter(prefix="/twilio", tags=["twilio"])
logger = logging.getLogger(__name__)


@router.post("/twilio-voice")
async def twilio_voice(request: Request):
    """
    Twilio incoming call webhook.
    
    Twilio calls this endpoint when a call comes in.
    Returns TwiML that connects the call to a WebSocket stream.
    """

    
    body = await request.form()
    logger.info(f"[Twilio] Body: {body}")
    print(f"[Twilio] Body: {body}")
    call_sid = body.get("CallSid")
    scheduled_call_id = request.query_params.get("scheduled_call_id")
    user_name = request.query_params.get("user_name")
    
    # Get host from request
    host = request.url.hostname or "localhost"
    # Use wss:// for secure WebSocket (or ws:// for non-secure)
    # Adjust port if needed - for production, you might want to use a different port or omit it
    port = request.url.port
    if port and port not in [80, 443]:
        stream_url = f"wss://{host}:{port}/twilio/ws?call_sid={call_sid}"
    else:
        stream_url = f"wss://{host}/twilio/ws?call_sid={call_sid}"
    
    logger.info(f"[Twilio] Streaming media to: {stream_url}")
    print(f"[Twilio] Streaming media to: {stream_url}")  # Also print to console for visibility
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}">
      <Parameter name="scheduled_call_id" value="{scheduled_call_id or ''}" />
      <Parameter name="user_name" value="{user_name or ''}"/>
    </Stream>
  </Connect>
  <Pause length="40"/>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/ws")
async def twilio_websocket(
    websocket: WebSocket,
    call_sid: Optional[str] = Query(None, description="Call SID from Twilio"),
):
    """
    WebSocket endpoint for Twilio Media Streams.

    Twilio will connect to this endpoint and send:
    - JSON metadata messages (events: connected, start, media, stop)
    - Binary audio data for transcription

    Query parameters:
    - call_sid: Call SID from Twilio (used to identify the call)
    """
    # Start cleanup task if not already running
    await context_manager.start_cleanup_task()

    # Handle WebSocket connection
    # phone_number will be extracted from Twilio Stream messages
    await twilio_websocket_handler.handle_connection(
        websocket=websocket, call_sid=call_sid
    )

