"""
Twilio API routes for WebSocket endpoints.
"""
from fastapi import APIRouter, WebSocket, Query, Request, Response, status
from typing import Optional
import logging
from urllib.parse import quote

from app.websocket.twilio_websocket import twilio_websocket_handler
from app.utils.context_manager import context_manager

router = APIRouter(prefix="/twilio", tags=["twilio"])
logger = logging.getLogger(__name__)


@router.get("/test-websocket-url")
async def test_websocket_url(request: Request):
    """
    Test endpoint to verify WebSocket URL generation.
    Returns the WebSocket URL that would be used for a call.
    """
    host = request.url.hostname or "localhost"
    port = request.url.port
    test_call_sid = "TEST_CALL_SID"
    test_phone = "+17245422869"
    
    query_params = f"call_sid={quote(str(test_call_sid))}&phone_number={quote(test_phone)}"
    if port and port not in [80, 443]:
        stream_url = f"wss://{host}:{port}/twilio/ws?{query_params}"
    else:
        stream_url = f"wss://{host}/twilio/ws?{query_params}"
    
    return {
        "websocket_url": stream_url,
        "host": host,
        "port": port,
        "note": "Test this URL with a WebSocket client to verify connectivity",
        "troubleshooting": {
            "issue": "If Twilio is not connecting, this is likely an ngrok limitation",
            "solutions": [
                "1. Check ngrok web interface at http://127.0.0.1:4040 to see connection attempts",
                "2. Try upgrading ngrok to paid plan (supports WebSockets)",
                "3. Use alternative tunnel: cloudflared tunnel --url http://localhost:8000",
                "4. Deploy to server with public IP/domain"
            ]
        }
    }


@router.post("/twilio-voice")
async def twilio_voice(request: Request):
    """
    Twilio incoming call webhook.
    
    Twilio calls this endpoint when a call comes in.
    Returns TwiML that connects the call to a WebSocket stream.
    """

    
    body = await request.form()

    call_sid = body.get("CallSid")
    # Extract phone number from Caller field (the phone number making the call)
    phone_number = body.get("Caller") or body.get("From")
    scheduled_call_id = request.query_params.get("scheduled_call_id")
    user_name = request.query_params.get("user_name")
    
    logger.info(f"[Twilio] Extracted phone_number: {phone_number}, call_sid: {call_sid}")
    
    # Get host from request
    host = request.url.hostname or "localhost"
    # Use wss:// for secure WebSocket (or ws:// for non-secure)
    # Adjust port if needed - for production, you might want to use a different port or omit it
    port = request.url.port
    # Pass phone_number as query parameter to WebSocket (URL encode to handle + signs)
    query_params = f"call_sid={quote(str(call_sid))}"
    if phone_number:
        query_params += f"&phone_number={quote(phone_number)}"
    if port and port not in [80, 443]:
        stream_url = f"wss://{host}:{port}/twilio/ws?{query_params}"
    else:
        stream_url = f"wss://{host}/twilio/ws?{query_params}"
    
    logger.info(f"[Twilio] Streaming media to: {stream_url}")
    print(f"[Twilio] Streaming media to: {stream_url}")  # Also print to console for visibility
    print(f"[Twilio] Full TwiML being returned:")
    
    # For Twilio Stream, we need to pass parameters via <Parameter> tags, not query params
    # Remove query parameters from URL and pass them as Stream Parameters instead
    base_stream_url = f"wss://{host}/twilio/ws" if (not port or port in [80, 443]) else f"wss://{host}:{port}/twilio/ws"
    
    # TwiML with Stream configuration
    # With <Connect>, track must be "inbound_track" (not "both")
    # With <Start>, track can be "inbound_track", "outbound_track", or "both_tracks"
    # The Stream will send audio as base64-encoded payloads in media events
    # First, greet the caller using <Say> (standard Twilio TTS)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Hello! Thank you for calling. I'm your AI assistant. How can I help you today?</Say>
  <Connect>
    <Stream url="{base_stream_url}" track="inbound_track">
      <Parameter name="call_sid" value="{call_sid or ''}" />
      <Parameter name="phone_number" value="{phone_number or ''}" />
      <Parameter name="scheduled_call_id" value="{scheduled_call_id or ''}" />
      <Parameter name="user_name" value="{user_name or ''}"/>
    </Stream>
  </Connect>
  <Pause length="40"/>
</Response>"""
    
    print(twiml)
    logger.info(f"[Twilio] TwiML Response:\n{twiml}")
    
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/ws")
async def twilio_websocket(
    websocket: WebSocket,
    call_sid: Optional[str] = Query(None, description="Call SID from Twilio"),
    phone_number: Optional[str] = Query(None, description="Phone number from webhook (Caller field)"),
):
    """
    WebSocket endpoint for Twilio Media Streams.

    Twilio will connect to this endpoint and send:
    - JSON metadata messages (events: connected, start, media, stop)
    - Binary audio data for transcription

    Query parameters:
    - call_sid: Call SID from Twilio (used to identify the call)
    - phone_number: Phone number from webhook (extracted from Caller field)
    """
    # Log BEFORE accepting connection to see if endpoint is hit
    logger.info(f"[Twilio] WebSocket endpoint FUNCTION CALLED - call_sid: {call_sid}, phone_number: {phone_number}")
    print(f"[Twilio] ====== WebSocket endpoint FUNCTION CALLED ======")
    print(f"[Twilio] call_sid: {call_sid}, phone_number: {phone_number}")
    print(f"[Twilio] WebSocket URL: {websocket.url}")
    print(f"[Twilio] WebSocket query params: {websocket.query_params}")
    
    # Start cleanup task if not already running
    await context_manager.start_cleanup_task()

    # Handle WebSocket connection
    # phone_number can come from query parameter (from webhook) or be extracted from Twilio Stream messages
    await twilio_websocket_handler.handle_connection(
        websocket=websocket, call_sid=call_sid, phone_number=phone_number
    )

