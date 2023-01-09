from pydantic import BaseModel


class RTCResponse(BaseModel):
    type: str
    sdp: str


class RTCRequest(BaseModel):
    type: str
    sdp: str
    camera_id: str | None
    user_id: str | None
    url: str | None
    token: str | None
