from pydantic import BaseModel


class RTCResponse(BaseModel):
    type: str
    sdp: str


class RTCRequest(BaseModel):
    type: str
    sdp: str
    url: str | None
