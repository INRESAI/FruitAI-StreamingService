from os import listdir
from uuid import uuid4

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai.model import FruitTrackingModel

from ..schema.rtc import RTCRequest, RTCResponse
from ..service.firebase import send_notification

rtc_router = APIRouter()


@rtc_router.post("", response_model=RTCResponse)
async def rtc_streaming(offer: RTCRequest):
    def handler(event_type: str, image: bytes):
        id = str(uuid4())
        with open(f"noti/{id}", "wb") as f:
            f.write(image)
        send_notification(
            token=offer.token,
            title="Fruit Tracking",
            body=f"Detected a {event_type} fruit",
            image=f"http://178.128.19.31:4600/noti/{id}"
        )
    pc = RTCPeerConnection(
        RTCConfiguration([
            RTCIceServer(
                urls=["stun:14.224.131.219:3478"],
            ),
            RTCIceServer(
                urls=["turn:14.224.131.219:3478"],
                username="turnserver",
                credential="turnserver",
            ),
        ])
    )
    model = FruitTrackingModel.start(offer.url or "test/test1.mp4")
    model.add_event_handler(handler)
    pc.addTrack(model.get_stream_track())
    await pc.setRemoteDescription(offer)
    await pc.setLocalDescription(await pc.createAnswer())
    return pc.localDescription


@rtc_router.get("/noti")
def all_notis():
    return listdir("noti")


@rtc_router.get("/noti/{id}")
def noti_image(id: str):
    def generator():
        with open(f"noti/{id}", "rb") as f:
            yield f.read()
    return StreamingResponse(generator(), media_type="image/jpeg")


@rtc_router.get("/preview")
def preview_stream(url: str | None):
    model = FruitTrackingModel.start(url or "test/test1.mp4")
    def generator():
        yield model.get_preview_image()
    return StreamingResponse(generator(), media_type="image/jpeg")


@rtc_router.get("/stop")
def stop_all_stream():
    for url in FruitTrackingModel._instances.copy():
        FruitTrackingModel.stop(url)
