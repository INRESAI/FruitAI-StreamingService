from os import listdir
from uuid import uuid4

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai.model import FruitTrackingModel

from ..schema.rtc import RTCRequest, RTCResponse
from ..service.firebase import send_notification
from ..service.backend_api import BackendApiSyncSession

rtc_router = APIRouter()


@rtc_router.post("", response_model=RTCResponse)
async def rtc_streaming(offer: RTCRequest):
    def handler(event_type: str, image: bytes):
        try:
            id = str(uuid4())
            with open(f"noti/{id}", "wb") as f:
                f.write(image)
            send_notification(
                token=offer.token,
                title="Fruit Tracking",
                body=f"Detected a {event_type} fruit",
                image=f"http://178.128.19.31:4600/rtc/noti/{id}",
                data={
                    "camera_id": str(offer.camera_id),
                }
            )
            session = BackendApiSyncSession()
            session.create_notification(
                title="Fruit Tracking",
                content=f"Detected a {event_type} fruit",
                camera_id=offer.camera_id,
                user_id=offer.user_id,
            )
            print(offer.token)
        except Exception as e:
            print(e)
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
    model = FruitTrackingModel.start(offer.camera_id, {"link": offer.url})
    if offer.token:
        model.add_event_handler(handler, offer.token)
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
def preview_stream(id: str):
    if FruitTrackingModel.status(id):
        model = FruitTrackingModel.get(id)
        def generator():
            yield model.get_preview_image()
        return StreamingResponse(generator(), media_type="image/jpeg")
    else:
        def generator():
            with open("default.png", "rb") as f:
                yield f.read()
        return StreamingResponse(generator(), media_type="image/png")


@rtc_router.get("/stop")
def stop_all_stream():
    for id in FruitTrackingModel._instances.copy():
        FruitTrackingModel.stop(id)
