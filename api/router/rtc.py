from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
from fastapi import APIRouter

from ai.model import FruitTrackingModel

from ..schema.rtc import RTCRequest, RTCResponse

rtc_router = APIRouter()


@rtc_router.post("", response_model=RTCResponse)
async def rtc_streaming(offer: RTCRequest):
    pc = RTCPeerConnection(
        RTCConfiguration([
            RTCIceServer(
                urls=["stun:14.224.131.219:3478"]
            ),
            RTCIceServer(
                urls=["turn:14.224.131.219:3478"],
                credential="turnserver",
                username="turnserver"
            ),
        ])
    )
    pc.addTrack(FruitTrackingModel.start(offer.url or "test2.mp4").get_stream_track())
    await pc.setRemoteDescription(offer)
    await pc.setLocalDescription(await pc.createAnswer())
    return pc.localDescription


@rtc_router.get("/stop")
async def stop_all_stream():
    for url in FruitTrackingModel._instances.copy():
        FruitTrackingModel.stop(url)
