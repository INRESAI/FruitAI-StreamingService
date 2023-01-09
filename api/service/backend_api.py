# from aiohttp import ClientSession
from requests import Session

BASE_URL = "http://178.128.19.31:3003"


# class BackendApiAsyncSession(ClientSession):
#     def __init__(self):
#         super().__init__(BASE_URL)

#     async def get_(self, ):
#         async with self.get("/docs") as response:
#             return await response.json()


class BackendApiSyncSession(Session):
    def __init__(self):
        super().__init__()

    def get_camera_info(self, camera_id: str) -> dict:
        return self.get(f"{BASE_URL}/camera/{camera_id}").json()

    def create_notification(self, title: str, content: str, camera_id: str, user_id: str) -> dict:
        return self.post(f"{BASE_URL}/notification", json={
            "title": title,
            "content": content,
            "cameraId": camera_id,
            "userId": user_id,
        }).json()
