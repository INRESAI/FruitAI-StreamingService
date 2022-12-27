from aiohttp import ClientSession


class BackendApi(ClientSession):
    BASE_URL = "http://localhost:5000"

    def __init__(self):
        super().__init__(self.BASE_URL)

    async def get_(self, ):
        async with self.get("/docs") as response:
            return await response.json()
