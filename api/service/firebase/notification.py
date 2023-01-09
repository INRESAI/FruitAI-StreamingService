import firebase_admin.messaging as fm


def send_notification(token: str, title: str, body: str, image: str|None = None, data: dict|None = None) -> str:
    return fm.send(
        fm.Message(
            notification=fm.Notification(
                title=title,
                body=body,
                image=image,
            ),
            android=fm.AndroidConfig(
                priority="high",
                notification=fm.AndroidNotification(
                    sound="default",
                    vibrate_timings_millis=[1000, 500, 1000],
                    color="#FFFFFF",
                    image=image,
                ),
            ),
            token=token,
            data=data or {},
        )
    )
