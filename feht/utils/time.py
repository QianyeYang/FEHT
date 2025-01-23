from datetime import datetime


def current_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
