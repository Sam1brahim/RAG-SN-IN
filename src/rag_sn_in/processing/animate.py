import sys
import time
import threading

def animate(stop_event, text="Starting Langfuse"):
    dots = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{text}{'.' * dots}{' ' * (3 - dots)}")
        sys.stdout.flush()
        dots = (dots + 1) % 4
        time.sleep(0.4)
    sys.stdout.write(f"\r{text}... Done!\n")