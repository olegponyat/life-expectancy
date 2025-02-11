import pyperclip
import pyautogui
import keyboard
import time
from datetime import datetime

clipboard_history = []

def monitor_clipboard():
    """Monitors the clipboard for new items."""
    while True:
        try:
            current_clip = pyperclip.paste()
            if len(clipboard_history) == 0 or clipboard_history[-1] != current_clip:
                clipboard_history.append(current_clip)
                if len(clipboard_history) > 2:
                    clipboard_history.pop(0)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error monitoring clipboard: {e}")
            time.sleep(1)

def paste_last_two_items():
    """Physically pastes the last two items in the clipboard history."""
    if len(clipboard_history) >= 2:
        pyperclip.copy("\n".join(clipboard_history))
        pyautogui.hotkey('ctrl', 'v')
        print("Pasted the last two items.")
    else:
        print("Not enough items in the clipboard history to paste.")

def wait_for_timer(target_time):
    """Waits until the target time."""
    print(f"Waiting for {target_time} to paste...")
    while True:
        now = datetime.now().strftime("%H:%M")
        if now == target_time:
            paste_last_two_items()
            break
        time.sleep(1)

import threading
clipboard_thread = threading.Thread(target=monitor_clipboard, daemon=True)
clipboard_thread.start()

print("Press 's' to start the timer.")

keyboard.wait("s")
print("Timer started. Pasting at 2:02.")

# Set the target time
target_time = "02:02"  # Adjust the time as needed
wait_for_timer(target_time)
