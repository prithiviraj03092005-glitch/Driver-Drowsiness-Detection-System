import pygame
import os

pygame.mixer.init()
_playing = False

def play_alarm(path="sounds/alarm.mp3"):
    global _playing
    if not os.path.exists(path):
        print(f"[ALARM ERROR] File not found: {path}")
        return

    if not _playing:
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            _playing = True
            print(f"[ALARM] Playing sound: {path}")
        except Exception as e:
            print(f"[ALARM ERROR] Failed to play: {e}")
            _playing = False

    if not pygame.mixer.music.get_busy():
        _playing = False
        