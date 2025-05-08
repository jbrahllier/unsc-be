"""
UNResolutionProcessor: ProcessHelpers.py

    an assortment of helper functions:
    - an animated printer (twirls while long processes run) for looks, funzies, visual confirmation it's still running
    - functions that list/print all files in a directory
    - standard elapsed time formatter for convenience
    - check/install libraries function
    - Y/N custom user input prompt returning T/F

"""
import os
import re
import subprocess
import sys
import importlib
import time
import threading

class AnimatedPrinter:
    def __init__(self):
        self.animation = "|/-\\"
        self.animation_running = False
        self.animation_lock = threading.Lock()

    def animate(self, state):
        if state:
            if not self.animation_running:
                self.animation_running = True
                threading.Thread(target=self.start_animation).start()
        else:
            self.animation_running = False

    def start_animation(self):
        while self.animation_running:
            for i in range(4):
                with self.animation_lock:
                    if not self.animation_running:
                        break
                    time.sleep(0.2)
                    sys.stdout.write("\r" + self.animation[i % len(self.animation)])
                    sys.stdout.flush()
        sys.stdout.write("\r")
        sys.stdout.flush()

    def safe_print(self, *args, **kwargs):
        with self.animation_lock:
            original_state = self.animation_running
            if self.animation_running:
                self.animate(False)
            print(*args, **kwargs)
            if original_state:
                self.animate(True)

# create a global instance of AnimatedPrinter
animated_printer = AnimatedPrinter()

def list_files_in_directory(directory_path):
    files = [os.path.join(root, file)
             for root, _, files in os.walk(directory_path)
             for file in files if file.endswith('.pdf')]
    
    sum_files = 0
    for file in files:
        modified_filename = re.sub(r'[^\w\s]', '-', re.sub(r'\.pdf$', '', os.path.basename(file)))
        animated_printer.safe_print(modified_filename)
        sum_files += 1
    
    animated_printer.safe_print(f"Number of files: {sum_files}")

def print_num_files_in_directory(directory_path):
    files = [os.path.join(root, file)
             for root, _, files in os.walk(directory_path)
             for file in files if file.endswith('.pdf')]
    sum_files = 0
    for _ in files:
        sum_files += 1
    
    animated_printer.safe_print(f"Number of files: {sum_files}")

def format_elapsed_time(elapsed_seconds):
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    return f"{hours}h {minutes}m {seconds:.2f}s"

def check_and_install_packages(packages):
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"{package} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            animated_printer.safe_print("Installed.")

def get_user_input(prompt):
    while True:
        user_input = input(prompt + " [Y/N]: ").strip().lower()
        if user_input in ['y', 'n']:
            return user_input == 'y'
        else:
            animated_printer.safe_print("Invalid input. Please enter 'Y' or 'N'.")
