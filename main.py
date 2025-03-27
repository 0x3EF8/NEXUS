import os
import sys
import io
import requests
import time
import threading
from tqdm import tqdm
from llama_cpp import Llama
import art
import pyperclip
import re
import questionary  

# === Configuration Settings ===

MODELS = {
    "phi3": {
        "name": "Phi-3-mini-4k-instruct-q4.gguf",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-q4.gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
    },
    "deepseek": {
        "name": "DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf"
    }
}

MODEL_DIRECTORY = "models"
MAX_SEQUENCE_LENGTH = 4096
NUM_THREADS = os.cpu_count() or 8  # Use all available CPU cores
GPU_LAYERS = 20  # Reduce layers for better compatibility
MAX_TOKENS = 1024  # Reduce to avoid memory overflows
STEALTH_MODE = False

CONFIG_FILE = os.path.abspath(__file__)

class Nexus:
    def __init__(self, model_directory: str):
        self.history = []
        self.num_requests = 0
        self.total_tokens = 0
        self.model_directory = model_directory
        self.current_model = None
        self.model = None

    def _ensure_model_exists(self, model_key: str):
        model_info = MODELS[model_key]
        model_path = os.path.join(self.model_directory, model_info["name"])
        if os.path.exists(model_path):
            return model_path

        if not STEALTH_MODE:
            print(f"[Nexus] Downloading model '{model_info['name']}'...")

        os.makedirs(self.model_directory, exist_ok=True)

        try:
            headers = {}
            with requests.get(model_info["url"], stream=True, headers=headers) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with open(model_path, "wb") as model_file, tqdm(total=total_size, unit="B", unit_scale=True, disable=STEALTH_MODE) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        model_file.write(chunk)
                        progress_bar.update(len(chunk))

            if not STEALTH_MODE:
                print(f"[Nexus] Model downloaded successfully.")

            return model_path

        except requests.RequestException as e:
            print(f"[Nexus] Model download failed: {e}")
            sys.exit(1)

    def load_model(self, model_key: str):
        model_path = self._ensure_model_exists(model_key)

        try:
            if not STEALTH_MODE:
                print(f"[Nexus] Loading model from '{model_path}'...")

            self.model = Llama(
                model_path=model_path,
                n_ctx=MAX_SEQUENCE_LENGTH,
                n_threads=NUM_THREADS,
                n_gpu_layers=GPU_LAYERS,
                verbose=False
            )
            self.current_model = model_key
            print(f"[Nexus] Model '{MODELS[model_key]['name']}' loaded successfully.")

        except Exception as e:
            print(f"[Nexus] Error loading model: {e}")
            sys.exit(1)

    def generate_response(self, prompt: str) -> str:
        if not self.model:
            return "No model loaded. Please select a model first."

        self.num_requests += 1
        start_time = time.time()

        try:
            sys.stderr = io.StringIO()
            self.history.append(f"<|user|>\n{prompt}\n")

            response = self.model(
                f"{''.join(self.history)}\n<|assistant|>",
                max_tokens=MAX_TOKENS,
                stop=["<|end|>"],
                echo=False
            )

            response_text = response['choices'][0]['text'].strip()
            self.total_tokens += len(response_text.split())
            self.history.append(f"<|assistant|>\n{response_text}")

            end_time = time.time()
            response_time = end_time - start_time

            print(f"\nNexus Response: {response_text}\n")
            print(f"\nResponse Time: {response_time:.2f} seconds | Total Requests: {self.num_requests}")

            return response_text

        except Exception as e:
            return f"An error occurred: {e}"

    def reset_conversation(self):
        os.system("cls" if os.name == "nt" else "clear")
        self.history = []
        print("\nConversation history has been reset.")
        return "Conversation reset successfully."

def show_ascii(model_name=None):
    if STEALTH_MODE:
        return
    ascii_art = art.text2art("Nexus")
    print(ascii_art)
    if model_name:
        print(f"\nCurrent Model: {MODELS[model_name]['name']}")
    print("\nType 'help' for available commands or 'exit' to quit.\n")

def select_model():
    choices = [f"{key} - {value['name']}" for key, value in MODELS.items()]
    selection = questionary.select("Select a model to use:", choices=choices).ask()
    return selection.split(" - ")[0] if selection else None

def main():
    assistant = Nexus(MODEL_DIRECTORY)
    assistant.reset_conversation()
    show_ascii()

    selected_model = select_model()
    if selected_model:
        assistant.load_model(selected_model)
    else:
        print("No model selected. Exiting.")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n[User]: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting Nexus. Goodbye!")
                sys.exit(0)
            elif user_input.lower() == "reset":
                assistant.reset_conversation()
            else:
                assistant.generate_response(user_input)
        except KeyboardInterrupt:
            print("\nExiting Nexus. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
