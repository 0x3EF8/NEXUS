import os
import sys
import io
import requests
import time
import threading
import re
from tqdm import tqdm
from llama_cpp import Llama
import art
import pyperclip
import questionary  

# === Helper for resource paths in frozen executables ===
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# === Configuration Settings ===

# Model configurations
MODELS = {
    "phi3": {
        "name": "Phi-3-mini-4k-instruct-q4.gguf",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
    },
    "deepseek": {
        "name": "DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf"
    }
}

# Configuration Constants
MODEL_DIRECTORY = "models"          # Directory where the models are stored
MAX_SEQUENCE_LENGTH = 4096          # Maximum context length for the AI
NUM_THREADS = 8                     # Number of CPU threads allocated
GPU_LAYERS = 35                     # Number of layers offloaded to the GPU for processing
MAX_TOKENS = 2048                   # Maximum tokens for each AI response
CUSTOM_PROMPT = False               # Custom prompt for AI initialization (set to False to disable)
MAX_HISTORY_SIZE = False            # Limit for conversation history (set to False to disable)
ENABLE_MONITORING = False           # Enable detailed response time and request tracking

# Stealth mode configuration:
# When True, no extra ASCII art, model headers, or loading animations will be shown.
STEALTH_MODE = False

# The file path for saving config changes is the current source file.
CONFIG_FILE = os.path.abspath(__file__)

def edit_config():
    """
    Interactive configuration editor that remains open until the user selects "Exit and Save".
    Editable settings: STEALTH_MODE, ENABLE_MONITORING, CUSTOM_PROMPT, MAX_HISTORY_SIZE.
    This editor reads and writes to the source file (CONFIG_FILE) so that changes persist.
    """
    editable_keys = ["STEALTH_MODE", "ENABLE_MONITORING", "CUSTOM_PROMPT", "MAX_HISTORY_SIZE"]
    
    print("\n=== Configuration Editor ===\n")
    
    while True:
        # Read current source file lines
        try:
            with open(CONFIG_FILE, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print("Error reading source file:", e)
            return

        config_values = {}
        # Look for lines at the top that start with an editable key.
        for line in lines:
            for key in editable_keys:
                if line.strip().startswith(key + " ="):
                    try:
                        value = eval(line.split("=")[1].split("#")[0].strip())
                    except Exception:
                        value = line.split("=")[1].split("#")[0].strip()
                    config_values[key] = value

        # Create a list of choices showing current values and an exit option.
        choices = [f"{key} = {config_values.get(key, 'N/A')}" for key in editable_keys]
        choices.append("Exit and Save")
        
        selection = questionary.select(
            "Select a setting to edit:",
            choices=choices
        ).ask()
        if not selection:
            print("No selection made. Exiting configuration editor.")
            break
        if selection == "Exit and Save":
            print("Exiting configuration editor. Changes saved.")
            break

        # Parse the key from the selection string (format: "KEY = VALUE")
        key_to_edit = selection.split(" = ")[0]
        current_value = config_values.get(key_to_edit, None)
        
        # For boolean settings, offer a toggle.
        if isinstance(current_value, bool):
            new_val_str = questionary.select(
                f"Set new value for {key_to_edit} (current: {current_value}):",
                choices=["True", "False"]
            ).ask()
            new_value = True if new_val_str == "True" else False
        else:
            # For non-boolean values, prompt for a new value.
            new_value_input = questionary.text(
                f"Enter new value for {key_to_edit} (current: {current_value}):"
            ).ask()
            try:
                new_value = int(new_value_input)
            except ValueError:
                try:
                    new_value = float(new_value_input)
                except ValueError:
                    new_value = f"'{new_value_input}'"

        # Update the corresponding line in the source file.
        new_lines = []
        for line in lines:
            if line.strip().startswith(key_to_edit + " ="):
                new_line = f"{key_to_edit} = {new_value}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        try:
            with open(CONFIG_FILE, "w") as f:
                f.writelines(new_lines)
            print(f"{key_to_edit} has been updated to {new_value}.\n")
        except Exception as e:
            print("Error writing to source file:", e)

def select_model():
    """
    Interactive model selector using arrow keys.
    Returns the key of the selected model.
    """
    choices = []
    for key, value in MODELS.items():
        choices.append(f"{key} - {value['name']}")
    selection = questionary.select(
        "Select a model to use:",
        choices=choices
    ).ask()
    if selection:
        # The selection format is "key - model_name"
        return selection.split(" - ")[0]
    else:
        print("No model selected. Exiting.")
        sys.exit(1)

class Nexus:
    def __init__(self, model_directory: str):
        self.history = []
        self.num_requests = 0
        self.total_tokens = 0
        self.model_directory = model_directory
        self.current_model = None
        self.model = None

        if CUSTOM_PROMPT:
            self.history.append(f"<|assistant|>\n{CUSTOM_PROMPT}\n")

    def _ensure_model_exists(self, model_key: str):
        """Checks if the model exists locally; otherwise, downloads it."""
        model_info = MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"Model key '{model_key}' not found in MODELS dictionary.")

        # Build the model path relative to the model directory.
        model_path = os.path.join(self.model_directory, model_info["name"])
        full_model_path = resource_path(model_path)

        # Check if the file exists and is indeed a file.
        if os.path.isfile(full_model_path):
            if not STEALTH_MODE:
                print(f"[Nexus] Using existing model: {full_model_path}")
            return  # Model exists, so no download is needed.

        if not STEALTH_MODE:
            print(f"[Nexus] Model '{model_info['name']}' not found locally. Downloading...")

        os.makedirs(resource_path(self.model_directory), exist_ok=True)

        try:
            headers = {}
            current_size = 0
            if os.path.exists(full_model_path):
                current_size = os.path.getsize(full_model_path)
                headers['Range'] = f"bytes={current_size}-"

            with requests.get(model_info["url"], stream=True, headers=headers) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0)) + current_size

                with open(full_model_path, "ab") as model_file, tqdm(
                    total=total_size, unit="B", unit_scale=True, initial=current_size, disable=STEALTH_MODE
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        model_file.write(chunk)
                        progress_bar.update(len(chunk))

            if not STEALTH_MODE:
                print(f"[Nexus] Model downloaded successfully: {full_model_path}")

        except requests.RequestException as e:
            raise Exception(f"[Nexus] Model download failed: {e}")

    def load_model(self, model_key: str):
        """Loads the AI model into memory for interaction."""
        self._ensure_model_exists(model_key)
        model_path = resource_path(os.path.join(self.model_directory, MODELS[model_key]["name"]))
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

    def generate_response(self, prompt: str) -> str:
        """Generates a response to the given prompt using the loaded AI model."""
        if not self.model:
            return "No model loaded. Please select a model first."

        self.num_requests += 1
        start_time = time.time()

        try:
            sys.stderr = io.StringIO()
            self.history.append(f"<|user|>\n{prompt}\n")

            if MAX_HISTORY_SIZE and len(self.history) > MAX_HISTORY_SIZE * 2:
                self.history = self.history[-(MAX_HISTORY_SIZE * 2):]

            spinner_thread = None
            if not STEALTH_MODE:
                spinner_thread = threading.Thread(target=self._show_loading_animation)
                spinner_thread.daemon = True  
                spinner_thread.start()

            response = self.model(
                f"{''.join(self.history)}\n<|assistant|>",
                max_tokens=MAX_TOKENS,
                stop=["<|end|>"],
                echo=False
            )

            response_text = response['choices'][0]['text'].strip()
            self.total_tokens += len(response_text.split())
            self.history.append(f"<|assistant|>\n{response_text}")

            if spinner_thread:
                self._stop_spinner(spinner_thread)

            end_time = time.time()
            response_time = end_time - start_time

            if STEALTH_MODE:
                print(response_text)
            else:
                print(f"\nNexus Response: {response_text}\n")
                if ENABLE_MONITORING:
                    print(f"\n[Monitoring Details]\n  • Response Time: {response_time:.2f} seconds\n  • Total Requests: {self.num_requests}\n  • Total Tokens Processed: {self.total_tokens}")

            return response_text

        except Exception as e:
            return f"An error occurred: {e}"

        finally:
            sys.stderr = sys.__stderr__

    def reset_conversation(self):
        """Resets the conversation history and clears the screen."""
        os.system("cls" if os.name == "nt" else "clear")
        self.history = []
        if CUSTOM_PROMPT:
            self.history.append(f"<|assistant|>\n{CUSTOM_PROMPT}\n")
        if not STEALTH_MODE:
            print("\nConversation history has been reset. Feel free to start afresh.")
            show_ascii(self.current_model)
        return "Conversation reset successfully."

    def show_help(self):
        """Displays help information."""
        if STEALTH_MODE:
            return "Commands: exit, reset, clear, cc, ca, model, config"
        else:
            return (
                "\n[Nexus Help Center]\n\n"
                "Available commands:\n"
                "  - exit     : Exit the program\n"
                "  - reset    : Clear conversation history\n"
                "  - clear    : Clear the screen\n"
                "  - cc       : Copy the most recent code snippet\n"
                "  - ca       : Copy the latest full response\n"
                "  - model    : Change the AI model\n"
                "  - config   : Edit and save configuration settings permanently\n"
                "\nFor more help, contact the developer 0x3ef8.\n"
            )

    def clear_screen(self):
        """Clears the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def process_command(self, command: str):
        """Handles special commands."""
        if command.lower() == "cc":
            code = self.extract_code(self.history[-1])
            pyperclip.copy(code)
            if not STEALTH_MODE:
                print("The most recent code snippet has been copied to your clipboard.")
        elif command.lower() == "ca":
            response = self.history[-1] if self.history else ""
            pyperclip.copy(response)
            if not STEALTH_MODE:
                print("The latest AI response has been copied to your clipboard.")
        elif command.lower() == "config":
            edit_config()

    def extract_code(self, response: str) -> str:
        """Extracts code blocks from the AI's response."""
        match = re.search(r'```(.*?)```', response, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _show_loading_animation(self):
        """Displays a spinner during processing."""
        spinner = ["|", "/", "-", "\\"]
        self.spinner_running = True
        while self.spinner_running:
            try:
                for s in spinner:
                    print(f"\rNexus is processing... {s}", end="", flush=True)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.spinner_running = False
                if not STEALTH_MODE:
                    print("\rOperation cancelled by user.")
                break

    def _stop_spinner(self, spinner_thread):
        """Stops the spinner."""
        self.spinner_running = False
        spinner_thread.join()

def show_ascii(model_name=None):
    """Displays ASCII art (only if stealth mode is off)."""
    if STEALTH_MODE:
        return
    ascii_art = art.text2art("Nexus")
    lines = ascii_art.splitlines()
    ascii_with_name = "\n".join(lines[:-1]) + f"\n{' ' * (len(lines[-1]) - 20)}Developed by @0x3ef8"
    print(ascii_with_name)
    if model_name:
        print(f"\nCurrent Model: {MODELS[model_name]['name']}")
    print("\nType 'help' for available commands or 'exit' to quit.\n")

def replace_file_path_with_content(text: str) -> str:
    """
    Searches for a Windows-style file path within the text.
    If found and the file exists, reads the file content and replaces the path with the content.
    """
    # Regular expression to capture a Windows file path (basic version)
    file_path_regex = r'([A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]+)'
    matches = re.findall(file_path_regex, text)
    for path in matches:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                # Replace the file path with its content in the prompt
                text = text.replace(path, file_content)
            except Exception as e:
                print(f"Error reading file at {path}: {e}")
    return text

def main():
    try:
        # If running as a frozen executable, change directory to the base path.
        if getattr(sys, 'frozen', False):
            os.chdir(sys._MEIPASS)
        assistant = Nexus(MODEL_DIRECTORY)
        assistant.clear_screen()
        show_ascii()

        if not STEALTH_MODE:
            print("Please select a model to start.")
        model_key = select_model()
        assistant.load_model(model_key)
        assistant.clear_screen()
        show_ascii(model_key)

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                if not STEALTH_MODE:
                    print("Thank you for using Nexus. Goodbye!")
                break

            if user_input.lower() == "reset":
                print(assistant.reset_conversation())
                continue

            if user_input.lower() == "help":
                print(assistant.show_help())
                continue

            if user_input.lower() == "clear":
                assistant.clear_screen()
                continue

            if user_input.lower() in ['cc', 'ca', 'config']:
                assistant.process_command(user_input.lower())
                continue

            if user_input.lower() == "model":
                model_key = select_model()
                assistant.load_model(model_key)
                assistant.clear_screen()
                show_ascii(model_key)
                continue

            # Check if the user input contains a valid file path and replace it with file content.
            processed_input = replace_file_path_with_content(user_input)
            if processed_input != user_input:
                print("\n[File content detected and loaded into the prompt.]\n")
            print()  
            assistant.generate_response(processed_input)

    except KeyboardInterrupt:
        if not STEALTH_MODE:
            print("\nProgram terminated by user.")
    except Exception as e:
        if not STEALTH_MODE:
            print(f"A critical error occurred: {e}")

if __name__ == "__main__":
    main()
