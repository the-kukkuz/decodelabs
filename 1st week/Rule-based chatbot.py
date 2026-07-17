import tkinter as tk
from datetime import datetime

class LogicEngine:

    def sanitize(self, text):
        return text.lower().strip()

    def process(self, raw_text):
        clean = self.sanitize(raw_text)

        if clean in ["exit", "bye", "quit"]:
            return "Goodbye. Shutting down.", True

        elif clean in ["hello", "hi", "hey"]:
            return "Hello. How can I assist you?", False

        elif clean in ["what are you?", "what are you"]:
            return "I am a rule-based AI logic engine.", False

        elif clean in ["help", "options"]:
            return (
                "Available commands:\n"
                "• hello\n"
                "• what are you?\n"
                "• time\n"
                "• help\n"
                "• exit"
            ), False

        elif clean in ["time", "what time is it?"]:
            return f"Current time: {datetime.now().strftime('%H:%M:%S')}", False

        else:
            return "I didn't understand that. Type 'help' to see options.", False


#interface

class ChatApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Rule-Based Chatbot")
        self.root.geometry("450x550")
        self.root.configure(bg="#2B2B2B")

        self.engine = LogicEngine()

        # Chat display
        self.chat = tk.Text(
            root,
            state="disabled",
            wrap="word",
            font=("Helvetica", 11),
            bg="#1E1E1E",
            fg="#FFFFFF",
            relief="flat",
            padx=10,
            pady=10
        )
        self.chat.pack(padx=10, pady=(10, 5), fill="both", expand=True)

        # Input bar frame (different colour)
        input_frame = tk.Frame(root, bg="#3C3F41", pady=8, padx=10)
        input_frame.pack(padx=10, pady=(0, 5), fill="x")

        self.entry = tk.Entry(
            input_frame,
            font=("Helvetica", 12),
            bg="#4E5254",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#5E6366",
            highlightcolor="#7EC8E3"
        )
        self.entry.pack(fill="x", ipady=6)
        self.entry.bind("<Return>", self.send)

        # Button frame
        btn_frame = tk.Frame(root, bg="#2B2B2B")
        btn_frame.pack(padx=10, pady=(0, 10), fill="x")

        # Send button (green)
        self.send_btn = tk.Button(
            btn_frame,
            text="SEND",
            font=("Helvetica", 11, "bold"),
            bg="#28A745",
            fg="#FFFFFF",
            activebackground="#218838",
            activeforeground="#FFFFFF",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=6,
            command=self.send
        )
        self.send_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        # Exit button (red)
        self.exit_btn = tk.Button(
            btn_frame,
            text="EXIT",
            font=("Helvetica", 11, "bold"),
            bg="#DC3545",
            fg="#FFFFFF",
            activebackground="#C82333",
            activeforeground="#FFFFFF",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=6,
            command=self.exit_app
        )
        self.exit_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

        # Welcome message
        self.display("Type 'help' for options.\n")
        self.entry.focus_set()

    def display(self, message):
        self.chat.config(state="normal")
        self.chat.insert("end", message + "\n")
        self.chat.config(state="disabled")
        self.chat.see("end")

    def send(self, event=None):
        user_text = self.entry.get()
        if not user_text.strip():
            return

        self.display(f"You: {user_text}")
        self.entry.delete(0, "end")

        response, should_exit = self.engine.process(user_text)
        self.display(f"Assistant: {response}")

        if should_exit:
            self.root.after(1000, self.root.destroy)

    def exit_app(self):
        self.display("Assistant: Goodbye. Shutting down.")
        self.root.after(1000, self.root.destroy)


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()