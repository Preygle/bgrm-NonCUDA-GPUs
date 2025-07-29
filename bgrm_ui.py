import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk # Import ttk for themed widgets
import os
import threading
from bgrm_onnx import BackgroundRemoverONNX

class BgRemoverUI:
    def __init__(self, master):
        self.master = master
        master.title("Background Remover")
        master.geometry("650x550") # Set a fixed window size
        master.resizable(False, False) # Make window non-resizable

        # Configure a style for ttk widgets
        style = ttk.Style()
        style.theme_use('clam') # Use 'clam' theme for a flatter look, or 'alt', 'default', 'classic'
        style.configure('TFrame', background='#f0f0f0') # Light grey background for the main frame
        style.configure('TLabel', font=('Segoe UI', 10), background='#f0f0f0')
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=5, background='#4CAF50', foreground='white') # Green button with white text
        style.map('TButton', background=[('active', '#45a049')], foreground=[('active', 'white')]) # Darker green on hover
        style.configure('TEntry', font=('Segoe UI', 10))
        style.configure('TText', font=('Consolas', 9))

        # Main frame for better organization and padding
        main_frame = ttk.Frame(master, padding="15 15 15 15", style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input Path
        self.input_label = ttk.Label(main_frame, text="Input Image/Folder:")
        self.input_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.input_path_var = tk.StringVar()
        self.input_entry = ttk.Entry(main_frame, textvariable=self.input_path_var, width=50)
        self.input_entry.grid(row=0, column=1, padx=5, pady=5)
        self.input_browse_file_button = ttk.Button(main_frame, text="Browse File", command=self.browse_input_file)
        self.input_browse_file_button.grid(row=0, column=2, padx=5, pady=5)
        self.input_browse_folder_button = ttk.Button(main_frame, text="Browse Folder", command=self.browse_input_folder)
        self.input_browse_folder_button.grid(row=0, column=3, padx=5, pady=5)

        # Output Path
        self.output_label = ttk.Label(main_frame, text="Output Folder:")
        self.output_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_path_var = tk.StringVar()
        self.output_entry = ttk.Entry(main_frame, textvariable=self.output_path_var, width=50)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5)
        self.output_browse_button = ttk.Button(main_frame, text="Browse", command=self.browse_output)
        self.output_browse_button.grid(row=1, column=2, padx=5, pady=5)

        # Process Button
        self.process_button = ttk.Button(main_frame, text="Process", command=self.start_processing_thread)
        self.process_button.grid(row=2, column=0, columnspan=4, pady=15)

        # Message Log
        self.log_label = ttk.Label(main_frame, text="Log:")
        self.log_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        
        # Use tk.Text for the log as ttk does not have a Text widget
        self.log_text = tk.Text(main_frame, height=15, width=70, state='disabled', wrap=tk.WORD) # Added wrap
        self.log_text.grid(row=4, column=0, columnspan=4, padx=5, pady=5)
        self.log_scroll = ttk.Scrollbar(main_frame, command=self.log_text.yview)
        self.log_scroll.grid(row=4, column=4, sticky='ns')
        self.log_text['yscrollcommand'] = self.log_scroll.set

        # Initialize BackgroundRemoverONNX with a logger that updates the UI
        self.remover = BackgroundRemoverONNX(model_path="rmbg.onnx", logger=self.log_message)

    def browse_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")])
        if file_path:
            self.input_path_var.set(file_path)

    def browse_input_folder(self):
        folder_path = filedialog.askdirectory(title="Select Input Folder")
        if folder_path:
            self.input_path_var.set(folder_path)

    def browse_output(self):
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_path_var.set(folder_path)

    def log_message(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # Auto-scroll to the end
        self.log_text.config(state='disabled')
        self.master.update_idletasks() # Update UI immediately

    def start_processing_thread(self):
        input_path = self.input_path_var.get()
        output_folder = self.output_path_var.get()

        if not input_path:
            messagebox.showerror("Error", "Please select an input image or folder.")
            return
        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END) # Clear previous logs
        self.log_text.config(state='disabled')
        self.log_message("Starting background removal...")
        self.process_button.config(state='disabled') # Disable button during processing

        # Run processing in a separate thread to keep UI responsive
        processing_thread = threading.Thread(target=self.process_images, args=(input_path, output_folder))
        processing_thread.start()

    def process_images(self, input_path, output_folder):
        try:
            if os.path.isfile(input_path):
                # Process single image
                output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_no_bg.png"
                output_path = os.path.join(output_folder, output_filename)
                self.remover.process_image(input_path, output_path, show_progress=True)
            elif os.path.isdir(input_path):
                # Process batch
                self.remover.process_batch(input_path, output_folder)
            else:
                self.log_message(f"Error: Invalid input path: {input_path}")

            self.log_message("\nProcessing complete!")
        except Exception as e:
            self.log_message(f"An unexpected error occurred: {e}")
        finally:
            self.process_button.config(state='normal') # Re-enable button

if __name__ == "__main__":
    root = tk.Tk()
    app = BgRemoverUI(root)
    root.mainloop()
