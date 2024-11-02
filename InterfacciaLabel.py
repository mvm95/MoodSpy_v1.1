import os
import threading
import time
import tkinter as tk
from tkinter import Toplevel, filedialog, messagebox
from matplotlib import pyplot as plt
from utils.labelUtils import extract_video_from_path, update_label_file, get_number_of_task, acceleration_plot, get_skeletons
from PIL import ImageTk, Image, ImageEnhance
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class Form:
    def __init__(self, master):
        self.current_frame = 0
        self.master = master
        self.master.title('LABEL DESTINI')
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f'{screen_width}x{screen_height}')
        self.master.bind('<Right>', lambda event: self.get_next())
        self.master.bind('<Left>', lambda event: self.get_prev())
        
        # Create and configure frames
        self.top_frame = tk.Frame(master)
        # self.top_frame.pack(pady=5)
        self.top_frame.grid(row=0, column=0, pady=5)
        
        self.video_frame = tk.Frame(master)
        # self.video_frame.pack(side=tk.LEFT, pady=5)
        self.video_frame.grid(row=1, column=0, pady= 10)

        # self.plot_frame = tk.Frame(master)
        # self.plot_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.inputs_frame = tk.Frame(master)
        # self.inputs_frame.pack(side=tk.RIGHT, pady=5)       
        self.inputs_frame.grid(row=1, column=1, pady= 10)

        self.control_frame = tk.Frame(self.inputs_frame)
        self.control_frame.pack( pady=5)

        self.slider_frame = tk.Frame(self.inputs_frame)
        self.slider_frame.pack( pady=5)

        self.bottom_frame = tk.Frame(self.inputs_frame)
        self.bottom_frame.pack( pady=5)

        # Top frame widgets
        self.select_file_button = tk.Button(self.top_frame, text='Open', command=self.select_file)
        self.select_file_button.grid(row=0, column=0, pady=10)
        
        self.file_label = tk.Label(self.top_frame, text='')
        self.file_label.grid(row=1, column=0, pady=5)

        # Video frame widgets
        self.videoOutput = tk.Label(self.video_frame)
        self.videoOutput.grid(row=0, column=0, columnspan=3, pady=10)

        # self.figure, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        # self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        # self.canvas.get_tk_widget().pack()
        # self.figure = plt.figure()
        # Control frame widgets
        self.play_img = ImageTk.PhotoImage(Image.open('images/play.jpg').resize((30, 30)))
        self.stop_img = ImageTk.PhotoImage(Image.open('images/stop.png').resize((30, 30)))
        self.next_img = ImageTk.PhotoImage(Image.open('images/next.jpg').resize((30, 30)))
        self.prev_img = ImageTk.PhotoImage(Image.open('images/previous.jpg').resize((30, 30)))

        self.play_button = tk.Button(self.control_frame, image=self.play_img, command=self.play, state=tk.NORMAL)
        self.play_button.grid(row=0, column=0, padx=10)

        self.stop_button = tk.Button(self.control_frame, image=self.stop_img, command=self.stop, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10)

        self.prev_button = tk.Button(self.control_frame, image=self.prev_img, command=self.get_prev, state=tk.NORMAL)
        self.prev_button.grid(row=0, column=2, padx=10)

        self.next_button = tk.Button(self.control_frame, image=self.next_img, command=self.get_next, state=tk.NORMAL)
        self.next_button.grid(row=0, column=3, padx=10)

        self.draw_skeleton = tk.BooleanVar()
        self.skeleton_checkBox = tk.Checkbutton(self.control_frame, text='Draw skeleton', variable=self.draw_skeleton)
        self.skeleton_checkBox.grid(row=1, column=0, padx = 10, pady= 5)

        self.slider = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.slider_changed)
        self.slider.grid(row=0, column=0, padx=10)

        self.bright_slider = tk.Scale(self.slider_frame, from_=0.5, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, command=self.bright_image)
        self.bright_slider.grid(row=1, column=0, padx=10, pady=5)
        self.brightness = 1.0

        self.start_ts_button = tk.Button(self.bottom_frame, text='Start TS', command=self.start_ts)
        self.start_ts_button.grid(row=0, column=0, padx=10)
        self.start_ts_label = tk.Label(self.bottom_frame, text='')
        self.start_ts_label.grid(row=1, column=0, pady=5)

        self.end_ts_button = tk.Button(self.bottom_frame, text='End TS', command=self.end_ts)
        self.end_ts_button.grid(row=0, column=1, padx=10)
        self.end_ts_label = tk.Label(self.bottom_frame, text='')
        self.end_ts_label.grid(row=1, column=1, pady=5)

        self.class_entry = tk.Entry(self.bottom_frame)
        self.class_entry.grid(row=0, column=2, padx=10)
        self.class_label = tk.Label(self.bottom_frame, text='Class')
        self.class_label.grid(row=1, column=2, pady=5)

        self.update_file_button = tk.Button(self.bottom_frame, text='Update File', command=self.update_file)
        self.update_file_button.grid(row=2, column=0, columnspan=3, pady=5)

        self.plot_window = None  
        self.plot_canvas = None  
        self.ax = None  
        self.plot_button = tk.Button(self.bottom_frame, text="Show Acceleration Plot", command=self.open_plot_window)
        self.plot_button.grid(row=3, column=0, columnspan=3, pady=5)

    def select_file(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            text = 'Selected file:' + self.file_path + f'    N Task: {get_number_of_task(self.file_path)}' +  f'\nPlan: {self.get_plan()}'
            self.file_label.config(text=text)
            # self.plan_label.config(text = f'Plan: {self.get_plan()}')
        else:
            self.file_label.config(text='No file selected')
        while True:
            try:
                self.video_list, self.ts_list = extract_video_from_path(self.file_path)
                self.slider.config(to=len(self.video_list)-1)
                video_name = os.path.dirname(self.file_path)
                video_name = os.path.dirname(video_name)
                self.video_name = os.path.basename(video_name)
                self.current_frame = 0
                self.show_image()
                # self.show_plot()
                break
            except Exception as e:
                print(e)
                flag = messagebox.askyesno('Select .bag file', 'No bag file is selected, want to choose another one?')
                if flag:
                    self.select_file()
                else:
                    self.file_label.config(text='No file selected')
                    break

    def get_plan(self):
        plan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.file_path))), 'Log_files', 'planning_vector.txt')
        # movement_dir = os.path.basename(os.path.dirname(os.path.dirname(initial_path)))
        with open(plan_path,'r') as f:
            plan =  str([eval(line[0] )for line in f.readlines() ][:-3])[1:-1]
        return plan


    def show_image(self):
        self.image = self.video_list[self.current_frame]
        if self.draw_skeleton.get():
            self.image = get_skeletons(self.image, self.file_path, self.ts_list[self.current_frame])
        h, w, _ = self.image.shape
        # self.image = ImageTk.PhotoImage(Image.fromarray(self.image))
        self.image = Image.fromarray(self.image)
        self.enhancer = ImageEnhance.Brightness(self.image)
        self.image = self.enhancer.enhance(self.brightness)
        self.image.resize((750, int(750*h/w)), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(self.image)
        self.videoOutput.configure(image=self.image)
        self.slider.set(self.current_frame)
        self.update_plot()

    def slider_changed(self, event):
        self.current_frame = self.slider.get()
        self.show_image()
    
    def bright_image(self, event):
        self.brightness = self.bright_slider.get()
        self.show_image()

    def play(self):
        self.is_streaming = True
        self.stop_button.config(state=tk.NORMAL)
        self.play_button.config(state=tk.DISABLED)
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        threading.Thread(target=self.show_video).start()

    def show_video(self):
        while self.is_streaming:
            self.show_image()
            self.current_frame += 1
            if self.current_frame >= len(self.video_list):
                self.current_frame = 0
            time.sleep(0.03)

    def open_plot_window(self):
        """Opens or updates the acceleration plot in a new window."""
        if self.plot_window is None or not self.plot_window.winfo_exists():
            # Create a new plot window if it doesn't exist
            self.plot_window = Toplevel(self.master)
            self.plot_window.title("Acceleration Plot")
            self.plot_window.geometry("800x600")
            # Create a figure and canvas for the plot
            figure = plt.Figure(figsize=(12, 6), dpi=100)
            self.ax = figure.add_subplot(111)
            self.plot_canvas = FigureCanvasTkAgg(figure, self.plot_window)
            self.plot_canvas.get_tk_widget().pack()

        # Update the plot with current data
        self.update_plot()

    def update_plot(self):
        """Updates the plot in the existing window."""
        if self.plot_canvas and self.ax:
            acceleration_plot(self.file_path, self.ts_list[self.current_frame], ax=self.ax)  # Update the plot
            self.plot_canvas.draw()  # Redraw the canvas with the updated plot


    def stop(self):
        self.is_streaming = False
        self.stop_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        self.prev_button.config(state=tk.NORMAL)

    def get_prev(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_image()

    def get_next(self):
        if self.current_frame < len(self.video_list) - 1:
            self.current_frame += 1
            self.show_image()
    
    def start_ts(self):
        self.first_ts = self.ts_list[self.current_frame]
        self.start_ts_label.config(text=self.first_ts)

    def end_ts(self):
        self.last_ts = self.ts_list[self.current_frame]
        self.end_ts_label.config(text=self.last_ts)

    def update_file(self):
        self.label_class = int(self.class_entry.get())
        update_label_file(self.label_class, self.first_ts, self.last_ts, self.video_name)



if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('2000x1000')
    form = Form(root)
    root.mainloop()
