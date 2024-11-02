import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from serial.tools.list_ports import comports 
from PIL import ImageTk, Image
from BioHarness import BioHarness
from RealSense import RealSense
import time
import numpy as np
import os
import threading
import cv2
import sys
from datetime import timezone, datetime
import subprocess
import shutil
from utils.experiment_utils import create_video_planning

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.stdout_or = sys.stdout
        self.stderr_or = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Scroll to the end
        self.text_widget.config(state=tk.DISABLED)

    def flush(self):
        pass

    def restore_stdout_stderr(self):
        sys.stderr = self.stderr_or
        sys.stdout = self.stdout_or

class AskMovementType:
    def __init__(self, master):
        self.master = master
        self.response = None
        self.top = tk.Toplevel(master)
        self.top.title("Excercize choice")
        self.top.geometry("300x150")
        self.label = tk.Label(self.top, text="Do you want to perform Stop task or Movement task?")
        self.label.pack(pady=10)
        self.stop_button = tk.Button(self.top, text="Stop", command=self.stop_task)
        self.stop_button.pack(side=tk.LEFT, padx=20)
        self.movement_button = tk.Button(self.top, text="Movement", command=self.movement_task)
        self.movement_button.pack(side=tk.LEFT, padx=20)  
        self.both_button = tk.Button(self.top, text="Both", command=self.both_task)
        self.both_button.pack(side=tk.LEFT, padx=20)
        self.top.grab_set()

    def stop_task(self):
        self.response = "Stop"
        self.top.destroy()

    def movement_task(self):
        self.response = "Movement"
        self.top.destroy()
    
    def both_task(self):
        self.response = 'Both'
        self.top.destroy()

class Form:
    def __init__(self, master):
        self.ts = None
        self.master = master
        self.pepperStart = False
        self.pepperStop = False

        self.streaming = False

        this_row = 0
        master.title('MOOD-SPY')
        self.subtitle_label = tk.Label(master, text = 'Multi-signal Observation & Output determination - Sensor based Psychophysiological Yield',
                                       font=("Helvetica", 12))
        self.subtitle_label.grid(row=this_row, column=0, columnspan=3, pady=10, padx= 10)
        
        #TAG ROW
        this_row += 1
        self.label_tag = tk.Label(master, text = 'Measurement Tag:')
        self.label_tag.grid(row = this_row, column=0, pady = 10)

        self.entry_tag = tk.Entry(master)
        self.entry_tag.grid(row = this_row, column=0, columnspan=2, pady = 10, padx = 5)
        
        self.update_tag = tk.Button(master, text = 'Update', command = self.display_id_info, state = tk.NORMAL)
        self.update_tag.grid(row = this_row, column = 1, columnspan=1)

        self.label_info = tk.Label(master, text = f'-')
        self.label_info.grid(row = this_row, column = 2, pady = 10)

        #ANNOTATION ROW
        this_row += 1
        self.annotation_tag = tk.Entry(master)
        self.annotation_tag.grid(row = this_row, column=0, pady = 10)

        self.annotation_tag_button = tk.Button(master, text = 'Annotate', command = self.annotate, state= tk.NORMAL)
        self.annotation_tag_button.grid(row=this_row, column=0, columnspan=2, pady = 10)

        #BIOHARNESS
        this_row += 1

        self.label_bioharness_com = tk.Label(master, text = 'Select Bioharness COM port:')
        self.label_bioharness_com.grid(row=this_row, column=0, pady = 10)

        self.combo_bioharness=ttk.Combobox(master, values = [port.device for port in comports()])
        self.combo_bioharness.grid(row=this_row, column=1, pady = 10)

        this_row += 1
        self.save_bioharness_data = tk.BooleanVar()
        self.checkbox_save_bioharness = tk.Checkbutton(master, text = 'Save Bioharness data', variable=self.save_bioharness_data)
        self.checkbox_save_bioharness.grid(row= this_row, column=0, columnspan=2, pady = 10)

        self.status_bioharness = tk.Label(master, text = '-')
        self.status_bioharness.grid(row = this_row, column = 1, pady = 10)

        #REALSENSE
        this_row += 1
        self.save_realsense_data = tk.BooleanVar()
        self.checkbox_save_realsense = tk.Checkbutton(master, text = 'Save RealSense data', variable=self.save_realsense_data)
        self.checkbox_save_realsense.grid(row= this_row, column=0, columnspan=2, pady = 5)

        self.status_realsense = tk.Label(master, text= '-')
        self.status_realsense.grid(row = this_row, column=1, pady = 5)

        #PEPPER
        this_row += 1
        self.save_pepper_data = tk.BooleanVar()
        self.checkbox_save_pepper = tk.Checkbutton(master, text = 'Save Pepper data', variable=self.save_pepper_data)
        self.checkbox_save_pepper.grid(row= this_row, column=0, columnspan=2, pady = 5)

        self.status_pepper = tk.Label(master, text= '-')
        self.status_pepper.grid(row = this_row, column=1, pady = 5)

        #ALL SENSORS
        this_row += 1
        self.save_all_data = tk.BooleanVar()
        self.checkbox_save_all = tk.Checkbutton(master, text = 'Save all data', variable = self.save_all_data, command = self.save_all_data_checkbox)
        self.checkbox_save_all.grid(row = this_row, column = 0, columnspan=2, pady = 5)

        #CONNECT AND STOP
        this_row += 1
        self.connect_button = tk.Button(master, text = 'Connect' , command= self.connect, state = tk.DISABLED)
        self.connect_button.grid(row= this_row, column=0, pady = 5)

        self.stop_button = tk.Button(master, text = 'Disconnect', command = self.disconnect, state = tk.DISABLED )
        self.stop_button.grid(row = this_row, column=1, pady = 10)

        #BASELINE, START VIDEO, STOP VIDEO
        this_row += 1
        self.baseline_button = tk.Button(master, text = 'Baseline', command = self.baseline, state = tk.DISABLED )
        self.baseline_button.grid(row = this_row, column=0, pady=10)

        self.start_video_button = tk.Button(master, text = 'Start Video', command=self.start_video, state = tk.DISABLED)
        self.start_video_button.grid(row = this_row, column=0, columnspan=2, pady = 10)

        self.stop_video_button = tk.Button(master, text='Stop Video', command=self.stop_video, state = tk.DISABLED)
        self.stop_video_button.grid(row = this_row, column=0, columnspan=4, pady = 10)

        self.questionaire_ecg_button = tk.Button(master, text='Questionaire', command = self.questionaire, state = tk.DISABLED)
        self.questionaire_ecg_button.grid(row = this_row, column = 1, columnspan=2, pady = 10)

        #START PEPPER, STOP PEPPER
        this_row += 1
        self.start_pepper_button = tk.Button(master, text = 'Start Pepper', command= self.start_pepper, state=tk.DISABLED)
        self.start_pepper_button.grid(row=this_row, column = 0, pady=10)

        self.pause_pepper_button = tk.Button(master, text='Pause Pepper', command = self.pause_pepper, state = tk.DISABLED)
        self.pause_pepper_button.grid(row = this_row, column =1, pady = 10)

        #OUTPUT VIDEO AND CONSOLE
        this_row += 1
        self.output = tk.Text(master, wrap ='word', state = 'disabled')
        self.output.grid(row=this_row, column=0, columnspan=3, pady=30)

        self.console_redirector= ConsoleRedirector(self.output)

        self.videoPlanningText = tk.Label(master)
        self.videoPlanningText.grid(row = 2, column=4, pady=10, padx=10)
        self.videoLabel = tk.Label(master)
        self.videoLabel.grid(row=3, column=4, rowspan=this_row+1-3, pady = 30, padx=10)
        
        for i in range(self.master.grid_size()[1]):
            self.master.grid_rowconfigure(i, weight=3)    
   
    def display_id_info(self, ask=True):
        tag_text = self.entry_tag.get()
        self.n_videos = 0
        self.n_task = 0
        if ask:
            askButton = AskMovementType(self.master)
            self.master.wait_window(askButton.top)
            self.task = askButton.response
        self.video_folders = os.path.join('Measures', tag_text, self.task)
        if not os.path.exists(os.path.join(self.video_folders, 'Log_files')):
            os.makedirs(os.path.join(self.video_folders, 'Log_files'))
        if not os.path.exists(os.path.join(self.video_folders, 'Log_files', 'planning_vector.txt')):
            self.planning_vector = create_video_planning(movement=self.task)
            with open(os.path.join(self.video_folders, 'Log_files', 'planning_vector.txt'), 'w') as file:
                for item in self.planning_vector:
                    file.write(str(item)+'\n')
            with open(os.path.join(self.video_folders, 'Log_files', 'current_try.txt'), 'w') as file:
                file.write('0')
        else:
            with open(os.path.join(self.video_folders, 'Log_files', 'planning_vector.txt'), 'r') as file:
                lines = file.readlines()
                self.planning_vector = [int(num.strip()) for num in lines]
        if ask:
            print(self.planning_vector)
        self.connection_log = os.path.join(self.video_folders, 'Log_files', 'Connection_log.txt')
        if os.path.exists(self.video_folders):
            videos = [v for v in os.listdir(self.video_folders) if 'V' in v]
            self.n_videos = len(videos)
            log_file = os.path.join(self.video_folders, 'Log_files', f'{tag_text}_pepper.txt')
            if os.path.exists(log_file):
                with open(log_file, 'r') as file:
                    for line in file:
                        if line.startswith('Stop task from timestamp') or line.startswith('Movement task from timestamp'):
                            self.n_task += 1
        with open(os.path.join(self.video_folders, 'Log_files', 'current_try.txt'), 'r') as file:
            self.current_try = int(file.read())
        print(self.planning_vector[self.current_try:self.current_try+10])
        self.label_info.config(text = f'ID: {self.entry_tag.get()}, task = {self.task}, num_video = {self.n_videos}, num_tasks = {self.n_task}, current_try = {self.current_try}')
        self.video_tag = f'{tag_text}V{(self.n_videos+1):03d}'
        self.video_path = os.path.join(self.video_folders, self.video_tag )
        if not os.path.exists(self.video_folders):
            os.makedirs(self.video_folders)
        self.baseline_path = os.path.join(self.video_folders, 'Baseline')
        self.connect_button.config(state = tk.NORMAL)

    def annotate(self):
        tag_text = self.entry_tag.get()
        if not tag_text:
            messagebox.showerror('Error', 'Please enter a Measurement ID name')
            return
        if not os.path.exists(self.connection_log):
            messagebox.showerror('Error', 'Please Connect at least one sensor')
            return
        text = self.annotation_tag.get()
        timestamp=int(datetime.now(timezone.utc).timestamp()*1000)
        with open(self.connection_log, 'a') as file:
            file.write(f'Timestamp: {timestamp} - {text}')

    def save_all_data_checkbox(self):
        if self.save_all_data.get():
            self.checkbox_save_bioharness.select()
            self.checkbox_save_bioharness.config(state=tk.DISABLED)
            self.checkbox_save_realsense.select()
            self.checkbox_save_realsense.config(state=tk.DISABLED)
            self.checkbox_save_pepper.select()
            self.checkbox_save_pepper.config(state = tk.DISABLED)
        else:
            self.checkbox_save_bioharness.config(state=tk.NORMAL)
            self.checkbox_save_realsense.config(state=tk.NORMAL)
            self.checkbox_save_pepper.config(state = tk.NORMAL)

    def connect(self):
        if not self.entry_tag.get():
            messagebox.showerror('Error', 'Please insert an ID tag')
            return
    
        if self.save_bioharness_data.get():
            bioharness_port = self.combo_bioharness.get()
            if not bioharness_port:
                messagebox.showerror('Error', 'Please select the COM port for the BioHarness')
                return
        
        self.connect_button.config(state = tk.DISABLED)
        self.baseline_button.config(state = tk.NORMAL)
        self.start_video_button.config(state = tk.NORMAL)
        # self.stop_video_button.config(state = tk.NORMAL)
        self.questionaire_ecg_button.config(state=tk.NORMAL)
        self.stop_button.config(state = tk.NORMAL)
        # self.start_pepper_button.config(state = tk.NORMAL)

        if self.save_bioharness_data.get():
            self.streaming_bh = True
            self.bioHarness = BioHarness(connect_bio = True, tag_text=self.entry_tag.get(), port =bioharness_port)
            self.bioHarness.connection_log = self.connection_log
            self.bioHarness.ConnectBioharness()
            self.status_bioharness.config(text = 'Connected')
            message = f'Connecting BioHarness to port: {self.bioHarness.port} with ID tag {self.entry_tag.get()}...'
            print(message)

        if self.save_realsense_data.get():
            self.streaming_rs = True
            self.realSense = RealSense(connect_rs=True, tag_text=self.entry_tag.get())
            self.realSense.connection_log = self.connection_log
            self.realSense.ConnectRealSense()
            with open(self.connection_log, 'a') as file:
                file.write(f'Realsense-start, {int(datetime.now(timezone.utc).timestamp() * 1000)} \n')
            self.status_realsense.config(text ='Connected' )
            message = f'Connecting RealSense with ID tag {self.entry_tag.get()}'
            print(message)
        
        if self.save_pepper_data.get():
            self.write_pepperState()
            self.pepperConnected = True
            pepper_thread = threading.Thread(target = self.run_pepper)
            pepper_thread.start()
            self.status_pepper.config(text = 'Connected')
            print(f'Pepper connected with ID tag {self.entry_tag.get()}')

    def disconnect(self):
        self.stop_button.config(state = tk.DISABLED)
        self.baseline_button.config(state = tk.DISABLED)
        self.start_video_button.config(state=tk.DISABLED)
        self.start_pepper_button.config(state=tk.DISABLED)
        self.pause_pepper_button.config(state=tk.DISABLED)
        self.streaming = False
        if not self.pepperStop:
            self.pepper_stop()
        time.sleep(2)
        message = '\n Stopping'
        print(message)

        if self.save_realsense_data.get():
            cv2.destroyAllWindows()
            try:
               self.realSense.pipe.stop()
            except:
                pass
            self.status_realsense.config(text = 'Connection stopped')
            message = 'Recording data RealSense stopped'
            print(message)

        if self.save_bioharness_data.get():
            try:
                self.bioHarness.trasm.close()
            except:
                pass
            self.status_bioharness.config(text = 'Connection stopped')
            message = 'Recording data Bioharness stopped'
            print(message)
        
        if self.save_pepper_data.get():
            self.pepper_stop()
            print('Pepper Disconnected')

        self.connect_button.config(state=tk.NORMAL)
        print('Done')
        self.display_id_info(ask = False)
    
    def questionaire(self):
        self.stop_video_button.config(state=tk.NORMAL)
        self.questionaire_ecg_button.config(state = tk.DISABLED)
        self.start_video_button.config(state=tk.DISABLED)
        questionaire_path = os.path.join('Measures', self.entry_tag.get())
        tasks = os.listdir(questionaire_path)
        questionaire_tag = 'Questionaire_1'
        for task in sorted(tasks):
            if 'Questionaire' in task:
                num = int(task.split('_')[1])
                questionaire_tag = f'Questionaire_{num + 1}'
        questionaire_path = os.path.join(questionaire_path, questionaire_tag)
        print(f'Started {questionaire_tag} acquisition')
        self.streaming = True
        if self.save_bioharness_data.get():
            self.bioHarness.save_bioharness_data = True
            self.bioHarness.save_path = os.path.join(questionaire_path, 'BioHarness')
            if os.path.exists(self.bioHarness.save_path):
                shutil.rmtree(self.bioHarness.save_path)
            os.makedirs(self.bioHarness.save_path)
            self.bioHarness.create_paths()
        with open(self.connection_log, 'a') as f:
            f.write(f'{questionaire_tag} started, {int(datetime.now(timezone.utc).timestamp() * 1000)} \n ')
        threading.Thread(target = self.start_bioharness).start()


    def baseline(self):
        tag_text = self.entry_tag.get()
        self.start_video_button.config(state =tk.DISABLED ) 
        # self.stop_video_button.config(state = tk.NORMAL)
        if not tag_text:
            messagebox.showerror('Error', 'Please enter an ID tag')
        tag_list = os.listdir(self.video_folders)
        if 'Baseline' in tag_list:
            sensor_baseline_list = os.listdir(os.path.join(self.video_folders, 'Baseline'))
            sx = self.save_bioharness_data.get()
            x = 'BioHarness' in sensor_baseline_list
            sy = self.save_realsense_data.get()
            y = 'RealSense' in sensor_baseline_list
            flag = (sx and x and not sy) or (sy and y and not sx) or (sx and x and sy and y)
            if flag:
                response = messagebox.askyesno('Warning', f'Baseline of ID {tag_text} already existed. Do you want to overwrite?')
                if not response:
                    return
        print('Acquiring Baseline')
        self.streaming = True
        video_tag = self.video_tag
        self.video_tag = 'Baseline'
        if self.save_realsense_data.get():
            self.realSense.save_rs_data = True
            self.realSense.save_path = os.path.join(self.baseline_path, 'RealSense')
            if os.path.exists(self.realSense.save_path):
                shutil.rmtree(self.realSense.save_path)
            os.makedirs(self.realSense.save_path)
            self.realSense.create_paths()
            self.realSense.ConnectRealSense()
        if self.save_bioharness_data.get():
            self.bioHarness.save_bioharness_data = True
            self.bioHarness.save_path = os.path.join(self.baseline_path, 'BioHarness')
            if os.path.exists(self.bioHarness.save_path):
                shutil.rmtree(self.bioHarness.save_path)
            os.makedirs(self.bioHarness.save_path)
            self.bioHarness.create_paths()
        with open(self.connection_log, 'a') as f:
            f.write(f'Started video {self.video_tag}, {int(datetime.now(timezone.utc).timestamp() * 1000)} \n')
        if self.save_bioharness_data.get():
            threading.Thread(target = self.start_bioharness).start()
        if self.save_realsense_data.get():
            threading.Thread(target = self.start_realsense).start()
        self.master.after(30000, self.stop_video)
        self.master.after(30000, lambda new_video_tag =video_tag: setattr(self, 'video_tag', new_video_tag))

    def start_video(self):
        tag_text = self.entry_tag.get()
        tag_list = os.listdir(self.video_folders)
        if 'Baseline' not in tag_list:
            messagebox.showerror('Error', f'Please acquire a baseline signal of the ID {tag_text}')
            return
        self.stop_video_button.config(state = tk.NORMAL)
        self.start_video_button.config(state = tk.DISABLED)
        self.start_pepper_button.config(state=tk.NORMAL)
        self.display_id_info(ask = False)
        self.streaming = True
        with open(self.connection_log, 'a') as f:
            f.write(f'Started video {self.video_tag}, {int(datetime.now(timezone.utc).timestamp() * 1000)} \n')
        print(f'Starting Video {self.video_tag}, {int(datetime.now(timezone.utc).timestamp() * 1000)} \n')
        if self.save_bioharness_data.get():
            self.bioHarness.save_bioharness_data = True
            self.bioHarness.save_path = os.path.join(self.video_folders, self.video_tag, 'BioHarness')
            self.bioHarness.create_paths()
            threading.Thread(target=self.start_bioharness).start()
        if self.save_realsense_data.get():
            self.realSense.save_rs_data = True
            self.realSense.save_path = os.path.join(self.video_folders, self.video_tag, 'RealSense')
            self.realSense.create_paths()
            self.realSense.ConnectRealSense()
            threading.Thread(target = self.start_realsense).start()

    def start_bioharness(self):
        while self.streaming:
            self.bioHarness.BHReceived()

    def start_realsense(self):
        while self.streaming:
            frame, self.ts = self.realSense.get_data()
            self.display_image(frame)     

    def display_image(self, image):
        frame = np.asanyarray(image)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with open(os.path.join(self.video_folders, 'Log_files', 'current_try.txt'), 'r') as file:
              self.current_try = int(file.read())
            if self.planning_vector[self.current_try+1] == 0 and not self.planning_vector[self.current_try+2] == 0:
               self.videoPlanningText.config(text = f'{self.planning_vector[self.current_try:self.current_try+10]}', fg='red', font = ('Helvetica',12))
            else:
                self.videoPlanningText.config(text = f'{self.planning_vector[self.current_try:self.current_try+10]}', fg='black', font=('Helvetica', 10))
        frame = ImageTk.PhotoImage(Image.fromarray(frame))
        self.frame = frame
        self.videoLabel.configure(image=frame)

    def stop_video(self):
        self.stop_video_button.config(state = tk.DISABLED)
        self.start_video_button.config(state = tk.NORMAL)
        self.questionaire_ecg_button.config(state = tk.NORMAL)
        self.pause_pepper()
        self.start_pepper_button.config(state = tk.DISABLED)
        with open(self.connection_log, 'a') as f:
            f.write(f'Stopped video {self.video_tag}, {int(datetime.now(timezone.utc).timestamp() * 1000)}\n')
        time.sleep(2)
        self.streaming = False
        if self.save_bioharness_data.get():
            self.bioHarness.save_bioharness_data = False
        if self.save_realsense_data.get():
            self.realSense.save_rs_data = False
            try:
               self.realSense.pipe.stop()
            except:
                pass
        print(f'\n Finishing video {self.video_tag}')
        self.display_id_info(ask = False)
        

    def run_pepper(self):
        script = 'Pepper.py'
        self.write_pepperState()
        subprocess.call(['python2', script, '--tag', self.entry_tag.get(), '--typeTask', self.task], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
   
    def start_pepper(self):
        self.start_pepper_button.config(state=tk.DISABLED)
        self.pepperStart = True
        self.write_pepperState()
        print(f'Pepper started at timestamp: {self.get_timestamp()}')
        self.pause_pepper_button.config(state=tk.NORMAL)
  
    def pause_pepper(self):
        self.pause_pepper_button.config(state=tk.DISABLED)
        self.pepperStart = False
        self.write_pepperState()
        print(f'Pepper paused at timestamp: {self.get_timestamp()}')
        self.start_pepper_button.config(state=tk.NORMAL)

    def pepper_stop(self):
        self.pepperStop = True
        self.pepperConnected = False
        self.write_pepperState()
        self.pepperStart = False
        self.pepperStop = False
        self.pepperConnected = False

    def write_pepperState(self):
        with open('pepperState.txt', 'w') as f:
            f.write(f'{int(self.pepperStart)} \n{int(self.pepperStop)} ')
   
    def get_timestamp(self):
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    
# main()
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1600x800')
    form = Form(root)
    root.mainloop()
