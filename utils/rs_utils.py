import os 
# import pyrealsense2 as rs
import pandas as pd
import numpy as np
import rosbag
import cv2
from utils.visualizedata_bioharness import extract_pepper_timestamps

MEASURE_PATH = '/home/marco/Desktop/Codes/MoodSpy/Measures'

SKELETON_JOINTS = {
    0: 'NOSE',
    1: 'LEFT EYE (INNER)',
    2: 'LEFT EYE',
    3: 'LEFT EYE (OUTER)',
    4: 'RIGHT EYE (INNER)',
    5: 'RIGHT EYE',
    6: 'RIGHT EYE (OUTER)',
    7: 'LEFT EAR',
    8: 'RIGHT EAR',
    9: 'MOUTH (LEFT)',
    10: 'MOUTH (RIGHT)',
    11: 'LEFT SHOULDER',
    12: 'RIGHT SHOULDER',
    13: 'LEFT ELBOW',
    14: 'RIGHT ELBOW',
    15: 'LEFT WRIST',
    16: 'RIGHT WRIST',
    17: 'LEFT PINKY',
    18: 'RIGHT PINKY',
    19: 'LEFT INDEX',
    20: 'RIGHT INDEX',
    21: 'LEFT THUMB',
    22: 'RIGHT THUMB',
    23: 'LEFT HIP',
    24: 'RIGHT HIP',
    25: 'LEFT KNEE',
    26: 'RIGHT KNEE',
    27: 'LEFT ANKLE',
    28: 'RIGHT ANKLE',
    29: 'LEFT HEEL',
    30: 'RIGHT HEEL',
    31: 'LEFT FOOT INDEX',
    32: 'RIGHT FOOT INDEX',
}
from mediapipe.framework.formats import landmark_pb2

def write_skeleton_data(file_name, landmarks):
    with open(file_name, 'a') as file:
        if os.stat(file_name).st_size == 0:  # Se il file è vuoto
            file.write('Joint,x-norm,y-norm,z-norm,confidence\n')
        for idx, landmark in enumerate(landmarks.landmark):
            file.write(f'{idx}, {landmark.x}, {landmark.y}, {landmark.z}, {landmark.visibility}\n')

def get_topic_table(reader):
    info = reader.get_type_and_topic_info()

    topic_tuple = info.topics.values()
    topics = info.topics.keys()

    message_types = [t1.msg_type for t1 in topic_tuple]
    n_messages = [t1.message_count for t1 in topic_tuple]
    frequency = [t1.frequency for t1 in topic_tuple]

    topic_table = pd.DataFrame(list(zip(topics, message_types, n_messages, frequency)), columns=['Topics', 'Types', 'Message Count', 'Frequency'])
    return topic_table

def depth_to_pc(data, intrinsic_matrix, depth_scale = 1):
    height, width = data.shape

    u,v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    data = data.flatten() * depth_scale
    hom_coord = np.vstack((u,v,np.ones_like(u)))

    points = np.linalg.inv(intrinsic_matrix) @ hom_coord * data
    points = points.T
    return points

# from visualizedata_bioharness import extract_pepper_timestamps
def create_video(subject):
    save_path= f'/home/marco/Desktop/Codes/MoodSpy/Measures/{subject}/RealSense/'
    clip_path = os.path.join(save_path,'clips')
    if not os.path.exists(clip_path):
        os.makedirs(clip_path)

    bagName = os.path.join(save_path,f'{subject}.bag')
    reader = rosbag.Bag(bagName)
    topic_table = get_topic_table(reader)
    topic_table = topic_table[(topic_table['Types'].str.contains('sensor_msgs')) & (topic_table['Message Count'] > 1)]
    topics = list(topic_table['Topics'])
    topics = [top for top in topics if 'Color' in top]

    pepper_log_file = os.path.join(MEASURE_PATH,subject,'Log_Files',f'{subject}_pepper.txt')
    # print(pepper_log_file)
    timestamps = extract_pepper_timestamps(pepper_log_file)
    timestamps = np.asanyarray(timestamps)
    time_table = timestamps.reshape(-1,2)
    # print(time_table)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    count = 1
    for clips in time_table:
        frames = []
        for topic, msg, t in reader.read_messages(topics=topics):
            data = msg.data
            timestamp = (int(msg.header.stamp.to_nsec()/1000000))       
            width = msg.width
            height = msg.height
            # print(clips[0], timestamp, clips[1])
            # print(clips[1] - timestamp)
            if timestamp >= clips[0] - 2000 and timestamp <= clips[1] + 2000:
                # print('im here')
                img = np.frombuffer(data, dtype = np.uint8).reshape((height,width,3))    
                # img = cv2.imread(img)

                text = f'Timestamp: {timestamp}'
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = text_size[1] + 10  # Adjust the value according to your preference

                # Add the text to the image
                if timestamp >= clips[0] and timestamp <= clips[1] :
                    color = (0,255,0)
                else:
                    color = (255,255,255)
                cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
                cv2.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness - 1)


                # Display the image
                # cv2.imshow('Image with Text', img)
                # cv2.waitKey(1)
                frames.append(img)
        cv2.destroyAllWindows()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
        video= cv2.VideoWriter(os.path.join(clip_path,f'clip_{count}.mp4'), fourcc, 5, (width, height))  # Adjust the frame rate (fps) as needed

        # Write each frame to the video
        for frame in frames:
            video.write(frame)

        # Release the video writer and close all OpenCV windows
        video.release()
        cv2.destroyAllWindows()
        count += 1




def post_processing_extraction(subject):
    save_path= f'/home/marco/Desktop/Codes/MoodSpy/Measures/{subject}/RealSense/'
    bagName = os.path.join(save_path,f'{subject}.bag')
    reader = rosbag.Bag(bagName)
    topic_table = get_topic_table(reader)
    topic_table.to_csv(os.path.join(save_path, f'{subject}_topics.csv'))
    depth_info = topic_table[(topic_table['Topics'].str.contains('Depth')) & (topic_table['Topics'].str.contains('camera_info'))]
    depth_info = list(depth_info['Topics'])
    for top, msg, t in reader.read_messages(topics=depth_info):
        intrinsic_matrix = msg.K
    intrinsic_matrix = np.asanyarray(intrinsic_matrix).reshape((3,3))
    topic_table = topic_table[(topic_table['Types'].str.contains('sensor_msgs')) & (topic_table['Message Count'] > 1)]
    topics = list(topic_table['Topics'])

    for topic, msg, t in reader.read_messages(topics = topics):
        timestamp = (msg.header.stamp.to_nsec())
        frame_id = msg.header.seq
        img_data = msg.data
        width = msg.width
        height = msg.height
    
        if 'Color' in topic:
            img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
            cv2.imwrite(os.path.join(save_path,subject,'Color',f'FrameID_{frame_id}_TimeStamp_{timestamp}.png'), img_array)
            print(f'Saved color information at frame {frame_id}')
        
        if 'Depth' in topic:
            img_array = np.frombuffer(img_data, dtype = np.uint16).reshape((height,width))
            cv2.imwrite(os.path.join(save_path,subject,'Depth',f'FrameID_{frame_id}_TimeStamp_{timestamp}.png'), img_array)
            points = depth_to_pc(img_array, intrinsic_matrix)
            # points = self.pc(get_depth_frame(img_array))
            # points = self.pc.calculate(img_data)
            # vert, text = points.get_vertices(), points.get_texture_coordinates()
            # points_data = np.asanyarray(vert).view(np.float32).reshape(-1,3)
            # texcoord = np.asanyarray(text).view(np.float32).reshape(-1,2)
            np.save(os.path.join(save_path,subject,'PointCloud',f'frameID_{frame_id}_TimeStamp_{timestamp}.npy'), points)
            print(f'Saved depth information at frame {frame_id}')

        if 'Infrared_1' in topic:
            img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
            cv2.imwrite(os.path.join(save_path,subject,'InfraRed',f'FrameID_{frame_id}_TimeStamp_{timestamp}_IR1.png'), img_array)
            print(f'Saved ir1 information at frame {frame_id}')

        if 'Infrared_2' in topic:
            img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
            cv2.imwrite(os.path.join(save_path,subject,'InfraRed',f'FrameID_{frame_id}_TimeStamp_{timestamp}_IR2.png'), img_array)
            print(f'Saved ir2 information at frame {frame_id}')

from tqdm import tqdm
def main():
    for sub in tqdm(sorted(os.listdir('/home/marco/Desktop/Codes/MoodSpy/Measures'))):
        print(sub)
        if 'IDU006' in sub or 'IDU007' in sub:
        # if not os.path.exists(f'/home/marco/Desktop/Codes/MoodSpy/Measures/{sub}/RealSense/clips/clip_1.mp4'):
            try:
                create_video(sub)
            except Exception as e:
                print(e)
    # create_video('IDU001V001')

if __name__ == '__main__':
    main()
# def write_skeleton_data(file_name, keypoints):
#     with open(file_name, 'a') as file:
#         if os.stat(file_name).st_size == 0: #se è vuoto
#             file.write('Joint, x-norm, y-norm, z-norm, confidence \n')
#         for idx, joint in enumerate(keypoints):
#             file.write(f'{SKELETON_JOINTS[idx]}, {joint["x"]}, {joint["y"]}, {joint["z"]}, {joint["visibility"]} \n')


# def write_skeleton_data(file_name, keypoints):
#     # Rimuovi gli spazi vuoti dalla stringa di input
#     data_string = keypoints.landmark
#     data_string = data_string.replace(" ", "")

#     # Estrai i valori di x, y, z, visibility dalla stringa
#     values = data_string.split(',')
#     values = [v for v in values if v]  # Rimuovi eventuali elementi vuoti

#     # Inizializza una lista vuota per i dati del joint
#     joint_data = []

#     # Raggruppa i valori in gruppi di 4 (x, y, z, visibility) e aggiungi a joint_data
#     for i in range(0, len(values), 4):
#         joint_data.append({
#             "x": float(values[i + 1]),
#             "y": float(values[i + 2]),
#             "z": float(values[i + 3]),
#             "visibility": float(values[i + 4])
#         })

#     # Scrivi i dati nel file specificato
#     with open(file_name, 'a') as file:
#         if os.stat(file_name).st_size == 0:  # Se il file è vuoto
#             file.write('Joint, x-norm, y-norm, z-norm, confidence\n')
#         for idx, joint in enumerate(joint_data):
#             file.write(f'{idx + 1}, {joint["x"]}, {joint["y"]}, {joint["z"]}, {joint["visibility"]}\n')

# # # Esempio di utilizzo
# # input_string = "[x: 0.65941566y: 0.051322997z: -0.33657226visibility: 0.99414176, ...]"  # Inserisci la tua stringa completa qui
# # output_file = "output.txt"
# # write_skeleton_data(output_file, input_string)
