# -*- coding: utf-8 -*-

import sys
import os
from naoqi import ALProxy
import argparse
import qi
import time
import numpy as np
from numpy import dot
from numpy.linalg import norm
from mpmath import csc
from collections import OrderedDict
import almath
import math
import pickle
import glob

#PATH_PKL = 'D:/Dottorato/progetto_AIRCA/report_NAO/Codice_python_NAO/Co-speech_gesture/the_hare_and_the_Tortoes/audio-pkl'
PATH_PKL = 'D:/Dottorato/progetto_AIRCA/report_NAO/Codice_python_NAO/Co-speech_gesture/the_hare_and_the_Tortoes/audio-pkl-ita'
#PATH_PKL = 'D:/Dottorato/progetto_AIRCA/report_NAO/Codice_python_NAO/Co-speech_gesture/Story_audio-pkl'
PATH_WAV = '/home/nao/wav/'

def xyz_to_angle(pose):
    ang = []
    data = pose.copy()

    # quando uso 10 punti
    vec_0_1 = data[1] - data[0]
    vec_1_3 = data[4] - data[1]
    vec_3_4 = data[5] - data[4]
    vec_4_3 = data[4] - data[5]
    vec_4_5 = data[6] - data[5]
    vec_1_6 = data[7] - data[1]
    vec_6_7 = data[8] - data[7]
    vec_7_6 = data[7] - data[8]
    vec_7_8 = data[9] - data[8]

    # left upper torso
    N_1_3_4 = np.cross(vec_1_3, vec_4_3)
    N_0_1_3 = np.cross(vec_1_3, vec_0_1)
    N_3_4_5 = np.cross(vec_4_3, vec_4_5)

    # left shoulder

    Rlut = np.cross(vec_0_1, N_0_1_3)
    QLSP = abs(np.arccos(dot(vec_0_1, np.cross(Rlut, vec_3_4)) / (norm(vec_0_1) * norm(np.cross(Rlut, vec_3_4)))))
    Jlsp = np.arccos(dot(vec_3_4, vec_0_1) / (norm(vec_3_4) * norm(vec_0_1)))  # intermediate variable

    if Jlsp <= np.pi / 2:
        LSP = -QLSP
    else:
        LSP = QLSP

    LSR = np.pi / 2 - np.arccos(dot(vec_3_4, Rlut) / (norm(vec_3_4) * norm(Rlut)))

    if np.rad2deg(LSR) < 0:
        LSR = abs(LSR)

    # left elbow

    Rlua = np.cross(vec_4_3, N_1_3_4)
    QLEY = abs(np.arccos(dot(N_1_3_4, np.cross(vec_4_3, vec_4_5)) / (norm(N_1_3_4) * norm(np.cross(vec_4_3, vec_4_5)))))

    Jley1 = np.arccos(dot(vec_4_5, N_1_3_4) / (norm(vec_4_5) * norm(N_1_3_4)))
    Jley2 = np.arccos(dot(vec_4_5, Rlua) / (norm(vec_4_5) * norm(Rlua)))

    if Jley1 <= np.pi / 2:
        LEY = -QLEY
    elif Jley1 > np.pi / 2 and Jley2 > np.pi / 2:
        LEY = QLEY
    elif Jley1 > np.pi / 2 and Jley2 <= np.pi / 2:
        LEY = QLEY - np.pi * 2

    LER = np.pi - np.arccos(dot(vec_4_5, vec_4_3) / (norm(vec_4_5) * norm(vec_4_3)))

    if np.rad2deg(LER) > 0:
        LER = -LER

    # controll robot’s joint limits

    if np.rad2deg(LSR) >= 89:
        LSR = np.deg2rad(85)
    elif np.rad2deg(LSR) <= 0.5:
        LSR = np.deg2rad(2)

    if np.rad2deg(LER) <= -89:
        LER = np.deg2rad(-85)
    elif np.rad2deg(LER) >= -0.5:
        LER = np.deg2rad(-2)

    if np.rad2deg(LSP) < 0 and np.rad2deg(LSP) <= -119:
        LSP = np.deg2rad(-115)
    elif np.rad2deg(LSP) > 0 and np.rad2deg(LSP) >= 119:
        LSP = np.deg2rad(115)

    if np.rad2deg(LEY) < 0 and np.rad2deg(LEY) <= -119:
        LEY = np.deg2rad(-115)
    elif np.rad2deg(LEY) > 0 and np.rad2deg(LEY) >= 119:
        LEY = np.deg2rad(115)

    # right shoulder
    N_0_1_6 = np.cross(vec_1_6, vec_0_1)
    N_1_6_7 = np.cross(vec_1_6, vec_7_6)

    Rrut = np.cross(vec_0_1, N_0_1_6)
    QRSP = abs(np.arccos(dot(vec_0_1, np.cross(Rrut, vec_6_7)) / (norm(vec_0_1) * norm(np.cross(Rrut, vec_6_7)))))
    Jrsp = np.arccos(dot(vec_0_1, vec_6_7) / (norm(vec_6_7) * norm(vec_0_1)))  # intermediate variable

    if Jrsp <= np.pi / 2:
        RSP = -QRSP
    else:
        RSP = QRSP

    RSR = np.pi / 2 - np.arccos(dot(vec_6_7, Rrut) / (norm(vec_6_7) * norm(Rrut)))

    if np.rad2deg(RSR) > 0:
        RSR = -RSR

    # right elbow

    Rrua = np.cross(vec_7_6, N_1_6_7)
    QREY = abs(
        np.arccos(dot(N_1_6_7, np.cross(vec_7_6, vec_7_8)) / (norm(N_1_6_7) * norm(np.cross(vec_7_6, vec_7_8)))))

    Jrey1 = np.arccos(dot(vec_7_8, N_1_6_7) / (norm(vec_7_8) * norm(N_1_6_7)))
    Jrey2 = np.arccos(dot(vec_7_8, Rrua) / (norm(vec_7_8) * norm(Rrua)))

    if Jrey1 <= np.pi / 2:
        REY = -QREY
    elif Jrey1 > np.pi / 2 and Jrey2 > np.pi / 2:
        REY = QREY
    elif Jrey1 > np.pi / 2 and Jrey2 <= np.pi / 2:
        REY = np.pi * 2 - QREY

    RER = np.pi - np.arccos(dot(vec_7_8, vec_7_6) / (norm(vec_7_8) * norm(vec_7_6)))

    if np.rad2deg(RER) < 0:
        RER = abs(RER)

    # controll robot’s joint limits

    if np.rad2deg(RSR) <= -89:
        RSR = np.deg2rad(-85)
    elif np.rad2deg(RSR) >= -0.5:
        RSR = np.deg2rad(-2)

    if np.rad2deg(RER) >= 89:
        RER = np.deg2rad(85)
    elif np.rad2deg(RER) <= 0.5:
        RER = np.deg2rad(2)

    if np.rad2deg(RSP) < 0 and np.rad2deg(RSP) <= -119:
        RSP = np.deg2rad(-115)
    elif np.rad2deg(RSP) > 0 and np.rad2deg(RSP) >= 119:
        RSP = np.deg2rad(115)

    if np.rad2deg(REY) < 0 and np.rad2deg(REY) <= -119:
        REY = np.deg2rad(-115)
    elif np.rad2deg(REY) > 0 and np.rad2deg(REY) >= 119:
        REY = np.deg2rad(115)

    angles = OrderedDict()
    angles['LShoulderRoll'] = LSR.item()
    angles['LShoulderPitch'] = LSP.item()
    angles['LElbowRoll'] = LER.item()
    angles['LElbowYaw'] = LEY.item()
    angles['RShoulderRoll'] = RSR.item()
    angles['RShoulderPitch'] = RSP.item()
    angles['RElbowRoll'] = RER.item()
    angles['RElbowYaw'] = REY.item()

    # print(angles)

    # print('\n ang', ang)

    return angles

class StoryTelling():

    def __init__(self, args):
        self.tts = ALProxy("ALTextToSpeech", args.ip, args.port)
        self.motion = ALProxy("ALMotion", args.ip, args.port)
        self.posture = ALProxy("ALRobotPosture", args.ip, args.port)
        self.audio = ALProxy("ALAudioPlayer", args.ip, args.port)
        self.tablet = ALProxy("ALTabletService", args.ip, args.port)
        self.alBehaviorManager = ALProxy("ALBehaviorManager", args.ip, args.port)


        self.angle_names = ["LShoulderRoll", "LShoulderPitch", "LElbowRoll", "LElbowYaw", "RShoulderRoll", "RShoulderPitch", "RElbowRoll", "RElbowYaw", "RHand", "LHand", "LWristYaw", "RWristYaw"]
        self.frame_rate = 4

        # First, wake up.
        self.motion.wakeUp()

        # posture standinit gives a more natural position for the robot. instead of being looking to the sky it looks in the direction of the horizon
        self.posture.goToPosture("StandInit", 0.8)

    def cospeech(self, skeletons, vid):

        names = list()  # name of the joints
        times = list()  # time in seconds taken for the movement
        keys = list()  # angular values that the joints assume with time
        frame = list()

        key_LSR = list()
        key_LSP = list()
        key_LER = list()
        key_LEY = list()
        key_RSR = list()
        key_RSP = list()
        key_RER = list()
        key_REY = list()


        for j in range(1, (skeletons['out_poses'].shape[0] / self.frame_rate) + 1):
            frame.append(j * 0.25)

        i = 0
        clip_mean = []
        for f in range(skeletons['out_poses'].shape[0]):

            if skeletons['out_poses'][f, 7, 0] == 0 or skeletons['out_poses'][f, 4, 0] == 0:
                print('no frame')
                continue

            clip_mean.append(skeletons['out_poses'][f])
            i += 1
            if i == self.frame_rate:
                mean_frame = np.asarray(clip_mean, dtype=np.float16)
                mean_frame = np.mean(mean_frame, axis=0)
                angles = xyz_to_angle(mean_frame)
                del clip_mean
                clip_mean = []
                i = 0
                key_LSR.append(angles['LShoulderRoll'])
                key_LSP.append(angles['LShoulderPitch'])
                key_LER.append(angles['LElbowRoll'])
                key_LEY.append(angles['LElbowYaw'])
                key_RSR.append(angles['RShoulderRoll'])
                key_RSP.append(angles['RShoulderPitch'])
                key_RER.append(angles['RElbowRoll'])
                key_REY.append(angles['RElbowYaw'])

            if ((f == skeletons['out_poses'].shape[0]) and (i != self.frame_rate)):
                mean_frame = np.asarray(clip_mean, dtype=np.float16)
                mean_frame = np.mean(mean_frame, axis=0)
                angles = xyz_to_angle(mean_frame)
                print('passato')
                del clip_mean
                clip_mean = []
                i = 0
                key_LSR.append(angles['LShoulderRoll'])
                key_LSP.append(angles['LShoulderPitch'])
                key_LER.append(angles['LElbowRoll'])
                key_LEY.append(angles['LElbowYaw'])
                key_RSR.append(angles['RShoulderRoll'])
                key_RSP.append(angles['RShoulderPitch'])
                key_RER.append(angles['RElbowRoll'])
                key_REY.append(angles['RElbowYaw'])

        leftArmEnable = False
        rightArmEnable = False
        self.motion.setMoveArmsEnabled(leftArmEnable, rightArmEnable)  # disables autonomous motions for the robot's arms

        names.append("LElbowRoll")
        times.append(frame)
        keys.append(key_LER)

        names.append("LElbowYaw")
        times.append(frame)
        keys.append(key_LEY)

        names.append("LShoulderPitch")
        times.append(frame)
        keys.append(key_LSP)

        names.append("LShoulderRoll")
        times.append(frame)
        keys.append(key_LSR)

        names.append("RElbowRoll")
        times.append(frame)
        keys.append(key_RER)

        names.append("RElbowYaw")
        times.append(frame)
        keys.append(key_REY)

        names.append("RShoulderPitch")
        times.append(frame)
        keys.append(key_RSP)

        names.append("RShoulderRoll")
        times.append(frame)
        keys.append(key_RSR)

        # audio_player_service.playFile("D:/Dottorato/progetto_AIRCA/report_NAO/Codice_python_NAO/Co-speech_gesture/i-icXZ2tMRM_709_5_000_0.wav")
        self.motion.post.angleInterpolation(names, keys, times, True)  # angle intespolation is the method that will give life to the movement
        self.audio.playFile(PATH_WAV + vid + '.wav')
        stiffnesses = 1.0
        self.motion.setStiffnesses(names, stiffnesses)  # stiffness make the robot keep the last position achieved without any unvoluntary autonomous movement

        '''
        time.sleep(0.5)
        self.posture.goToPosture("StandInit", 0.5)

        
        for an in self.angle_names:
            angle_reset = 0.
            if an == 'LShoulderPitch' or an == 'RShoulderPitch':
                angle_reset = angle_reset + np.pi / 2
            if an == 'LHand' or an == 'RHand':
                angle_reset = 0.50
            self.motion.setAngles(an, angle_reset, 0.05)
        '''
    '''
    def _getAbsoluteUrl(self, partial_url):
        
        subPath = os.path.join(self.packageUid(), os.path.normpath(partial_url).lstrip("\\/"))
        # We create TabletService here in order to avoid
        # problems with connections and disconnections of the tablet during the life of the application
        return "http://%s/apps/%s" % (self._getTabletService().robotIp(), subPath.replace(os.path.sep, "/"))
    '''
    def showimage(self):
        try:
            # Display a local image located in img folder in the root of the web server
            # The ip of the robot from the tablet is 198.18.0.1
            self.tablet.hideImage()
            time.sleep(3)

            self.alBehaviorManager.runBehavior('simple-tabletpage-2edde8' + '/behavior_1')



        except Exception as e:
            print("Error was: ", e)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    gesture = StoryTelling(args)
    gesture.showimage()



    files = os.listdir(PATH_PKL)

    files_without_extension = [os.path.splitext(file)[0] for file in files]
    files_without_extension.sort()

    unique_file_names = set()


    for filename in files_without_extension:
        if filename not in unique_file_names:
            unique_file_names.add(filename)
            print(filename)

            pickle_file = glob.glob(PATH_PKL + '/' + filename + '.pkl')

            if pickle_file:
                with open(pickle_file[0], 'rb') as file:
                    skeleton = pickle.load(file)

            gesture.cospeech(skeleton, filename)
            time.sleep(1)


    gesture.posture.goToPosture("StandInit", 0.5)
    gesture.tablet.hideImage()