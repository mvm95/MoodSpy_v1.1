import os
from typing import List
import serial
import threading
from datetime import datetime, timezone
from utils.visualizedata_bioharness import plot_data
# import time
import struct

BH_MSG_DICT ={
    'CRC' : 0x5E,
    'DLC' : 0x01,
    'DLC_SUMMARY' : 0x02,
    'ETX' : 0x03,
    'GENERAL' : 0x14,
    'BREATHING' : 0x15,
    'ECG' : 0x16,
    'RR' : 0x19,
    'SUMMARY' : 0xBD,
    'ACC' : 0x1E,
    'ACC_100' : 0xBC,
    'PAYLOAD' : 0x01,
    'STX' : 0x02,
    'UpdatePeriodLow' : 0x01,
    'UpdatePeriodHigh' : 0x00
}

class BioHarness():
    def __init__(self,
                tag_text='default_Bioharness',
                save_bioharness_data = True,
                connect_bio=False,
                port = '/dev/rfcomm0'):
        self.trasm = None
        self.tag_text = tag_text # da inserire da form
        self.save_bioharness_data = save_bioharness_data # da inserire da form
        self.connect_bio = connect_bio
        self.stop_bh_event = threading.Event()
        #creo le cartelle nella cartella dove è collocato il file
        # self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Measures") 

        self.file_bio1 = 'BR'
        self.file_bio2 = 'ECG'
        self.file_bio3 = 'SUMMARY'
        self.file_bio4 = 'RR'
        self.file_bio5 = 'GENERAL'
        self.file_bio6 = 'ACC'

        self.port = port
        self.baudrate = 9600
        self.bytesize=8
        self.parity='N'
        self.stopbits=1
        self.timeout=1

        self.isConnected = False
        self.bioharness_lock = threading.Lock()
    def create_paths(self):
        if self.save_bioharness_data:
            # self.dir_log = os.path.join(self.save_path, "Log_Files")
            # self.dir_bio = os.path.join(self.save_path)
            # if not os.path.exists(self.save_path):os.makedirs(self.save_path)
            # if not os.path.exists(self.dir_log):os.makedirs(self.dir_log)
            self.dir_bio = self.save_path
            if not os.path.exists(self.dir_bio):os.makedirs(self.dir_bio)

    def connect_serial(self, port):
        try:
            self.trasm = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout
            )
            print(f"Connected to {self.trasm.name}")
        except Exception as e:
            print(f"Error: {e}")

    def Bioharness_DecodeData(self, start, nloop, mod, buf):
        data = []
        i = start

        for _ in range(1, nloop + 1):
            bits1 = struct.unpack('<H', buf[i:i+2])[0]
            n = bits1 & 0x03FF
            data.append(n)
            i += 1

            bits2 = struct.unpack('<H', buf[i:i+2])[0]
            n = (bits2 >> 2) & 0x03FF
            data.append(n)
            i += 1

            bits3 = struct.unpack('<H', buf[i:i+2])[0]
            n = (bits3 >> 4) & 0x03FF
            data.append(n)
            i += 1

            bits4 = struct.unpack('<H', buf[i:i+2])[0]
            n = (bits4 >> 6) & 0x03FF
            data.append(n)
            i += 2

        if mod >= 1:
            bits5 = struct.unpack('<H', buf[i:i+2])[0]
            n5 = bits5 & 0x03FF
            data.append(n5)
            i += 1

        if mod >= 2:
            bits6 = struct.unpack('<H', buf[i:i+2])[0]
            n6 = (bits6 >> 2) & 0x03FF
            data.append(n6)
            i += 1

        if mod == 3:
            bits6 = struct.unpack('<H', buf[i:i+2])[0]
            n6 = (bits6 >> 4) & 0x03FF
            data.append(n6)

        return data

    def BHReceived(self):
        with self.bioharness_lock:
            try:
                if self.isConnected:
                    bytes_to_read = self.trasm.in_waiting #retrieve number of bytes in the buffer
                    com_buffer = self.trasm.read(bytes_to_read) #create a byte array to hold the awaiting data
                    #  hex_string = ' '.join(f'{byte:02X}' for byte in com_buffer)

                    if self.save_bioharness_data : 
                        if not os.path.exists(os.path.join(self.dir_bio)): os.makedirs(os.path.join(self.dir_bio))
                    if len(com_buffer) >= 5:
                        val1 = str(int.from_bytes(com_buffer[1:3], byteorder='little', signed=True))
                        #GENERAL DATA PACKET
                        if val1 == '13600':
                            convstamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
                            year = int.from_bytes(com_buffer[4:6], byteorder='little', signed=True)
                            month = com_buffer[6]
                            day = com_buffer[7]
                            timestamp = int.from_bytes(com_buffer[8:12], byteorder='little', signed=True)
                            # Heart Rate
                            val3 = int.from_bytes(com_buffer[12:14], byteorder='little', signed=True)
                            # Posture
                            val2 = int.from_bytes(com_buffer[18:20], byteorder='little', signed=True)
                            # Respiration Rate
                            val4 = int.from_bytes(com_buffer[14:16], byteorder='little', signed=True)
                            # Battery Voltage
                            val5 = com_buffer[26]
                            if self.save_bioharness_data: #per quando metto il form
                                with open(f'{self.dir_bio}/{self.file_bio5}.csv', 'a') as file: #se è vuoto
                                    if os.stat(os.path.join(self.dir_bio,f'{self.file_bio5}.csv')).st_size == 0:
                                        file.write('posture, heart rate, respiration rate, battery voltage, year, month, day, ts, timestampnow\n')
                                    file.write(f'{val2}, {val3}, {val4}, {val5}, {year}, {month}, {day}, {timestamp}, {convstamp}\n')
                                # print(f'Bioharness General Data acquired at timestamp {timestamp}')


                        #Breathing waveform packet management
                        if val1 == '8225':
                            convstamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
                            year = int.from_bytes(com_buffer[4:6], byteorder='little', signed=True)
                            bits_month = com_buffer[6]
                            month = int.from_bytes(bits_month.to_bytes(2, byteorder='little'), byteorder='little', signed=True)
                            bits_day = com_buffer[7]
                            day = int.from_bytes(bits_day.to_bytes(2, byteorder='little'), byteorder='little', signed=True)
                            bits_tempo = com_buffer[8:12]
                            timestamp = int.from_bytes(bits_tempo, byteorder='little', signed=True)
                            data_br = self.Bioharness_DecodeData(12, 4, 2, com_buffer) 
                            if self.save_bioharness_data:
                                with open(f"{self.dir_bio}/{self.file_bio1}.csv", "a") as file:
                                    file.write(f"{', '.join(map(str, data_br))}, {year}, {month}, {day}, {timestamp}, {convstamp}\n")
                                # print(f'Breathing waveform Data acquired at timestamp {timestamp}')


                        # GESTIONE ECG PACKET
                        if val1 == "22562":
                            convstamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
                            year = int.from_bytes(com_buffer[4:6], byteorder='little', signed=True)
                            month = com_buffer[6]
                            day = com_buffer[7]
                            timestamp = int.from_bytes(com_buffer[8:12], byteorder='little', signed=True)
                            # Decode ECG data
                            data_ecg = self.Bioharness_DecodeData(12, 15, 3, com_buffer)
                            data_ecg = [ecg *.025 for ecg in data_ecg]

                            if self.save_bioharness_data:
                                with open(f"{self.dir_bio}/{self.file_bio2}.csv", "a") as file:
                                    file.write(', '.join(map(str, data_ecg)) + f", {year}, {month}, {day}, {timestamp}, {convstamp}\n") #corretto data_ecg da data_ecg[:63]
                                # print(f'ECG waveform acquired at timestamp {timestamp}')
                        
                        #Summary data packet management
                        if val1 == "18219":
                            convstamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
                            year = int.from_bytes(com_buffer[4:6], byteorder='little', signed=True)
                            month = com_buffer[6]
                            day = com_buffer[7]
                            # Milliseconds of day
                            timestamp = int.from_bytes(com_buffer[8:12], byteorder='little', signed=True)
                            # Heart Rate
                            val2 = int.from_bytes(com_buffer[13:15], byteorder='little', signed=True)
                            # Respiration Rate
                            val3 = int.from_bytes(com_buffer[15:17], byteorder='little', signed=True)
                            # Posture
                            val4 = int.from_bytes(com_buffer[19:21], byteorder='little', signed=True)
                            # Breathing Wave Amplitude
                            val5 = int.from_bytes(com_buffer[28:30], byteorder='little', signed=True)
                            # Heart Rate Variability
                            val6 = int.from_bytes(com_buffer[38:40], byteorder='little', signed=True)
                            # System Confidence
                            val7 = com_buffer[40]
                            if self.save_bioharness_data:
                                with open(os.path.join(self.dir_bio,f'{self.file_bio3}.csv'), 'a') as file:
                                    if os.stat(os.path.join(self.dir_bio,f'{self.file_bio3}.csv')).st_size == 0: #se è vuoto
                                        file.write('heart rate, respiration rate, posture, breathing wave amplitude, heart rate variability, system confidence, year, month, day, ts, convts\n')
                                    file.write(f"{val2}, {val3}, {val4}, {val5}, {val6}, {val7}, {year}, {month}, {day}, {timestamp}, {convstamp}\n")

                        ## ACC data packet magzement
                        if val1 == '21541':
                            convstamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
                            year = int.from_bytes(com_buffer[4:6], byteorder='little', signed=True)
                            month = com_buffer[6]
                            day = com_buffer[7]
                            timestamp = int.from_bytes(com_buffer[8:12], byteorder='little', signed=True)
                            data_acc = self.Bioharness_DecodeData(12, 15, 0, com_buffer) #5*3
                            if self.save_bioharness_data:
                                with open(os.path.join(self.dir_bio,f'{self.file_bio6}.csv'), 'a') as file:
                                    file.write(f"{', '.join(map(str, data_acc))}, {year}, {month}, {day}, {timestamp}, {convstamp}\n")
                                # print(f'Acceleration data acquired at timestamp {timestamp}')

                        # RR data packet management
                        if val1 == "11556":
                            convstamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
                            year = int.from_bytes(com_buffer[4:6], byteorder='little', signed=True)
                            month = com_buffer[6]
                            day = com_buffer[7]
                            # Milliseconds of day
                            timestamp = int.from_bytes(com_buffer[8:12], byteorder='little', signed=True)
                            # RR data samples
                            rr_samples = [int.from_bytes(com_buffer[i:i+2], byteorder='little', signed=True) for i in range(12, 46, 2)] #corretto 46 da 44
                            if self.save_bioharness_data:
                                with open(f"{self.dir_bio}/{self.file_bio4}.csv", "a") as file:
                                    file.write(f"{', '.join(map(str, rr_samples))}, {year}, {month}, {day}, {timestamp}, {convstamp}\n")
                                # print(f'RR data acquired at timestamp {timestamp}')
                else:
                    self.tram.close()
            except serial.SerialException as e:
                self.trasm.close()
                print(f'Serial Exception: {e}')

    def ConnectBioharness(self):
        if self.connect_bio:
            if not self.trasm:
                self.connect_serial(self.port)
                self.isConnected = True
            try:
                with self.bioharness_lock:
                    if not self.trasm.is_open :
                        self.trasm.open()
                    if self.trasm.is_open:
                        print('Opened trasmission with BioHarness')

                        bytes_to_send4 = bytes([BH_MSG_DICT['STX'], BH_MSG_DICT['BREATHING'], BH_MSG_DICT['DLC'], BH_MSG_DICT['PAYLOAD'], BH_MSG_DICT['CRC'], BH_MSG_DICT['ETX']])
                        self.trasm.write(bytes_to_send4)

                        bytes_to_send3 = bytes([BH_MSG_DICT['STX'], BH_MSG_DICT['ECG'], BH_MSG_DICT['DLC'], BH_MSG_DICT['PAYLOAD'], BH_MSG_DICT['CRC'], BH_MSG_DICT['ETX']])
                        self.trasm.write(bytes_to_send3)

                        bytes_to_send2 = bytes([BH_MSG_DICT['STX'], BH_MSG_DICT['RR'], BH_MSG_DICT['DLC'], BH_MSG_DICT['PAYLOAD'], BH_MSG_DICT['CRC'], BH_MSG_DICT['ETX']])
                        self.trasm.write(bytes_to_send2)

                        bytes_to_send = bytes([BH_MSG_DICT['STX'], BH_MSG_DICT['SUMMARY'], BH_MSG_DICT['DLC_SUMMARY'], BH_MSG_DICT['UpdatePeriodLow'], BH_MSG_DICT['UpdatePeriodHigh'], BH_MSG_DICT['CRC'], BH_MSG_DICT['ETX']])
                        self.trasm.write(bytes_to_send)

                        bytes_to_send5 = bytes([BH_MSG_DICT['STX'], BH_MSG_DICT['GENERAL'], BH_MSG_DICT['DLC'], BH_MSG_DICT['PAYLOAD'], BH_MSG_DICT['CRC'], BH_MSG_DICT['ETX']])
                        self.trasm.write(bytes_to_send5)

                        bytes_to_send6 = bytes([BH_MSG_DICT['STX'], BH_MSG_DICT['ACC'], BH_MSG_DICT['DLC'], BH_MSG_DICT['PAYLOAD'], BH_MSG_DICT['CRC'], BH_MSG_DICT['ETX']])
                        self.trasm.write(bytes_to_send6)
                        # with open(os.path.join(self.dir_log, f'{self.tag_text}_Events.txt'),'a') as file:
                        with open(self.connection_log, 'a') as file:
                            file.write(f'Bioharness_start, {int(datetime.now(timezone.utc).timestamp() * 1000)} \n')
            except Exception as e:
                print(f'Error connecting BioHaness: {e}')
                
    def read_and_save_data(self):
        try:
            while True:
                self.BHReceived()
        # except KeyboardInterrupt:
        #     print(f"Recording stopped. Data saved ")
        except Exception as e:
            print(f"Error: {e}")
        finally:
        #    print(time.time())
            self.trasm.close()
            print(f"Recording stopped. Data saved ")


def main():
    try:
        port = 'COM4'
        bioHarness = BioHarness(save_bioharness_data = True,
                                connect_bio = True,
                                tag_text = 'prova_meeting', 
                                port = port)
        bioHarness.connection_log = os.path.join('Measures',bioHarness.tag_text, 'Connection_log.txt')
        bioHarness.save_path= os.path.join('Measures',bioHarness.tag_text)
        bioHarness.create_paths()
        bioHarness.ConnectBioharness()
        bioHarness.read_and_save_data()
    except KeyboardInterrupt:
        print("Ctrl+C input, closing Bioharness connection")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print('Extracting data')
        plot_data(bioHarness.tag_text, 
                  data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Measures")
                )
        print('Done')

from utils.visualizedata_bioharness import plot_data
if __name__ == '__main__':
    main()
    # for sub in os.listdir('Measures'):
    #     if 'IDU003' in sub:
    #           plot_data(sub, data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Measures"))



