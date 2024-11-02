import sys
sys.path.append('C:\Users\pynaoqi\lib')
import qi
# from naoqi import ALProxy as Proxy
import time
import argparse
from datetime import datetime
import os
import random
import threading

#Behavior in Pepper are saved in the opposite way
MOVEMENT_DICTIONARY = {
  'leftUp' : 'rightUp',
  'leftDown' : 'rightDown',
  'rightUp' : 'leftUp',
  'rightDown' : 'leftDown'
}
def get_timestamp():
  # utc_now = datetime.utcnow()
  timestamp = int(round(time.time() * 1000))
  return timestamp

class GoNogo(object):
  def __init__(self, IP = '192.168.1.8', PORT = 9559, tag_text='test', taskType= 'Stop', test = False):
    super(GoNogo, self).__init__()
    self.IP = IP
    self.PORT = PORT
    self.tag_text = tag_text
    self.taskType = taskType
    self.session = qi.Session()
    self.session.connect('tcp://'+IP+':'+str(PORT))
    self.led = self.session.service('ALLeds')
    self.tts = self.session.service('ALTextToSpeech')
    self.tts.say('Mi sono connesso!')
    self.behavior = self.session.service('ALBehaviorManager')
    self.autonomous = self.session.service('ALAutonomousLife')
    self.motion = self.session.service('ALMotion')
    self.posture = self.session.service('ALRobotPosture')
    if not self.autonomous.getState() == 'disabled':
      self.autonomous.setState('disabled')
    self.posture.goToPosture('StandInit', 1)
    self.started = False
    self.stopped = False
    self.dir_path = os.path.join('Measures', self.tag_text, self.taskType, 'Log_files')
    if not os.path.exists(self.dir_path):
      os.makedirs(self.dir_path)
    self.dir_file = os.path.join(self.dir_path, tag_text + '_pepper.txt')
    self.get_num_task()
    self.get_planningVector_and_currentItem()
    self.read_state()
    if test:
      self.get_behavior('rightDown')
      self.get_behavior('rightUp')
      self.get_behavior('leftDown')
      self.get_behavior('leftUp')
    self.get_behavior('blue')

    self.pepper_lock = threading.Lock()

  def get_behavior(self, behavior):
    self.behavior.runBehavior('marco_destini-e4dbc5/' + behavior)
    if behavior in ['red', 'blue', 'yellow', 'green']:
      self.color = behavior
      if self.color == 'blue':
        self.write('Pepper in \'Blue\' state, current try: ' + str(self.current_try) +'\n')
    else:
      self.behavior.stopBehavior('marco_destini-e4dbc5/' + behavior)

  def run_movement(self):
    self.get_behavior('red')
    time.sleep(0.5) 
    movement_list = ['rightUp', 'rightDown', 'leftUp', 'leftDown']
    choice = random.randint(0,3)
    movement = movement_list[choice]
    print movement
    self.get_behavior(movement)
    movement = MOVEMENT_DICTIONARY[movement]
    return movement
  
  def get_num_task(self):
    self.num_task = 0
    if os.path.exists(self.dir_file):
      with open(self.dir_file, 'r') as f:
        for l in f:
          if l.startswith('Stop task from') or l.startswith('Movement task from'):
            self.num_task += 1
  
  def getRunTime(self):
    return random.uniform(1, 2.5)

  def getStopTime(self):
    return random.uniform(2, 3.5)

  def write(self, text):
    # log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Measures', self.tag_text, 'Log_Files', self.tag_text + '_pepper.txt')
    log_file = os.path.join(self.dir_path, self.tag_text + '_pepper.txt')
    with open(log_file, 'a') as file:
      file.write(text)

  def read_state(self):
    with open('pepperState.txt', 'r') as file:
      lines = file.readlines()
      self.started = bool(int(lines[0].strip()))
      self.stopped = bool(int(lines[1].strip()))
      file.close()

  def get_planningVector_and_currentItem(self):
    planning_vector_file = os.path.join(self.dir_path, 'planning_vector.txt')
    with open(planning_vector_file, 'r') as f:
      self.planning_vector = f.readlines()
    self.planning_vector = [int(line.strip()) for line in self.planning_vector]
    current_try_file = os.path.join(self.dir_path, 'current_try.txt')
    with open(current_try_file, 'r') as f:
      self.current_try = int(f.read())
  
  def update_currentItem(self):
    current_try_file = os.path.join(self.dir_path, 'current_try.txt')
    with open(current_try_file, 'w') as f:
      f.write(str(self.current_try))
      
  def run_exercise(self,choice):
    if choice == 0:
      t1 = get_timestamp()
      self.get_behavior('green')
      time.sleep(self.getRunTime())
      t2 = get_timestamp()
      text = 'Go task from timestamps: ' + str(t1) + ' - ' + str(t2) + '\n'
      self.write(text)
      self.get_behavior('yellow')
      time.sleep(0.1)
    elif choice == 1:
      # self.get_behavior('green')
      # time.sleep(self.getRunTime())
      # self.get_behavior('yellow')
      # time.sleep(0.1)
      t1 = get_timestamp()
      self.get_behavior('red')
      time.sleep(self.getStopTime())
      t2 = get_timestamp()
      text = 'Stop task from timestamps: ' + str(t1) + ' - ' + str(t2) + '\n'
      self.write(text)
      self.read_state()
      self.num_task += 1
      self.get_behavior('yellow')
      time.sleep(0.1)
    elif choice == 2:
      # self.get_behavior('green')
      # time.sleep(self.getRunTime())
      # self.get_behavior('yellow')
      # time.sleep(0.1)
      t1 = get_timestamp()
      # self.get_behavior('red')
      # time.sleep(0.5)
      movement = self.run_movement()
      t2 = get_timestamp()
      text = 'Movement task from timestamps: ' + str(t1) + ' - ' + str(t2) + ',' + movement + '\n'
      self.write(text)
      self.read_state()
      self.num_task += 1   
      self.get_behavior('yellow')
      time.sleep(0.1)

  def start(self):
    self.read_state()
    first_istance = True
    with self.pepper_lock:
      print 'Connected'
      text = 'Pepper connect at ' + str(get_timestamp()) + '\n'
      self.write(text)
      try:
        while True:
          if not self.started:
            if not first_istance:
              first_istance = True
            if first_istance:
              time.sleep(0.3)
            self.read_state()
            if not self.color == 'blue':
              self.get_behavior('blue')
          else:
            # if self.current_try >= len(self.planning_vector):
            if self.current_try > 77:
              self.tts.setParameter('speed', 10)
              self.tts.say('ABBIAMO FINITO, GRAZIE!')
              break
            if first_istance:
              first_istance = False
            self.run_exercise(self.planning_vector[self.current_try])
            self.current_try += 1
            self.update_currentItem()
          if self.stopped:
            break
          self.read_state()
      except Exception as e:
        self.tts.setParameter('speed', 10)
        self.tts.setLanguage( 'English')
        self.tts.say(str(e))
        self.tts.setLanguage( 'Italian')
        print(e)
      finally:
        self.write('------------------------- \n')
        self.write('Current try: ' + str(self.current_try) +'\n')
        self.write('------------------------- \n')
        self.behavior.stopAllBehaviors()
        self.get_behavior('blue') 
        print 'Done'
  
def main():
    gonogo = GoNogo(test=True)
    # gonogo.start()

if __name__ == '__main__':
  # main()
  parser = argparse.ArgumentParser()
  parser.add_argument('--ip', type = str, default= '192.168.1.7',
                      help = 'Pepper ID address. Default is 192.168.1.7')
  # parser.add_argument('--ip', type = str, default= '169.254.212.6',
  #                     help = 'Pepper ID address. Default is 192.168.1.7')
  parser.add_argument('--port', type = int, default= 9559,
                      help = 'Pepper port. Default is 9559')
  parser.add_argument('--tag', type = str, default= 'test',
                      help = 'Experiment tag text, default is \'test\'')
  parser.add_argument('--typeTask', type = str, default= 'Movement',
                      help = 'Experiment type task, \' Stop \' or \' Movement \',  default is \' Movement \'')
  args = parser.parse_args()
  # args.ip ='192.168.1.120'
  try:
    gonogo = GoNogo(IP = args.ip, PORT = args.port, tag_text=args.tag, taskType=args.typeTask)
  except RuntimeError:
    args.ip ='192.168.1.8' #Pepper 2
    gonogo = GoNogo(IP = args.ip, PORT = args.port, tag_text=args.tag, taskType=args.typeTask)
  # gonogo = GoNogo(IP = '127.0.0.1', PORT=56782)

  gonogo.start()