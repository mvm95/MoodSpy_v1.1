import os
import random
import pandas as pd
import pygame
from utils.experiment_utils import create_video_planning
import sys
from datetime import datetime, timezone

pygame.init()
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
# SCREEN = pygame.display.set_mode((800, 600))
pygame.display.set_caption("PsyGame")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BACKGROUND = ( 255, 190, 152)
NUM_SCREENS =  77
NUM_RED_SCREENS = 16
STIMULUS_DISPLAY_TIME = 1000
FIXATION_TIME_GONOGO = 750
FIXATION_TIME_STROOP = 500
INTERVAL_TRIALS = 20

SCREENS = {
    0 : GREEN,
    1 : RED
}

COLOR_DICT = {
    'ROSSO' : RED,
    'BLU' : BLUE,
    'GIALLO' : YELLOW,
    'VERDE' : GREEN,
    'NERO' : BLACK,
    'BIANCO' : WHITE
}

STROOP_DICT = {
    0 : 'ROSSO',
    1 : 'VERDE',
    2 : 'GIALLO',
    3 : 'BLU'
}

KEY_DICT_STROOP = {
    'r' : 'ROSSO',
    'v' : 'VERDE',
    'g' : 'GIALLO',
    'b' : 'BLU'
}

TRIALS = {
    0 : 'Go',
    1 : 'NoGo'
}

BUTTON_COLOR_INACTIVE = (200, 200, 200)
BUTTON_COLOR_ACTIVE = (150, 150, 150)
BUTTON_TEXT_COLOR = (0, 0, 0)

class Button:
    def __init__(self, text, x, y, width, height, action):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.action = action
        self.rect = pygame.Rect(x, y, width, height)
        self.rect_border = pygame.Rect(x+5, y+5, width+10, height+10)
        self.color = BUTTON_COLOR_INACTIVE

    def draw(self):
        pygame.draw.rect(SCREEN, BLACK, self.rect_border)
        pygame.draw.rect(SCREEN, self.color, self.rect)
        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        SCREEN.blit(text_surface, text_rect)

    def handle_event(self, event):
        ret = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                ret = self.action()
        return ret

def exit_game():
    return pygame.quit()

def exit_flag(event):
    flag1 = event.type == pygame.QUIT
    flag2 = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
    return flag1 or flag2

def draw_input_box(font, text, x, y, width, height, content):
    input_box = pygame.Rect(x, y, width, height)
    pygame.draw.rect(SCREEN, BLACK, input_box, 2)
    draw_text(text, font, BLACK, x - 20, y - 30)
    return input_box

def draw_text(text, font, color, x, y):
    # text_surface = font.render(text, True, color)
    # text_rect = text_surface.get_rect()
    # text_rect.center = (x, y)
    # SCREEN.blit(text_surface, text_rect)
    words = text.split()
    total_width = sum(font.size(word)[0] for word in words) + font.size(" ")[0] * (len(words) - 1)
    current_x = x - total_width / 2
    rects = []
    for word in words:
        current_color = COLOR_DICT[word] if word in COLOR_DICT else color
        word_surface_black = font.render(word + ' ', True, BLACK)
        word_rect_black = word_surface_black.get_rect()
        word_rect_black.topleft = (current_x + 1, y + 1)  
        SCREEN.blit(word_surface_black, word_rect_black)
        word_surface = font.render(word +' ', True, current_color)
        word_rect = word_surface.get_rect()
        word_rect.topleft = (current_x, y)
        SCREEN.blit(word_surface, word_rect)
        current_x += word_rect.width# + font.size(" ")[0]
        rects.append(word_rect)
    return rects

def draw_gonogo_epileptic(type_trial):
    letter = 'X' if type_trial == 'NoGo' else 'O'
    font = pygame.font.Font(None, 100)
    x = SCREEN_WIDTH // 2
    y = SCREEN_HEIGHT // 2
    text_surface = font.render(letter, True, WHITE)
    SCREEN.blit(text_surface, (x,y))

def show_fixation(time, color = WHITE):
    SCREEN.fill(color)
    pygame.display.flip()
    pygame.time.delay(time)

def draw_stroop_color(color_tuple):
    font = pygame.font.Font(None, 40)
    x = SCREEN_WIDTH // 2
    y = SCREEN_HEIGHT // 2
    word = STROOP_DICT[color_tuple[0]]
    color = COLOR_DICT[STROOP_DICT[color_tuple[1]]]
    text_surface = font.render(word, True, color)
    SCREEN.blit(text_surface, (x, y))

def input_screen():
    SCREEN.fill(BACKGROUND)
    font = pygame.font.Font(None, 36)
    outer_box = pygame.Rect((SCREEN_WIDTH // 3 + 20) - 10, (SCREEN_HEIGHT//2 - 40) - 10 , 1000 + 20, 43 + 20)
    input_box = pygame.Rect(SCREEN_WIDTH // 3 + 20, SCREEN_HEIGHT//2 - 40 , 1000, 43)
    color = BLACK
    active = False
    text = ''
    done = False
    while not done:
        for event in pygame.event.get():
            if exit_flag(event):
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = True
                else:
                    active = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    done = True
                    break
                if active:
                    if event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
        SCREEN.fill(BACKGROUND)
        draw_text('INSERT USER ID', font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT//2 - 100)
        pygame.draw.rect(SCREEN, color, outer_box, 15)
        pygame.draw.rect(SCREEN, WHITE, input_box, 0, 2)
        txt_surface = font.render(text, True, color)    
        width = max(350, txt_surface.get_width() + 20)
        input_box.w = width
        outer_box.w = max(370, txt_surface.get_width() + 40)
        text_rect = txt_surface.get_rect(center=input_box.center)
        SCREEN.blit(txt_surface, text_rect)
        pygame.display.flip()
        # pygame.time.Clock().tick(30)
    return text

def GoNogo(user_id = None):
    if not user_id:
        user_id = input_screen()
    font = pygame.font.Font(None, 30)
    SCREEN.fill(BACKGROUND)
    draw_text("Premi la BARRA SPAZIATRICE quando vedi lo schermo VERDE , non premere NULLA quando vedi lo schermo ROSSO", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    draw_text("Premi la BARRA SPAZIATRICE per iniziare il gioco", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)
    pygame.display.flip()
    planned_vector = create_video_planning()
    del planned_vector[-4:]
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if exit_flag(event):
                running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
    index = 0
    running = True
    columns = ['TRIAL', 'STIMULUS', 'RESPONSE', 'CORRECT', 'REACTION TIME', 'TIMESTAMP']
    df =  []
    type_trial = 'None'
    response_type = 'None'
    response_time = 'None'
    while running and index <NUM_SCREENS:
        for event in pygame.event.get():
            if exit_flag(event):
                running = False
                break
                # pygame.quit()
                # sys.exit()
        SCREEN.fill(BLACK)
        pygame.display.flip()
        show_fixation(FIXATION_TIME_GONOGO)
        color = SCREENS[planned_vector[index]] 
        SCREEN.fill(color)
        pygame.display.flip()
        response = False
        type_trial = TRIALS[planned_vector[index]]
        start_time = pygame.time.get_ticks() 
        while not response:
            current_time = pygame.time.get_ticks()
            if current_time - start_time >= STIMULUS_DISPLAY_TIME:
                timestamp =  int(datetime.now(timezone.utc).timestamp() * 1000)
                response = True
                response_time = -1
                response_type = 'not answered'
            else:
                for event in pygame.event.get():
                    if exit_flag(event):
                        running = False
                        response = True
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        timestamp =  int(datetime.now(timezone.utc).timestamp() * 1000)
                        response = True
                        response_time = current_time - start_time
                        response_type = 'key pressed'
        index += 1
        correct = 1 if (type_trial == 'Go' and response_type == 'key pressed') or (type_trial == 'NoGo' and response_type == 'not answered') else 0
        data = {
            'TRIAL' : index,
            'STIMULUS' : type_trial,
            'RESPONSE' : response_type,
            'CORRECT' : correct,
            'REACTION TIME' : response_time,
            'TIMESTAMP' : timestamp
        }
        df.append(data)
        print(f'Trial {index}, {type_trial} : {response_type}')
        pygame.time.delay(INTERVAL_TRIALS)
    save_path = os.path.join('Measures', user_id)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(os.path.join(save_path,'GoNogo_results.csv'), index=False)
    SCREEN.fill(WHITE)
    draw_text("THE END", font, (0, 0, 0), SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    pygame.display.flip()
    pygame.time.delay(2000) 
    # pygame.quit()

def epileptic_GoNoGO(user_id = None):
    if user_id == None:
        user_id = input_screen()
    font = pygame.font.Font(None, 30)
    SCREEN.fill(BACKGROUND)
    draw_text("Premi la BARRA SPAZIATRICE quando vedi la lettera O , non premere NULLA quando vedi la lettera X", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    draw_text("Premi la BARRA SPAZIATRICE per iniziare il gioco", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)
    pygame.display.flip()
    planned_vector = create_video_planning()
    del planned_vector[-4:]
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if exit_flag(event):
                running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
    while waiting:
        for event in pygame.event.get():
            if exit_flag(event):
                running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
    index = 0
    running = True
    columns = ['TRIAL', 'STIMULUS', 'RESPONSE', 'CORRECT', 'REACTION TIME', 'TIMESTAMP']
    df =  []
    type_trial = 'None'
    response_type = 'None'
    response_time = 'None'
    while running and index <NUM_SCREENS:
        for event in pygame.event.get():
            if exit_flag(event):
                running = False
                break
                # pygame.quit()
                # sys.exit()
        pygame.display.flip()
        show_fixation(FIXATION_TIME_GONOGO, color=BLACK)
        type_trial = TRIALS[planned_vector[index]]
        draw_gonogo_epileptic(type_trial)
        pygame.display.flip()
        response = False
        start_time = pygame.time.get_ticks() 
        while not response:
            current_time = pygame.time.get_ticks()
            if current_time - start_time >= STIMULUS_DISPLAY_TIME:
                timestamp =  int(datetime.now(timezone.utc).timestamp() * 1000)
                response = True
                response_time = -1
                response_type = 'not answered'
            else:
                for event in pygame.event.get():
                    if exit_flag(event):
                        running = False
                        response = True
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        timestamp =  int(datetime.now(timezone.utc).timestamp() * 1000)
                        response = True
                        response_time = current_time - start_time
                        response_type = 'key pressed'
        index += 1
        correct = 1 if (type_trial == 'Go' and response_type == 'key pressed') or (type_trial == 'NoGo' and response_type == 'not answered') else 0
        data = {
            'TRIAL' : index,
            'STIMULUS' : type_trial,
            'RESPONSE' : response_type,
            'CORRECT' : correct,
            'REACTION TIME' : response_time,
            'TIMESTAMP' : timestamp
        }
        df.append(data)
        print(f'Trial {index}, {type_trial} : {response_type}')
        pygame.time.delay(INTERVAL_TRIALS)
    save_path = os.path.join('Measures', user_id)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(os.path.join(save_path,'GoNogo_2_results.csv'), index=False)
    SCREEN.fill(WHITE)
    draw_text("THE END", font, (0, 0, 0), SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    pygame.display.flip()
    pygame.time.delay(2000) 



def Stroop(user_id = None):
    if not user_id:
       user_id = input_screen()
    key_list = [pygame.K_r, pygame.K_v, pygame.K_g, pygame.K_b]
    font = pygame.font.Font(None, 30)
    try_flag = True
    while try_flag:
        SCREEN.fill(BACKGROUND)
        draw_text("Prima seguirà un piccolo esercizio di prova per familiarizzare con il gioco", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2  -160)
        draw_text("Sullo schermo appariranno alcune parole", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2  -120)
        draw_text("Indica il più velocemente possibile il colore delle parole presentate, premendo i tasti:", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 -80)
        draw_text(" \' r \' se la parola è di colore ROSSO", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 -40)
        draw_text(" \' v \' se la parola è  di colore VERDE", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 )
        draw_text(" \' g \' se la parola è di colore GIALLO", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 +40)
        draw_text(" \' b \' se la parola è di colore BLU", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80)
        draw_text("Premi la BARRA SPAZIATRICE per iniziare il gioco", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 120)
        pygame.display.flip()
        try_combinations = []
        idx = 0
        while idx < 5:
            num = random.randint(0,3)
            try_combinations.append((num,num))
            idx += 1
        while idx < 10:
            num1 = random.randint(0,3)
            num2 = random.randint(0,3)
            if not num1 == num2:
                try_combinations.append((num1,num2))
                idx += 1
        random.shuffle(try_combinations)
        random.shuffle(try_combinations)
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if exit_flag(event):
                    running = False
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False
        running = True
        index = 0
        try_correct = 0
        while running and index<len(try_combinations):
            for event in pygame.event.get():
                if exit_flag(event):
                    running = False
                    break
                    # pygame.quit()
                    # sys.exit()
            SCREEN.fill(BLACK)
            pygame.display.flip()
            show_fixation(FIXATION_TIME_STROOP)
            SCREEN.fill(BLACK)
            pygame.display.flip()
            response = False
            color_tuple = try_combinations[index]
            draw_stroop_color(color_tuple)
            pygame.display.flip()
            start_time = pygame.time.get_ticks()
            while not response:
                current_time = pygame.time.get_ticks()
                for event in pygame.event.get():
                    if exit_flag(event):
                        running = False
                        response = True
                        break
                    if event.type == pygame.KEYDOWN and event.key in key_list:
                        timestamp =  int(datetime.now(timezone.utc).timestamp() * 1000)
                        response = True
                        response_time = current_time - start_time
                        response_type = event.unicode
            if not color_tuple[0] == color_tuple[1]:
                try_correct += int(KEY_DICT_STROOP[response_type] == STROOP_DICT[color_tuple[1]])
            index += 1
            pygame.time.delay(INTERVAL_TRIALS)    
        if try_correct >= 3:
            try_flag = False
            SCREEN.fill(BACKGROUND)
            pygame.display.flip()
        else:
            SCREEN.fill(BACKGROUND)
            draw_text(f"Hai totalizzato {5-try_correct} errori nelle incongruenti, RIPROVA. Chiedi assistenza qualora ne dovessi sentire il bisogno", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 )
            pygame.display.flip()
        pygame.time.delay(2000) 
    SCREEN.fill(BACKGROUND)
    draw_text("Nella sessione successiva avverrà l' effettiva prova del gioco", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2  -160)
    draw_text("Sullo schermo appariranno alcune parole", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2  -120)
    draw_text("Indica il più velocemente possibile il colore delle parole presentate, premendo i tasti:", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 -80)
    draw_text(" \' r \' se la parola è di colore ROSSO", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 -40)
    draw_text(" \' v \' se la parola è  di colore VERDE", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 )
    draw_text(" \' g \' se la parola è di colore GIALLO", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 +40)
    draw_text(" \' b \' se la parola è di colore BLU", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80)
    draw_text("Premi la BARRA SPAZIATRICE per iniziare il gioco", font, BLACK, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 120)
    pygame.display.flip()
    waiting = True
    combinations = []
    for i in range(4):
        # 3 congruent and 3 incongruent for each color, 2 congruent here and the rest in the next cycle
        combinations.append((i,i))
        combinations.append((i,i))
        for j in range(4):
            combinations.append((i,j))
    random.shuffle(combinations)
    random.shuffle(combinations) #another time, just because
    while waiting:
        for event in pygame.event.get():
            if exit_flag(event):
                running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
    index = 0
    running = True
    columns = ['TRIAL', 'WORD', 'COLOR', 'CONGRUENT', 'ANSWER', 'CORRECT', 'REACTION TIME', 'TIMESTAMP']
    df = []
    color_tuple = (0,0)
    response_time = 'None'
    response_type = 'r'
    while running and index<len(combinations):
        for event in pygame.event.get():
            if exit_flag(event):
                running = False
                break
                # pygame.quit()
                # sys.exit()
        SCREEN.fill(BLACK)
        pygame.display.flip()
        show_fixation(FIXATION_TIME_STROOP)
        SCREEN.fill(BLACK)
        pygame.display.flip()
        response = False
        color_tuple = combinations[index]
        draw_stroop_color(color_tuple)
        pygame.display.flip()
        start_time = pygame.time.get_ticks()
        while not response:
            current_time = pygame.time.get_ticks()
            for event in pygame.event.get():
                if exit_flag(event):
                    running = False
                    response = True
                    break
                if event.type == pygame.KEYDOWN and event.key in key_list:
                    timestamp =  int(datetime.now(timezone.utc).timestamp() * 1000)
                    response = True
                    response_time = current_time - start_time
                    response_type = event.unicode
        index += 1
        data = {
            'TRIAL' : index,
            'WORD' : STROOP_DICT[color_tuple[0]],
            'COLOR' : STROOP_DICT[color_tuple[1]],
            'CONGRUENT' : int( color_tuple[0] == color_tuple[1]),
            'ANSWER' : response_type,
            'CORRECT' : int(KEY_DICT_STROOP[response_type] == STROOP_DICT[color_tuple[1]]),
            'REACTION TIME' : response_time,
            'TIMESTAMP' : timestamp
        }
        df.append(data)
        print(f'Trial {index}, word {STROOP_DICT[color_tuple[0]]} written in {STROOP_DICT[color_tuple[1]]}, answer : {response_type}')
        pygame.time.delay(INTERVAL_TRIALS)
    save_path = os.path.join('Measures', user_id)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(os.path.join(save_path,'Stroop_results.csv'), index=False)
    SCREEN.fill(WHITE)
    draw_text("THE END", font, (0, 0, 0), SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    pygame.display.flip()
    pygame.time.delay(2000) 
    # pygame.quit()

def main():
    user_id = input_screen()
    SCREEN.fill(BACKGROUND)
    button_size = 300
    button_height = 100
    padding = 20
    buttons = [
        Button("Go-NoGo", SCREEN_WIDTH//4 - button_size //2-padding, SCREEN_HEIGHT//2 - padding - button_height, button_size, button_height, lambda: GoNogo(user_id)),
        Button("Go-NoGo 2", SCREEN_WIDTH//2 - button_size//2 , SCREEN_HEIGHT//2 - padding - button_height, button_size, button_height, lambda: epileptic_GoNoGO(user_id)),
        Button("Stroop", SCREEN_WIDTH*3//4 - button_size//2 +padding , SCREEN_HEIGHT//2 - padding - button_height, button_size, button_height, lambda : Stroop(user_id)),
        Button("Change User ID", SCREEN_WIDTH //2 - button_size - padding, SCREEN_HEIGHT//2 + padding, button_size, button_height, input_screen),
        Button("Exit", SCREEN_WIDTH // 2 + padding , SCREEN_HEIGHT //2  + padding , button_size, button_height, exit_game)
        ]
    for button in buttons:
        button.draw()
    pygame.display.flip()
    running = True
    while running:
        SCREEN.fill(BACKGROUND)
        for event in pygame.event.get():
            if exit_flag(event):
                running = False
            for button in buttons:
                result = button.handle_event(event)
                if result is not None:
                    user_id = result 
        for button in buttons:
          button.draw()
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
