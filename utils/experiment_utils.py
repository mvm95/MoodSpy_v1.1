import random

def find_swappable_idx(vector):
    swappable_idx = []
    for idx in range(1,len(vector)-1):
        if vector[idx] == 0 and vector[idx] == vector[idx-1] and vector[idx] ==  vector[idx+1]:
            swappable_idx.append(idx)
    return swappable_idx

def create_video_planning(movement = 'Stop'):
    try:
        n_task = 77
        num_activities = 17
        n_gialli = 6
        vector = [0]*n_task
        ones_idx = random.sample(range(n_task), num_activities)
        for idx in ones_idx:
            vector[idx] = 1
        swappable_idx = find_swappable_idx(vector)
        flag = False
        if not swappable_idx:
            flag = True
        if not flag:
            for idx in range(1, len(vector)-1):
                if vector[idx] == 1 and (vector[idx] == vector[idx-1] or vector[idx] ==  vector[idx+1]):
                    idx_to_replace = random.choice(swappable_idx)
                    vector[idx_to_replace] = 1
                    vector[idx] = 0
                    swappable_idx = find_swappable_idx(vector)
                    if not swappable_idx:
                        flag = True
                        break
        if flag:
            return  create_video_planning(movement)
        if vector[0] == 1:
            return create_video_planning(movement)        
        ones_idx = []
        c = [0 , 0]
        for idx, n in enumerate(vector):
            c[n] += 1
            if n == 1:
                ones_idx.append(idx)
        if not c == [n_task - num_activities, num_activities]:
            return create_video_planning(movement)
        if movement == 'Movement':
            for idx, n in enumerate(vector):
                if n == 1:
                   vector[idx] = 2
        elif movement == 'Both':
            twos_idx = random.sample(ones_idx, n_gialli)
            for idx in twos_idx:
                vector[idx] = 2
        for _ in range(4):
            vector.append(0)
    except RecursionError:
       return create_video_planning(movement)  
    return vector

def main():
    import time
    movement_list = ['Stop', 'Movement', 'Both']
    for idx in range(100):
        movement = movement_list[idx % 3]
        v = create_video_planning(movement)
        print(v)
        c = [0 , 0, 0]
        for _, n in enumerate(v):
            c[n] += 1
        print(c)
        time.sleep(0.5)

if __name__ == '__main__':
    main()
            
