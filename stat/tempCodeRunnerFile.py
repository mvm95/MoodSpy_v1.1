 n1, n2 in zip(real_planning_vector, b):
            if n1 != n2:
                print(n1,n2,sum)
            sum += 1
        print('------')
