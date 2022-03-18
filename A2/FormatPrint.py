# just prepared for a 6*6 gridWorld
def format_print(lst):
    for i in range(6):
        for j in range(6):
            print("%.2f" %lst[6*i+j], end=" ")
        print("")
