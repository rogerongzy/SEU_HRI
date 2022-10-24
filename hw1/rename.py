import cv2
import os 

# rename script

# changzhou_chamber_
# teaching_one_
# nine_bridge_
# communicate_chamber_
# communicate_experiment_
# PE_office_
# library_
# dormitory_peach_
# playground_paech_
# basketball_peach_
# canteen_peach_
# canteen_orange_

index = 0

for file in os.listdir(): 
    print(file)
    if file.endswith('jpg'):
        name = 'changzhou_chamber_' + str(index) + '.jpg'
        os.rename(file, name)
        index += 1
    

