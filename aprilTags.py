import math
import json

#this is a list of [tag nums, distances] to travel to
aprilTargets = ([2, 70], [46, 80], [41, 80], [36, 80], [24, 80])

#finds the closes tag to a point
def getClosestTag(x, z):
    f = open('worldPoints.json', 'r')
    worldPoints = json.load(f)
    f.close()
    keys = worldPoints.keys()

    best_distance = 100000
    for i in range(0, 49):
        if str(i) in keys: 
            if keys[i][0][2] < 0: #only look at keys <1  (outside)   
                full_tag_dat = worldPoints[str(i)]
                tag_loc = (full_tag_dat[0][0], full_tag_dat[0][2])
                dist = math.sqrt(pow(x - tag_loc[0], 2) + pow(z - tag_loc[1], 2))
                if dist < best_distance:
                    best_distance = dist
                    best_tag = i
    print("selected target:" + str(best_tag))
    return (best_tag, 80)