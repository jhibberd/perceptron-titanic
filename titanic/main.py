from collections import defaultdict


result = []
parsed_header = False
with open("/home/jhibberd/projects/learning/titanic/train.csv") as f:
    for ln in f.readlines():

        # Ignore the header
        if not parsed_header:
            parsed_header = True
            continue

        # Ignore newline char
        ln = ln[:-1]

        ln = ln.split(",")

        result.append(float(ln[-3]))

result.sort()
print result

"""
Male
(False, True):  468 81% 
(True, True):   109 19%
                577

Female
(False, False):  81 26%
(True, False):  233 74%
                314

75% of females survived, 81% of males survived
"""
