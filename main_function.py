import cv2 as cv
import os

def match_fingerprint(fingerprint: str, fingerprints: list, exact_match=True):
    """
    ...
    fingerprint: (string) Filepath of input fingerprint file

    fingerprints: (string) List of filepaths of fingerprints to match

    exact_match: (boolean) True; Match best fingerprint from list

    False; Match list of possible matches
    ...

    """

    if exact_match:
        match_factor = 0.1
    else:
        match_factor = 0.3

    fingerprint = cv.imread(fingerprint)

    sift = cv.SIFT_create()

    keypoints_1, descriptor_1 = sift.detectAndCompute(fingerprint, None)


    scores = {}
    best_score = [0]

    for fingerprint in fingerprints:

        file = fingerprint
        filename = file.split("/")[-1]
        fingerprint = cv.imread(file)
        keypoints_2, descriptor_2 = sift.detectAndCompute(fingerprint, None)

        matches = cv.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(descriptor_1, descriptor_2, k=2)

        match_points = []

        for p, q in matches:
            if p.distance < match_factor * q.distance:
                match_points.append(p)
        
        
        keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        
        if len(match_points) / keypoints  * 100 > best_score[-1]:
            scores.clear()
            best_score.append(len(match_points) / keypoints  * 100)
            scores[filename] = best_score[-1]
            

    #max_score = max(list(scores.values()))
    #match = list(scores.keys())[list(scores.values()).index(max_score)]
    match = list(scores.keys())[0]
    if len(scores) != 0:
        if exact_match:
            print(scores)
            print(f"Match Found: {match}, Score:{scores[match]}")

        else:
            print("Possible matches found:")
            for match, score in scores.items():
                print(f"File:{match}, Score:{round(score, 2)}")

# filepath to input fingerprint file
fingerprint = "fingerprints/test/test (4).bmp"

# fingerprint files directory
filepath = "fingerprints/result"

# create list of all fingerprints in file directory
fingerprints = [f"{filepath}/{file}" for file in os.listdir(filepath)]

# run function to match fingerprint
match_fingerprint(fingerprint, fingerprints, exact_match=True)