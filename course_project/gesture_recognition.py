import cv2
import mediapipe as mp
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

# Dictionary to map hand gestures to their corresponding emoji\
emoji_collection = {
    'left_hand': '\U0001FAF2',
    'right_hand': '\U0001FAF1',
    'hand_up': '\U0001F91A',
    'thumbs_up': '\U0001F44D',
    'peace_sign': '\U0000270C',
    'point_left': '\U0001F448',
    'point_right': '\U0001F449',
    'point_up': '\U0001F446',
    'point_down': '\U0001F447',
}

# Extracts x and y coordinates from a landmark
def extract_landmark_positions(landmark):
    landmark_x=float(str(landmark).split('\n')[0][3:])
    landmark_y=float(str(landmark).split('\n')[1][3:])
    return landmark_x, landmark_y

# Determines the orientation of the hand based on landmarks
def orientation(coordinate_landmark_0, coordinate_landmark_9):
    x0= float(str(coordinate_landmark_0).split('\n')[0][3:])
    y0= float(str(coordinate_landmark_0).split('\n')[1][3:])
    
    x9= float(str(coordinate_landmark_9).split('\n')[0][3:])
    y9= float(str(coordinate_landmark_9).split('\n')[1][3:])
    
    # Calculate slope to determine the orientation
    if abs(x9 - x0) < 0.05:
        m = 1000000000
    else:
        m = abs((y9 - y0)/(x9 - x0))       

    # Determine orientation based on the slope
    if m>=0 and m<=1:
        if x9 > x0:
            return "Right"
        else:
            return "Left"

    if m>1:
        if y9 < y0:
            return "Up"
        else:
            return "Down"
        
# checks for left hand gesture
def check_left_hand(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (x8, y8), (x7, _) = extract_landmark_positions(result.landmark[8]), extract_landmark_positions(result.landmark[7])
    (x12, y12), (x11, y11) = extract_landmark_positions(result.landmark[12]), extract_landmark_positions(result.landmark[11])
    (x16, y16), (x15, y15) = extract_landmark_positions(result.landmark[16]), extract_landmark_positions(result.landmark[15])
    (x20, y20), (x19, y19) = extract_landmark_positions(result.landmark[20]), extract_landmark_positions(result.landmark[19])
    (x6, _), (x10, _) = extract_landmark_positions(result.landmark[6]), extract_landmark_positions(result.landmark[10])
    (x14, _), (x18, _) = extract_landmark_positions(result.landmark[14]), extract_landmark_positions(result.landmark[18])
    
    if direction == 'Down' or direction == 'Right' or direction == 'Up':
        return False
    
    if (y3>y4) and (y4<y8<y12<y16<y20) and (x7>x8) and (x11>x12) and (x15>x16) and (x19>x20):
        return True
    
    return False

# checks for right hand gesture
def check_right_hand(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (x8, y8), (x7, _) = extract_landmark_positions(result.landmark[8]), extract_landmark_positions(result.landmark[7])
    (x12, y12), (x11, y11) = extract_landmark_positions(result.landmark[12]), extract_landmark_positions(result.landmark[11])
    (x16, y16), (x15, y15) = extract_landmark_positions(result.landmark[16]), extract_landmark_positions(result.landmark[15])
    (x20, y20), (x19, y19) = extract_landmark_positions(result.landmark[20]), extract_landmark_positions(result.landmark[19])
    (x6, _), (x10, _) = extract_landmark_positions(result.landmark[6]), extract_landmark_positions(result.landmark[10])
    (x14, _), (x18, _) = extract_landmark_positions(result.landmark[14]), extract_landmark_positions(result.landmark[18])
    
    if direction == 'Down' or direction == 'Left' or direction == 'Up':
        return False
    
    if (y3>y4) and (y4<y8<y12<y16<y20) and (x7<x8) and (x11<x12) and (x15<x16) and (x19<x20):
        return True
    
    return False

# checks for hand up gesture
def check_hand_up(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (_, y7), (_, y8) = extract_landmark_positions(result.landmark[7]), extract_landmark_positions(result.landmark[8])
    (_, y11), (_, y12) = extract_landmark_positions(result.landmark[11]), extract_landmark_positions(result.landmark[12])
    (_, y15), (_, y16) = extract_landmark_positions(result.landmark[15]), extract_landmark_positions(result.landmark[16])
    (_, y19), (_, y20) = extract_landmark_positions(result.landmark[19]), extract_landmark_positions(result.landmark[20])
    
    if check_thumbs_up(result):
        return False
    
    if direction == 'Down' or direction == 'Left' or direction == 'Right':
        return False
    
    if (y4<y3) and (y8<y7) and (y12<y11) and (y16<y15) and (y20<y19) and (y4>y8) and (y4>y12) and (y4>y16) and (y4>y20):
        return True
        
    return False

# checks for thumbs up gesture
def check_thumbs_up(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (x3, y3), (x4, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (x5, y5), (x8, y8) = extract_landmark_positions(result.landmark[5]), extract_landmark_positions(result.landmark[8])
    (x9, y9), (x12, y12) = extract_landmark_positions(result.landmark[9]), extract_landmark_positions(result.landmark[12])
    (x13, y13), (x16, y16) = extract_landmark_positions(result.landmark[13]), extract_landmark_positions(result.landmark[16])
    (x17, y17), (x20, y20) = extract_landmark_positions(result.landmark[17]), extract_landmark_positions(result.landmark[20])
    
    if direction == 'Up' or direction == 'Down':
        return False
    
    if y3<y4:
        return False
    
    if direction == 'Left':
        if (x5<x8) and (x9<x12) and (x13<x16) and (x17<x20) and (y4<y5<y9<y13<y17):
            return True
    
    elif direction == 'Right':
        if (x5>x8) and (x9>x12) and (x13>x16) and (x17>x20) and (y4<y5<y9<y13<y17):
            return True
        
    return False

# checks for peace sign gesture
def check_peace_sign(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (_, y7), (_, y8) = extract_landmark_positions(result.landmark[7]), extract_landmark_positions(result.landmark[8])
    (_, y11), (_, y12) = extract_landmark_positions(result.landmark[11]), extract_landmark_positions(result.landmark[12])
    (_, y15), (_, y16) = extract_landmark_positions(result.landmark[15]), extract_landmark_positions(result.landmark[16])
    (_, y19), (_, y20) = extract_landmark_positions(result.landmark[19]), extract_landmark_positions(result.landmark[20])
    (_, y13), (_, y17) = extract_landmark_positions(result.landmark[13]), extract_landmark_positions(result.landmark[17])
    (_, y14), (_, y18) = extract_landmark_positions(result.landmark[14]), extract_landmark_positions(result.landmark[18])
    
    if direction == 'Down' or direction == 'Right' or direction == 'Left':
        return False
    
    if (y7>y8) and (y11>y12) and (y16>y15) and (y20>y19) and (y3>y4) and (y4>y14) and (y4>y18):
        return True
    
    return False

# checks for pointing left gesture
def check_point_left(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (x8, y8), (x7, _) = extract_landmark_positions(result.landmark[8]), extract_landmark_positions(result.landmark[7])
    (x12, y12) = extract_landmark_positions(result.landmark[12])
    (x16, y16) = extract_landmark_positions(result.landmark[16])
    (x20, y20) = extract_landmark_positions(result.landmark[20])
    (x6, _), (x10, _) = extract_landmark_positions(result.landmark[6]), extract_landmark_positions(result.landmark[10])
    (x14, _), (x18, _) = extract_landmark_positions(result.landmark[14]), extract_landmark_positions(result.landmark[18])
    
    if direction == 'Down' or direction == 'Right' or direction == 'Up':
        return False
    
    if (y3>y4) and (y4<y8<y12<y16<y20) and (x6>x7>x8) and (x12>x10) and (x16>x14) and (x20>x18):
        return True
    
    return False

# checks for pointing right gesture
def check_point_right(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (x8, y8), (x7, _) = extract_landmark_positions(result.landmark[8]), extract_landmark_positions(result.landmark[7])
    (x12, y12) = extract_landmark_positions(result.landmark[12])
    (x16, y16) = extract_landmark_positions(result.landmark[16])
    (x20, y20) = extract_landmark_positions(result.landmark[20])
    (x6, _), (x10, _) = extract_landmark_positions(result.landmark[6]), extract_landmark_positions(result.landmark[10])
    (x14, _), (x18, _) = extract_landmark_positions(result.landmark[14]), extract_landmark_positions(result.landmark[18])
    
    if direction == 'Down' or direction == 'Left' or direction == 'Up':
        return False
    
    if (y3>y4) and (y4<y8<y12<y16<y20) and (x6<x7<x8) and (x12<x10) and (x16<x14) and (x20<x18):
        return True
    
    return False

# checks for pointing up gesture
def check_point_up(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (x7, y7), (x8, y8) = extract_landmark_positions(result.landmark[7]), extract_landmark_positions(result.landmark[8])
    (x9, y9), (x12, y12) = extract_landmark_positions(result.landmark[9]), extract_landmark_positions(result.landmark[12])
    (x13, y13), (x16, y16) = extract_landmark_positions(result.landmark[13]), extract_landmark_positions(result.landmark[16])
    (x17, y17), (x20, y20) = extract_landmark_positions(result.landmark[17]), extract_landmark_positions(result.landmark[20])
    
    if direction == 'Down' or direction == 'Left' or direction == 'Right':
        return False
    
    if (y3>y4) and (y7>y8) and (y12>y9) and (y16>y13) and (y20>y17) and ((x7>x9>x13>x17) or (x7<x9<x13<x17)):
        return True
    
    return False

# checks for pointing left gesture
def check_point_down(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = extract_landmark_positions(result.landmark[3]), extract_landmark_positions(result.landmark[4])
    (x7, y7), (_, y8) = extract_landmark_positions(result.landmark[7]), extract_landmark_positions(result.landmark[8])
    (x9, _), (_, y12) = extract_landmark_positions(result.landmark[9]), extract_landmark_positions(result.landmark[12])
    (x13, _), (_, y16) = extract_landmark_positions(result.landmark[13]), extract_landmark_positions(result.landmark[16])
    (x17, _), (_, y20) = extract_landmark_positions(result.landmark[17]), extract_landmark_positions(result.landmark[20])
    (_, y14), (_, y10) = extract_landmark_positions(result.landmark[14]), extract_landmark_positions(result.landmark[10])
    (_, y18) = extract_landmark_positions(result.landmark[18])
    
    if direction == 'Up' or direction == 'Left' or direction == 'Right':
        return False
    
    if (y3<y4) and (y7<y8) and (y12<y10) and (y16<y14) and (y20<y18) and ((x7>x9>x13>x17) or (x7<x9<x13<x17)):
        return True
    
    return False
    


# Shuffles the emoji collection dictionary
def get_shuffled_dictionary():
    items = list(emoji_collection.items())
    random.shuffle(items)
    shuffled_dict = dict(items)
    return shuffled_dict

# Checks the gesture based on the given string and returns the appropriate function call
def check_actions(string, direction):
    if string == 'hand_up':
        return check_hand_up(direction)
    elif string == 'thumbs_up':
        return check_thumbs_up(direction)
    elif string == 'peace_sign':
        return check_peace_sign(direction)
    elif string == 'point_left':
        return check_point_left(direction)
    elif string == 'point_right':
        return check_point_right(direction)
    elif string == 'point_up':
        return check_point_up(direction)
    elif string == 'point_down':
        return check_point_down(direction)
    elif string == 'left_hand':
        return check_left_hand(direction)
    elif string == 'right_hand':
        return check_right_hand(direction)
    else:
        return False

# Setting the dimensions for the video capture window
frameWidth = 720
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# Initializing MediaPipe solutions for hand tracking
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

seq_dict = get_shuffled_dictionary()
start_time = 0 #set timer to 0

# Main loop for the application
while True:
    #Shuffle emoji sequence and setup for hand gesture recognition
    seq_dict = get_shuffled_dictionary()
    success, img = cap.read()
    img= cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape

    # define font paths  
    regular_font = "assets/regular.ttf"
    bold_font = "assets/bold.ttf"

    # screen display
    game_display = np.zeros((h, w, 3), np.uint8)
    game_display[:] = [240, 207, 137]  
    font = ImageFont.truetype(regular_font, 20)
    font_title = ImageFont.truetype(bold_font, 45)
    font_bold = ImageFont.truetype(bold_font, 30)
    img_pil = Image.fromarray(game_display)
    draw = ImageDraw.Draw(img_pil)
    light_blue_bgr = (230, 216, 173)
    
    # draw text for welcome screen
    draw.text((550, 200), 'Welcome!', embedded_color=True, font = font_title, fill=(255, 255, 255))
    draw.text((435, 260), 'Pathways Pain Relief', embedded_color=True, font = font_title, fill=(255, 255, 255))
    draw.text((460, 350), 'Welcome to your daily hand exercise journey!', embedded_color=True, font = font, fill=(0, 0, 0))
    draw.text((410, 370), 'We are here to guide you through simple yet effective', embedded_color=True, font = font, fill=(0, 0, 0))
    draw.text((410, 390), 'routines designed to alleviate pain and improve flexibility.', embedded_color=True, font = font, fill=(0, 0, 0))
    draw.text((410, 410), 'These exercises daily can significantly enhance your quality of', embedded_color=True, font = font, fill=(0, 0, 0))
    draw.text((410, 430), 'life by reducing discomfort', embedded_color=True, font = font, fill=(0, 0, 0))
    draw.text((600, 490), 'Press enter to begin', embedded_color=True, font = font, fill=(102, 0, 204))
    seq = np.array(img_pil)
    img[:, :] = seq
    total_gestures = 0

    # Wait for the user to press enter to begin
    if cv2.waitKey(1) == 13:
        start_time = time.time() #start timer
        while True:
            success, img = cap.read()
            img= cv2.flip(img,1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            sequence = ''.join(seq_dict.values())
            string_sequence = list(seq_dict.keys())
            
            emojis = "assets/emojis.ttf"
            emoji_sequence = np.zeros((50, 460, 3),np.uint8)
            font = ImageFont.truetype(emojis, 40)
            img_pil = Image.fromarray(emoji_sequence)
            draw = ImageDraw.Draw(img_pil)
            draw.text((0, 0), sequence, embedded_color=True, font = font, fill=(255, 255, 255))
            seq = np.array(img_pil)
            
            img[:50, 420:880] = seq
            
            elapsed_time = time.time() - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            cv2.putText(img, 'Time: '+formatted_time, (980, 600), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            cv2.putText(img, 'Signs Completed: ' + str(total_gestures), (980, 650), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
            
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    direction = results.multi_hand_landmarks[-1]
                    if check_actions(string_sequence[total_gestures], direction):
                        total_gestures += 1
                        cv2.imshow('image', img)
                        if total_gestures == 9:
                            break
            
            cv2.imshow('image', img)
            if cv2.waitKey(1)==27:
                break
            
            if total_gestures == 9:
                break
            
    tota_time = round(time.time() - start_time, 0)
    # when all gestures have been completed end and show end screen
    while total_gestures == 9:
        success, img = cap.read()
        img= cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        count_display = np.zeros((h, w, 3),np.uint8)*255
        count_display[:] = [240, 207, 137] 
        img_pil = Image.fromarray(count_display)
        draw = ImageDraw.Draw(img_pil)
        draw.text((460, 280), 'Excercise Complete!', embedded_color=True, font = font_title, fill=(255, 255, 255))
        draw.text((500, 350), 'Total Time taken: '+str(tota_time), embedded_color=True, font = font_bold, fill=(0, 0, 0))
        seq = np.array(img_pil)
        img[:, :] = seq
        
        cv2.imshow('image', img)
        if cv2.waitKey(1)==27:
                break
            
    cv2.imshow('image', img)
    if cv2.waitKey(1)==27:
            break