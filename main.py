import cv2
import mediapipe as mp
import math 

#ตั้งค่าHand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# พิกัดของปลายนิ้ว (Landmarks ID)
# 4:นิ้วโป้ง,8:นิ้วชี้,12:นิ้วกลาง,16:นิ้วนาง,20:นิ้วก้อย
tip_ids = [4, 8, 12, 16, 20]

# กล้องเว็บแคม
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

while True:
    success, img = cap.read()
    if not success:
        break

    # พลิกภาพแบบเหมือนกระจก
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # ตรวจสอบว่าตรวจจับมือได้ไหม
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            # เอาไว้เก็บพิกัดของ Landmark ทั้ง 21 จุด
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                
            # วาดจุด (Landmarks) และเส้นเชื่อมบนมือ
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # การนับนิ้ว
            fingers = []
            
            # ตรวจสอบว่าเป็นมือซ้ายหรือมือขวา
            handedness_data = results.multi_handedness[0] if results.multi_handedness else None
            handedness = handedness_data.classification[0].label if handedness_data else "Unknown"

            # ตรรกะสำหรับนิ้วโป้ง (Thumb - ใช้แกน X)
            if handedness == "Right": # มือขวา
                 if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0]-1][1]:
                     fingers.append(1)
                 else:
                     fingers.append(0)
            elif handedness == "Left": # มือซ้าย
                 if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0]-1][1]:
                     fingers.append(1)
                 else:
                     fingers.append(0)
            else:
                fingers.append(0)


            # ตรรกะสำหรับนิ้วอื่นๆ (Index, Middle, Ring, Pinky - ใช้แกน Y)
            for id in range(1, 5): 
                # ถ้าพิกัด Y ของปลายนิ้ว (tip_ids[id]) < พิกัด Y ของข้อต่อถัดลงมา (tip_ids[id]-2) แสดงว่านิ้วยกขึ้น
                if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            total_fingers = fingers.count(1)
            
            # แสดงผลจำนวนนิ้ว
            cv2.putText(img, f'Fingers: {total_fingers}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (0, 255, 0), 3) # สีเขียว (0, 255, 0)


    # แสดงหน้าต่างผลลัพธ์
    cv2.imshow("Hand Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()