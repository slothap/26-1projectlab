import cv2
import time
import numpy as np
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolov8n.pt')

# 2. 스마트폰 카메라 연동
video_url = 'http://172.30.19.196:8080/video'
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(video_url) //카메라로 연결하면 이걸로

# ROI 기본 설정
ret, test_frame = cap.read()
if ret:
    h, w, _ = test_frame.shape
    roi_w, roi_h = int(w / 3), int(h / 3)
    roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2 
else:
    roi_x, roi_y, roi_w, roi_h = 200, 150, 250, 200

# 상태
STATE_EMPTY = "EMPTY"       
STATE_IN_USE = "IN_USE"     
STATE_GRACE = "GRACE"       

current_state = STATE_EMPTY 
current_user_id = None      
usage_start_time = 0        
grace_start_time = 0        
GRACE_TIME_LIMIT = 5.0      

# 출입 인증 연동
active_members = {} 
member_history = {} 

def get_member_id(track_id):
    if track_id not in active_members:
        active_members[track_id] = f"USER_{track_id:03d}"
    return active_members[track_id]

# 기구의 통계적 평균과 편차 임의 설정 sec
GLOBAL_MU = 20.0 
GLOBAL_SIGMA = 7.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("IP 주소 재확인")
        break

    results = model.track(frame, persist=True, classes=0, verbose=False)
    ids_in_roi = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # 사용자 추적 포인트 잡기
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1) 
            
            if roi_x < cx < roi_x + roi_w and roi_y < cy < roi_y + roi_h:
                ids_in_roi.append(track_id)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            member_id = get_member_id(track_id)
            cv2.putText(frame, f"ID: {member_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 상태 전환 및 타이머 로직
    if current_state == STATE_EMPTY:
        if ids_in_roi: 
            current_user_id = ids_in_roi[0] 
            current_state = STATE_IN_USE
            usage_start_time = time.time()  
            
    elif current_state == STATE_IN_USE:
        if current_user_id not in ids_in_roi:
            if len(ids_in_roi) > 0:
                current_user_id = ids_in_roi[0]
            else:
                current_state = STATE_GRACE     
                grace_start_time = time.time()  
            
    elif current_state == STATE_GRACE:
        if len(ids_in_roi) > 0: 
            current_user_id = ids_in_roi[0] 
            usage_start_time += (time.time() - grace_start_time)
            current_state = STATE_IN_USE    
        else:
            elapsed = time.time() - grace_start_time
            if elapsed > GRACE_TIME_LIMIT:  
                actual_duration = grace_start_time - usage_start_time
                if actual_duration > 3.0:
                    member_id = get_member_id(current_user_id)
                    if member_id not in member_history:
                        member_history[member_id] = []
                    member_history[member_id].append(int(actual_duration))

                current_state = STATE_EMPTY 
                current_user_id = None

    # 신호등 색상 및 상태 텍스트 판별
    traffic_color = (0, 255, 0) # 기본 초록색
    status_text = "EMPTY (Available)"
    detail_text = "" # 두 번째 줄 - 상세 정보
    
    if current_state == STATE_IN_USE:
        used_time = int(time.time() - usage_start_time)
        member_id = get_member_id(current_user_id)
        history = member_history.get(member_id, [])
        
        if len(history) < 2: #3->2이어도 충분
            current_mu = GLOBAL_MU
            current_sigma = GLOBAL_SIGMA
            stat_type = "G"
        else:
            current_mu = float(np.mean(history))
            current_sigma = max(1.0, float(np.std(history)))
            stat_type = "L"

        t_on = current_mu + 0.524 * current_sigma
        t_off = current_mu + 1.645 * current_sigma
        
        #TODO: 현재 사용 시간과 t_off을 고정으로 표시하기
        detail_text = f"Used: {used_time}s / Max Limit: {int(t_off)}s"
        
        # !!!상태별로 타이머 기준을 Ton, Toff로 세분화
        if used_time >= t_off:
            traffic_color = (0, 0, 255) # 빨간색 (이상치)
            over_time = int(used_time - t_off)
            status_text = f"OUTLIER [{stat_type}] - Over: +{over_time}s"
            
        elif used_time >= t_on:
            traffic_color = (0, 255, 255) # 노란색 (종료 임박)
            remain_to_off = max(0, int(t_off - used_time))
            status_text = f"ENDING SOON [{stat_type}] - Limit In: ~{remain_to_off}s"
            
        else:
            traffic_color = (0, 0, 255) # 빨간색 (사용 중)
            remain_to_on = max(0, int(t_on - used_time))
            status_text = f"IN USE [{stat_type}] - Yellow In: ~{remain_to_on}s"
            
    elif current_state == STATE_GRACE:
        traffic_color = (0, 165, 255) # 주황색 (자리 비움)
        remain_time = max(0.0, GRACE_TIME_LIMIT - (time.time() - grace_start_time))
        status_text = f"AWAY - {remain_time:.1f}s left"

    # UI - 기구 상태 (신호등 & 텍스트)
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), traffic_color, 2)
    cv2.circle(frame, (roi_x + 15, roi_y - 25), 10, traffic_color, -1)
    
    # (메인 상태)
    cv2.putText(frame, status_text, (roi_x + 35, roi_y - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, traffic_color, 2)
                
    # (상세 시간 정보)
    if detail_text:
        cv2.putText(frame, detail_text, (roi_x + 35, roi_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, traffic_color, 1)

    dashboard_overlay = frame.copy()
    dashboard_h = max(60, 40 + len(member_history) * 25)
    cv2.rectangle(dashboard_overlay, (10, 10), (320, dashboard_h), (0, 0, 0), -1)
    cv2.addWeighted(dashboard_overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, "--- User Usage History ---", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset = 55
    for m_id, times in member_history.items():
        if times:
            avg_t = int(np.mean(times))
            times_str = ", ".join([f"{t}s" for t in times[-3:]])
            history_text = f"{m_id}: Avg {avg_t}s | [{times_str}]"
            cv2.putText(frame, history_text, (15, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_offset += 25

    # 화면 출력 부분
    cv2.imshow("Weightpass Vision AI", frame)

    # 박스 이동
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):       
        break
    elif key == ord('a'):     
        roi_x -= 20
    elif key == ord('d'):     
        roi_x += 20
    elif key == ord('w'):     
        roi_y -= 20
    elif key == ord('s'):     
        roi_y += 20
        
    roi_x = max(0, min(roi_x, w - roi_w))
    roi_y = max(0, min(roi_y, h - roi_h))

cap.release()
cv2.destroyAllWindows()