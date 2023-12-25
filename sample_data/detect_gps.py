import argparse
import math
import cv2
from ultralytics import YOLO 

# Argument parser 설정
parser = argparse.ArgumentParser(description="GPS Coordinates Calculation from Image")
parser.add_argument('--impath', type=str, help="Path to the image file")
parser.add_argument('--cam_angle', type=float, default=0.0, help="Camera direction angle in degrees") # added
args = parser.parse_args()

# 이미지 파일 경로에서 파일 이름 추출 (확장자 제외)
image_file_name = args.impath.split('/')[-1].split('.')[0]

# 카메라 방향각(북쪽 기준 시계 반대방향)added
camera_angle = args.cam_angle

# 이미지 파일 경로
image_path = args.impath
# 이미지 불러오기
img = cv2.imread(image_path)
# 이미지 크기 얻기
image_height, image_width, _ = img.shape

# 사전 훈련된 YOLOv8x model 로드
model = YOLO('/Users/kukut/Desktop/EE2_project/ultralytics/v8x_visdrone.pt')
# 이미지에서 추론 실행
results = model(image_path, save = True)

# 주어진 정보
camera_height = 25  # 카메라 높이, 단위: 미터
field_of_view_deg = 79.52  # 화각, 단위: 도
# image_width =   # 이미지 가로 픽셀
# image_height =   # 이미지 세로 픽셀
camera_latitude = 37.540683333333334  # 촬영 장소의 위도
camera_longitude = 127.07928611111111111  # 촬영 장소의 경도


def calculate_gps_from_pixel_offset(pixel_offset_x, pixel_offset_y, image_width, image_height, camera_height, field_of_view_deg):
    # 이미지의 실제 대각선 길이 계산  
    diagonal_distance = 2 * camera_height * math.tan(math.radians(field_of_view_deg / 2))

    # 이미지의 가로 및 세로 실제 길이 계산
    image_width_meters = diagonal_distance * (image_width / math.sqrt(image_width**2 + image_height**2))
    image_height_meters = diagonal_distance * (image_height / math.sqrt(image_width**2 + image_height**2))
    
    # 픽셀 오프셋을 실제 좌표로 변환
    offset_x_meters = (pixel_offset_x / image_width) * image_width_meters
    offset_y_meters = (-pixel_offset_y / image_height) * image_height_meters # 이 부분 수정됨. 원래 좌표계로 변환 됨.
    
    # 카메라 방향각을 고려한 좌표 보정 계산
    offset_x_meters_correction = offset_x_meters * math.cos(math.radians(camera_angle)) - offset_y_meters * math.sin(math.radians(camera_angle))
    offset_y_meters_correction = offset_x_meters * math.sin(math.radians(camera_angle)) + offset_y_meters * math.cos(math.radians(camera_angle))

    # 사람의 GPS 좌표 계산
    # 지구 반지름은 평균값으로 6,371 km로 가정하면, 1도의 거리는 지구의 반지름을 360으로 나눈 값, 약 111 km 이다.
    person_latitude = camera_latitude + (offset_y_meters_correction / 111000) 
    person_longitude = camera_longitude + (offset_x_meters_correction / (111000 * math.cos(math.radians(camera_latitude))))

    return person_longitude, person_latitude

# GPS좌표와 라벨 리스트 초기화
people_gps_coordinates = []
detected_labels = []

j = 0
for r in results:
    for box in r.boxes:
        # 클래스 인덱스 추출
        class_index = box.cls.item()  # 클래스 인덱스를 추출하고, .item()을 사용하여 Python 정수로 변환
        label_name = r.names[class_index]  # 클래스 인덱스를 사용하여 라벨 이름 추출

        # 바운딩 박스의 중심 좌표 계산
        x_center = (box.xyxy[j][0] + box.xyxy[j][2]) / 2
        y_center = (box.xyxy[j][1] + box.xyxy[j][3]) / 2
        
        x_center = x_center.item()
        y_center = y_center.item()

        # 이미지의 중심 좌표를 사용하여 GPS 좌표 계산
        person_gps_coordinates = calculate_gps_from_pixel_offset(x_center - image_width/2, y_center - image_height/2, image_width, image_height, camera_height, field_of_view_deg)

        people_gps_coordinates.append(person_gps_coordinates)
        detected_labels.append(label_name)  # 라벨 이름 저장
    j = j + 1

# 탐지된 라벨에 따른 마커 색상 지정
label_marker_styles = {
    'person': 'http://maps.google.com/mapfiles/ms/icons/green-dot.png',
    'motorcycle': 'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
    'truck': 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png',
    'bicycle': 'http://maps.google.com/mapfiles/ms/icons/orange-dot.png',
    'bus': 'http://maps.google.com/mapfiles/ms/icons/purple-dot.png',
    'car': 'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
}


# 탐지된 대상을 색상과 함께 표시하는 KML 파일 생성
kml_template = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    %s
  </Document>
</kml>
"""

placemark_template = """    <Placemark>
      <name>%s</name>
      <Style>
        <IconStyle>
          <Icon>
            <href>%s</href>
          </Icon>
        </IconStyle>
      </Style>
      <Point>
        <coordinates>%s,%s</coordinates>
      </Point>
    </Placemark>
"""

placemark_contents = ""
for i, coordinates in enumerate(people_gps_coordinates):
    label = detected_labels[i]
    marker_style = label_marker_styles.get(label, 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png')
    placemark_contents += placemark_template % (label, marker_style, coordinates[0], coordinates[1])

# KML 양식에 위치정보 쓰기
kml_content = kml_template % placemark_contents

# KML 파일 이름 설정 (이미지 이름 사용)
kml_file_name = f"{image_file_name}_locations.kml"

# KML file 저장
with open(kml_file_name, 'w') as f:
    f.write(kml_content)