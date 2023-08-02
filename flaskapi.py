import io
from PIL import Image
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# yolov5s로 학습한 학습 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='halibut/jobs6/weights/best.pt', force_reload=True, skip_validation=True)

@app.route('/') #라우팅 테스트
def home():
    return 'Hello, World!'

def detect_image(image):
    results = model(image) #학습모델로 이미지 처리
    detections = results.pandas().xyxy[0].to_dict(orient='records') #감지된 객체 추출
    return detections #객체 추출 반환

@app.route('/detect', methods=['POST'])
def detect():
        file = request.files['file'] # 요청에서 업로드 된 이미지 가져오기 (form-data: file|"파일명.jpg")

        if not file:  # 파일이 존재하지 않을시 JSON 처리
            return jsonify({'error': 'No file'}), 400

        # 이미지 읽기 및 처리
        if file:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)) #불러온 이미지를 바이트 형식으로 받기 (이미지 처리)
            detections = detect_image(image) # 학습모델함수에 해당 이미지 값 넣기
            return jsonify({"result": detections}), 200 # 처리 결과 JSON으로 반환
        else:
            return jsonify({"error": "Invalid request"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)