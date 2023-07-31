import cv2
import numpy as np
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='halibut/jobs6/weights/best.pt', force_reload=True, skip_validation=True) #학습 모델 로드

def detect_image(image):
    results = model(image) #학습모델로 이미지 처리
    detections = results.pandas().xyxy[0].to_dict(orient='records') #감지된 객체 추출
    return detections #객체 추출 반환

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['image'] # 요청에서 업로드 된 이미지 가져오기

        if not file: #파일이 존재하지 않을시 JSON 처리
            return jsonify({'error': 'No image provided'}), 400

        # 이미지 읽기 및 처리
        image = torch.from_numpy(cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR))
        detections = detect_image(image)
        return jsonify(detections) #결과 JSON으로 반환

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
