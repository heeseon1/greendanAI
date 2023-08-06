import io
import requests
from PIL import Image
from flask import Flask, request, jsonify, send_file
import torch

app = Flask(__name__)

# yolov5s로 학습한 학습 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='halibut/jobs6/weights/best.pt', force_reload=True, skip_validation=True)

@app.route('/') #라우팅 테스트
def home():
    return 'Hello, World!'

def detect_image(image_resize):
    results = model(image_resize) #학습모델로 이미지 처리
    results.render()
    detections = results.pandas().xyxy[0].to_dict(orient='records') #감지된 객체 추출
    return results, detections #객체 추출 반환

@app.route('/detect', methods=['POST'])
def detect():
        file = request.files['file'] # 요청에서 업로드 된 이미지 가져오기 (form-data: file|"파일명.jpg")

        if not file:  # 파일이 존재하지 않을시 JSON 처리
            return jsonify({'error': 'No file'}), 400

        # 이미지 읽기 및 처리
        if file:
            image = Image.open(file)
            img_resize = image.resize((600,600))
            image_result, detections = detect_image(img_resize)

            temp_img = io.BytesIO()
            image_result.save(temp_img)

            temp_img.seek(0)

            django_url = 'http://127.0.0.1:8000/photo/send_photo_to_AI'
            files = {'file': ('temp_image.jpg', temp_img, 'image/jpeg')}
            response = requests.post(django_url, files=files)

            if response.status_code == 200:
                return send_file(io.BytesIO(response.content), mimetype='image/jpeg')
            else:
                return jsonify({"error": "개체 검출된 이미지를 장고 서버로 전송하는데 실패했습니다."}), 500

        else:
            return jsonify({"error": "잘못된 요청입니다."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)