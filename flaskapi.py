import base64
import io
import requests
from PIL import Image
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# yolov5s로 학습한 학습 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='halibut/jobs6/weights/best.pt', force_reload=True,
                       skip_validation=True)

@app.route('/')  # 라우팅 테스트
def home():
    return 'Hello, World!'


def detect_image(image_resize):

    try:
        results = model(image_resize)  # 학습모델로 이미지 처리
        results.render()
        detections = results.pandas().xyxy[0].to_dict(orient='records')  # 감지된 객체 추출

        ai_result = 'ETC'
        ai_name = '기타'

        if detections:
            for label in detections:
                if label['name'] == 'Nomal':
                    ai_result = 'GOOD'
                    ai_name = '정상'
                    break

                elif label['name'] in ['LMe', 'LMm', 'LMl']:
                    ai_result = 'BAD'
                    ai_name = '토마토 잎 곰팡이병'
                    break

                elif label['name'] in ['LCVe', 'LCVm', 'LCVl']:
                    ai_result = 'BAD'
                    ai_name = '토마토 황화잎말이 바이러스'
                    break

            return results, ai_result, ai_name  # 객체 추출 반환

        return results, ai_result, ai_name

    except Exception as e:
        ai_result = 'ETC'
        ai_name = '기타'
        print(f"image_error: {str(e)}")
        return image_resize, ai_result, ai_name


@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['file']  # 요청에서 업로드 된 이미지 가져오기 (form-data: file|"파일명.jpg")
        print(file)

        if not file:  # 파일이 존재하지 않을시 JSON 처리
            return jsonify({'error': 'No file'}), 400

        # 이미지 읽기 및 처리
        image = Image.open(io.BytesIO(file.read()))
        print("image:", image)

        img_resize = image.resize((600, 600))
        image_result, ai_result, ai_name = detect_image(img_resize)

        print("image_result:", image_result)
        print("ai_result:", ai_result)

        image_result.render()

        ai_images = []

        for i in image_result.ims: #ims로 이름 변경
            ai_byte = io.BytesIO()
            ai_img = Image.fromarray(i)
            ai_img.save(ai_byte, format='jpeg')
            ai_images.append(base64.b64encode(ai_byte.getvalue()).decode('utf-8'))

        result = {
            'code': 200,
            'msg': 'successful',
            'result': {
                'ai_result': ai_result,
                'ai_images': ai_images,
                'ai_name': ai_name
             }
         }

        return jsonify(result)

    except Exception as e:
        print(f"image_error: {str(e)}")
        return jsonify({"error":"image error"})


@app.route('/img')
def image_json():
    image_path = 'file.jpg'
    with open(image_path, 'rb')as image_file:
        image_date = image_file.read()
        encoded_image = base64.b64encode(image_date).decode('utf+8')

    json = {'image_base64':encoded_image}
    return jsonify(json)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
