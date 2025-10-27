import json
import base64
import time
import urllib.request
import os
import urllib.error
import urllib.parse
import cv2
import numpy as np


SERVER = 'http://localhost:5000'


def make_data_uri():
    # If a sample image exists in attached_assets/sample_face.jpg use it
    sample_path = 'attached_assets/sample_face.jpg'
    if os.path.exists(sample_path):
        with open(sample_path, 'rb') as f:
            b = f.read()
            b64 = base64.b64encode(b).decode('ascii')
            return f'data:image/jpeg;base64,{b64}'

    # fallback: create a simple placeholder image (100x100) - not a real face but exercises endpoints
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (80, 150, 200)
    # draw a simple face-like circle so mediapipe has a tiny chance (not guaranteed)
    cv2.circle(img, (50, 40), 18, (220, 180, 140), -1)
    _, buf = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def http_post_json(path, payload):
    url = SERVER + path
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        'Content-Type': 'application/json'
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print('HTTPError', e.code, e.read().decode('utf-8'))
        raise
    except Exception:
        raise


def main():
    print('Smoke test started against', SERVER)
    datauri = make_data_uri()

    print('1) Registering test student...')
    payload = {
        'name': 'TEST_USER_SMOKE',
        'roll_number': 'SMOKE-001',
        'class_name': 'SmokeClass',
        'section': 'S',
        'registration_number': 'SMOKE-REG-1',
        'images': [datauri]
    }
    r = http_post_json('/api/register', payload)
    print('Register response:', r)
    if not r.get('success'):
        print('Register failed, aborting smoke test')
        return 1
    sid = r.get('student_id')

    print('2) Starting training...')
    r = http_post_json('/api/train', {})
    print('Train start response:', r)
    if not r.get('success'):
        print('Failed to start training (it may already be running)')

    # poll status
    timeout = 120
    start = time.time()
    while True:
        try:
            with urllib.request.urlopen(SERVER + '/api/train/status') as resp:
                st = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            print('Status error', e)
            break

        print('Training status:', st)
        if not st.get('running'):
            break
        if time.time() - start > timeout:
            print('Training timeout')
            break
        time.sleep(1)

    print('3) Calling predict endpoint...')
    r = http_post_json('/api/predict', {'image': datauri})
    print('Predict response:', r)

    # cleanup: delete the test student we created
    try:
        delete_url = SERVER + f'/api/students/{sid}'
        req = urllib.request.Request(delete_url, method='DELETE')
        with urllib.request.urlopen(req) as resp:
            print('Delete response:', resp.read().decode('utf-8'))
    except Exception as e:
        print('Cleanup delete failed:', e)

    print('Smoke test finished (student id =', sid, ')')
    return 0


if __name__ == '__main__':
    exit(main())