import requests

resp = requests.post("http://127.0.0.1:8080", files={'file': open('1.jpg', 'rb')})

print(resp.json())
