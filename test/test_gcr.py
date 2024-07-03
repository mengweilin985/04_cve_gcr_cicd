import requests

resp = requests.post("https://ml-run-service-rlhfgxhqqq-as.a.run.app", files={'file': open('0.jpg', 'rb')})

print(resp.json())
