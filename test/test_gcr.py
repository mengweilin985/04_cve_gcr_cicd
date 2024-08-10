import requests

resp = requests.post("https://github-action-cve-run-nyluz5gvmq-as.a.run.app", files={'file': open('0.jpg', 'rb')})

print(resp.json())
