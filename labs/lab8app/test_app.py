import requests

data = {"model" : "logged_model2"}


url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=data)
# print(response.json())
print(response.status_code)
print(response.text)


curl -X 'POST' \
'http://127.0.0.1:8000/predict' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{"model" : "logged_model2"}'