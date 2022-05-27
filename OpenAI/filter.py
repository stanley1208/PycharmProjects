import requests
import json
from api_key import key

url = "https://api.openai.com/v1/engines/content-filter-alpha-c4/completions"

payload = json.dumps({
  "prompt": "<|endoftext|>Are you religions?\n--\nLabel:",
  "max_tokens": 1,
  "temperature": 0,
  "top_p": 0
})
headers = {
  'Authorization': 'Bearer '+key,
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
