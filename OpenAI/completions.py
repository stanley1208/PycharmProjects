import json
import requests
import os

apiKey="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"

headers={
    'Content-Type':'application/json',
    'Authorization':'Bearer '+apiKey
}

data=json.dumps({
    "prompt":"Once upon a time",
    "max_tokens":15
})

url='https://api.openai.com/v1/engines/davinci/completions'
result=requests.post(url,headers=headers,data=data)
print(result.json())

