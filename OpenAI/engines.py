import requests
import os

apiKey="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"

headers={
    'Authorization':'Bearer '+apiKey
}

result=requests.get('https://api.openai.com/v1/engines',headers=headers)

print(result.json())