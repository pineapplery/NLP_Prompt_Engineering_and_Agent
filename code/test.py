import os
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_BASE'] = 'https://api.yesapikey.com/v1'
os.environ['OPENAI_API_KEY'] = open('gpt3keys.txt').read().strip()

model = ChatOpenAI(model='gpt-3.5-turbo')
print(model.invoke('测试连通性，请回答"OK"'))