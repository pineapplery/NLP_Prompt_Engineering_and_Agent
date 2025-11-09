import os
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ["OPENAI_API_KEY"] = "sk-QcunsEWQPceYxCmnD130Aa9e281d4dAbB35799046f29C3Ec"
os.environ["OPENAI_BASE_URL"] = "https://api.yesapikey.com/v1"
model = ChatOpenAI(model="gpt-3.5-turbo")


#os.environ["DEEPSEEK_API_KEY"] = "your_deepseek_api_key"
#os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"
#model = ChatDeepSeek(model="deepseek-chat")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个AI助手，请回答用户的问题。"),
        ("user", "{input}")
    ]
)

chain = prompt | model | StrOutputParser()

response = chain.invoke({"input": "帮我写一篇关于ai的技术文章，100字"})
print(response)
