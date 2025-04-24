from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="zephyr-7b-beta",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio"
)

print(llm.invoke("Write a summary on the impact of AI in education."))
