from pydantic import BaseModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

class Product(BaseModel):
    name: str
    price: int
    brand: str


llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0)


structured_llm=llm.with_structured_output(Product,method="json_schema")


response=structured_llm.invoke("Give me information about iphone 16")


print(response)
print(response.name)


