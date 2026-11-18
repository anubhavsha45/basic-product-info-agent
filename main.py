import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient
from langchain_core.messages import HumanMessage
from pydantic import BaseModel,Field
from langchain.agents.structured_output import ToolStrategy



load_dotenv()
tavily=TavilyClient()


class ProductInfo(BaseModel):
    name: str=Field(description="Name of the product")
    brand : str=Field(description="Brand of the product")
    category: str=Field(description="Product category")
    price: int=Field(description="Product price in INR")



@tool
def get_products()->str:
    """
    Get All the products from the products.json file

    """
    with open("products.json","r") as file:
        data=json.load(file)

    return str(data)


@tool
def search(query: str)->str:
    """
    Search all the internet for the information.
    Args:
        query:The search query .
    Return:
        Search results from the internet.

    """

    print(f"Searching for the {query}")
    return str(tavily.search(query=query,max_results=2))




llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0)
tools=[get_products,search]
agent=create_agent(model=llm,tools=tools)



def main():
    print("Welcome to product info agent!")

    result=agent.invoke({
        "messages":[
            HumanMessage(content="Find the i phone 16 in our product database and give me its details")
        ]
    })

    print(result["messages"][-1].content)



if __name__=="__main__":
    main()