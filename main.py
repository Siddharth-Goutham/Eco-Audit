import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from typing import TypedDict, Optional, Annotated
import operator
from langchain_core.messages import SystemMessage, HumanMessage
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults


class EcoAudit(TypedDict):
    location:str
    distance: float
    commute_method: str
    housing_status: str
    total_emission: Optional[float]
    summary_dict: Annotated[list[dict], operator.add]
    region: str
    username: str

# FIX 1: Safely fetched the OpenRouter API key using .get() to prevent NameError
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="meta-llama/llama-3.1-8b-instruct",
    temperature=0,
)


@tool
def carbon_emission(location: str, distance: float, commute_method: str):
    '''Calculate carbon emissions based on location, distance travelled and commute method.'''

    messages = [
        SystemMessage(
            content="You are a carbon emission calculator. Find the emission factor in kg CO2 per km. Return ONLY a numeric value like 0.20. No explanation."),
        HumanMessage(content=f"Location is {location}, commute is {commute_method}")
    ]
    response = model.invoke(messages).content

    emission_factor = float(response)

    total_carbon = distance * emission_factor

    return round(total_carbon, 4)

# FIX 2: Used .get() instead of calling os.environ as a function to prevent TypeError
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "your_tavily_api_key_here")

web_search = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
@tool
def research_finder(query: str):
    """Searches the live internet for local eco-resources, e-waste centers, and composting facilities.
    Pass a specific search query like 'nearest e-waste recycling center in Urwa Store Mangalore'.
    """
    results = web_search.invoke({"query": query})
    return str(results)

@tool
def govt_subsidary(location: str, housing_status:str):
    '''To check if housing_status is owner then install solar subsidy and if not owner then utility tariffs'''
    if housing_status.lower()=="owner":
        return f"PM Surya Ghar Rooftop Solar Subsidy available in {location}."
    return f"Green Energy Utility Tariffs available for renters in {location}."

def carbon_emission_calculator(state: EcoAudit):
    llm_bind = model.bind_tools([carbon_emission], tool_choice="carbon_emission")
    messages=[
        SystemMessage(content="You are a carbon emission calculator. You have to calculate the carbon emission based on the parameters given my the HumanMessage."),
        HumanMessage(content=
                     f"Location is {state.get('location')},"
                     f" commute is {state.get('commute_method')}"
                     f" and distance travelled per day is {state.get('distance')}")
    ]
    r=llm_bind.invoke(messages)
    tool_args = r.tool_calls[0]['args']
    tool_result=float(carbon_emission.invoke(tool_args))

    return {"summary_dict": [{"calculated_emission": tool_result, "Node": 'carbon_emission_calculator'}]}

def research_node(state: EcoAudit):

    bind_llm = model.bind_tools([research_finder], tool_choice="research_finder")

    messages = [
        SystemMessage(content="You are a research assistant. Create a highly specific internet search query to find the exact address of an e-waste center for the user's location. Pass that query to your search tool. Provide the exact location precisely according to the human given content."),
        HumanMessage(content=f"city: {state.get('location')} region: {state.get('region', '')}")
    ]

    r = bind_llm.invoke(messages)
    tool_args = r.tool_calls[0]['args']

    tool_result = research_finder.invoke(tool_args)

    parser_message = [
        SystemMessage(content="""I am passing you a list of e-waste centers extracted from a DuckDuckGo web search. You have to extract ONLY 1 nearest center's information using exactly this format:

        Name of the center: [name]
        Location_situated: [location]
        Contact Number: [number or N/A]
        Address: [address]"""),

        HumanMessage(content=f"Raw Web Extract:\n{tool_result}")
    ]

    r2 = model.invoke(parser_message).content

    return {'summary_dict': [{"E-Waste center": r2, "Node": 'research_node'}]}


def govt_subs(state: EcoAudit):
    llm_binder=model.bind_tools([govt_subsidary])
    messages=[
    SystemMessage(content="You are good in government policies and schemes. You have to provide the information regarding the subsidaries using tool"),
    HumanMessage(content=f"location:{state.get('location')} and housing_status: {state.get('housing_status')} ")
    ]
    r=llm_binder.invoke(messages)
    tool_args = r.tool_calls[0]['args']
    tool_output=govt_subsidary.invoke(tool_args)

    return {'summary_dict': [{'govt_subsidy': tool_output, 'node': 'govt_sub'}]}


def compiler(state:EcoAudit):
    message=[
        SystemMessage(content="You are an expert environmental auditor."
        "You will be provided with a user's lifestyle profile and a dictionary of raw audit data. Dont hallucinate and strictly stick to the provided Raw_Audit data."
        "Write a professional, personalized report that includes:"
        "1. A summary of their carbon emission. In bracket write the username provided by the user"
        "2. Personalized recommendations on how to reduce emissions based on their" "specific commute method."
       "3. Safety measures and steps to utilize the provided e-waste center and government subsidies."
        "Format this beautifully in Markdown."),
        HumanMessage(content=f"User Profile:"
        f"- Location: {state.get('location')}, {state.get('region')}"
        f"- Commute Method: {state.get('commute_method')}"
       f"- Daily Distance: {state.get('distance')} km"
        f"- Housing Status: {state.get('housing_status')}"
        f"- Username: {state.get('username')}"

        "Raw Audit Data to include:"
        f"{state.get('summary_dict')}")
    ]
    response=model.invoke(message).content
    new_record = {
        "Final_Report": response,
        "Node": "compiler"
    }

    return {"summary_dict": [new_record]}


workflow=StateGraph(EcoAudit)
workflow.add_node('carbon_emission_calculator', carbon_emission_calculator)
workflow.add_node('research_node',research_node)
workflow.add_node('govt_subs',govt_subs)
workflow.add_node('compiler',compiler)

workflow.add_edge(START, 'carbon_emission_calculator')
workflow.add_edge(START, 'research_node')
workflow.add_edge(START, 'govt_subs')

workflow.add_edge('carbon_emission_calculator', 'compiler')
workflow.add_edge('research_node', 'compiler')
workflow.add_edge('govt_subs','compiler')

workflow.add_edge('compiler',END)

graph=workflow.compile()
