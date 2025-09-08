
from langchain.chains.router import MultiRouteChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from kb.pdf_loader import PDFLoader
from kb.retriever import Retriever
from nlu.entity_extractor import EntityExtractor
from scheduler.csv_handler import CSVHandler
from scheduler.dialog_flow import DialogFlow
from escalation import Escalation
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os

PDF_PATH = '/home/ebrahim/Arpuplus_project/healthcare_clinic_detailed.pdf'
CSV_PATH = '/home/ebrahim/Arpuplus_project/doctors_weekly_schedule.csv'

# Load KB and tools
loader = PDFLoader(PDF_PATH)
kb_text = loader.extract_text()
retriever = Retriever(kb_text)
entity_extractor = EntityExtractor()
csv_handler = CSVHandler(CSV_PATH)
dialog_flow = DialogFlow(csv_handler)
escalation = Escalation()

# Define LangChain tools

def kb_tool_func(input):
    return retriever.search(input, use_rag=True)

def action_tool_func(input):
    details = entity_extractor.extract(input)
    required = ['name', 'contact', 'date', 'time', 'service']
    for field in required:
        if field not in details or not details[field]:
            return f"Please provide your {field}: "
    return dialog_flow.schedule_appointment(details)

def escalation_tool_func(input):
    return escalation.escalate()

kb_tool = Tool(
    name="KnowledgeBase",
    func=kb_tool_func,
    description="Use this tool to answer questions about the clinic, services, or general info."
)
action_tool = Tool(
    name="AppointmentScheduler",
    func=action_tool_func,
    description="Use this tool to schedule appointments or perform actions."
)
escalation_tool = Tool(
    name="Escalation",
    func=escalation_tool_func,
    description="Use this tool to escalate to a human agent."
)

# General LLM (local flan-t5-small)
flan_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
flan_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
flan_pipe = pipeline('text2text-generation', model=flan_model, tokenizer=flan_tokenizer)
def flan_llm_func(x):
    result = flan_pipe(x, max_new_tokens=64, do_sample=False)[0]['generated_text']
    return result.strip()
llm_tool = Tool(
    name="GeneralLLM",
    func=flan_llm_func,
    description="General conversation or small talk."
)

# Router prompt
router_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Classify the user input: {input}
Route to: GeneralLLM, KnowledgeBase, AppointmentScheduler, or Escalation.
"""
)

router_chain = MultiRouteChain(
    chains={
        "GeneralLLM": llm_tool,
        "KnowledgeBase": kb_tool,
        "AppointmentScheduler": action_tool,
        "Escalation": escalation_tool,
    },
    router_prompt=router_prompt,
    default_chain=llm_tool
)

def orchestrate(user_input):
    return router_chain.run(user_input)
