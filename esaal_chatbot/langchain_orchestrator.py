

# Simple orchestrator without LangChain dependencies
from kb.pdf_loader import PDFLoader
from kb.retriever import Retriever
from nlu.intent_classifier import IntentClassifier
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
intent_classifier = IntentClassifier()
entity_extractor = EntityExtractor()
csv_handler = CSVHandler(CSV_PATH)
dialog_flow = DialogFlow(csv_handler)
escalation = Escalation()

# General LLM (local flan-t5-small)
flan_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
flan_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
flan_pipe = pipeline('text2text-generation', model=flan_model, tokenizer=flan_tokenizer)
def flan_llm_func(x):
    result = flan_pipe(x, max_new_tokens=64, do_sample=False)[0]['generated_text']
    return result.strip()

def orchestrate(user_input):
    intent = intent_classifier.classify(user_input)
    if intent == "greeting":
        return "Hello! How can I assist you today?"
    elif intent == "kb_query":
        return retriever.search(user_input, use_rag=True)
    elif intent == "schedule_appointment":
        details = entity_extractor.extract(user_input)
        required = ['name', 'contact', 'date', 'time', 'service']
        for field in required:
            if field not in details or not details[field]:
                return f"Please provide your {field}: "
        return dialog_flow.schedule_appointment(details)
    elif intent == "escalate":
        return escalation.escalate()
    else:
        # Fallback to general LLM
        return flan_llm_func(user_input)
