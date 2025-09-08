# Entry point for ESAAL Chatbot Agent

from kb.pdf_loader import PDFLoader
from kb.retriever import Retriever
from nlu.intent_classifier import IntentClassifier
from nlu.entity_extractor import EntityExtractor
from scheduler.csv_handler import CSVHandler
from scheduler.dialog_flow import DialogFlow
from escalation import Escalation
import os

PDF_PATH = '/home/ebrahim/Arpuplus_project/healthcare_clinic_detailed.pdf'  # Adjust path if needed
CSV_PATH = '/home/ebrahim/Arpuplus_project/doctors_weekly_schedule.csv'


def main():
    print("Welcome to ESAAL Chatbot Agent!")
    print("Loading knowledge base from PDF...")
    loader = PDFLoader(PDF_PATH)
    kb_text = loader.extract_text()
    print("PDF loaded. Text length:", len(kb_text))

    retriever = Retriever(kb_text)
    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()
    csv_handler = CSVHandler(CSV_PATH)
    dialog_flow = DialogFlow(csv_handler)
    escalation = Escalation()

    # Ensure CSV exists with correct columns
    if not os.path.exists(CSV_PATH):
        import pandas as pd
        pd.DataFrame(columns=['name', 'contact', 'date', 'time', 'service']).to_csv(CSV_PATH, index=False)

    # For escalation tracking
    fail_count = 0

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break
        intent = intent_classifier.classify(user_input)
        if intent == "kb_query":
            answer = retriever.search(user_input, use_rag=True)
            print(f"Bot: {answer}")
            fail_count = 0
        elif intent == "schedule_appointment":
            # Multi-turn dialog to collect info
            details = entity_extractor.extract(user_input)
            required = ['name', 'contact', 'date', 'time', 'service']
            for field in required:
                if field not in details or not details[field]:
                    details[field] = input(f"Bot: Please provide your {field}: ")
            print("Bot: Confirming your appointment...")
            confirmation = dialog_flow.schedule_appointment(details)
            print(f"Bot: {confirmation}")
            fail_count = 0
        elif intent == "escalate" or fail_count >= 2:
            print(f"Bot: {escalation.escalate()}")
            fail_count = 0
        else:
            print("Bot: Sorry, I didn't understand that.")
            fail_count += 1

if __name__ == "__main__":
    main()
