# Simple intent classifier using keywords and semantic similarity

from sentence_transformers import SentenceTransformer, util
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

loader = PDFLoader(PDF_PATH)
kb_text = loader.extract_text()
retriever = Retriever(kb_text)
entity_extractor = EntityExtractor()
csv_handler = CSVHandler(CSV_PATH)
dialog_flow = DialogFlow(csv_handler)
escalation = Escalation()
flan_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
flan_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
flan_pipe = pipeline('text2text-generation', model=flan_model, tokenizer=flan_tokenizer)
def flan_llm_func(x):
    result = flan_pipe(x, max_new_tokens=64, do_sample=False)[0]['generated_text']
    return result.strip()

class IntentOrchestrator:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Four explicit intents: KB, Action, Escalation, FAQ/self-description
        self.intents = {
            'schedule_appointment': [
                'I want to schedule an appointment',
                'Book a doctor visit',
                'I need to see a doctor',
                'I want to book a checkup',
                'I want to book a consultation',
                'Can I make an appointment?',
                'I need to see a specialist',
                'I want to reserve a slot',
                'Schedule a meeting',
                'I want to see a physician',
                'Can I book a visit?',
                'I want to make a reservation',
                'I want to see a dermatologist',
                'I want to see a cardiologist',
                'I want to see a pediatrician',
                'I want to see an orthopedist',
                'I want to see a doctor for a checkup',
                'I want to see a doctor for a follow-up',
                'I want to see a doctor for a test',
                'I want to see a doctor for a vaccination',
            ],
            'kb_query': [
                'What services do you offer?',
                'Tell me about the clinic',
                'What are your opening hours?',
                'Can you provide information about your doctors?',
                'What does your clinic offer?',
                'What kind of treatments do you provide?',
                'What is available at your clinic?',
                'What can I get here?',
                'What are your specialties?',
                'What medical services do you have?',
                'What is your clinic offer?',
                'What do you provide?',
                'What can I ask you?',
                'What is your address?',
                'Where are you located?',
                'How can I contact you?',
                'What is your phone number?',
                'What is your email?',
                'What are your working hours?',
                'What is your location?',
                'What is your schedule?',
                'What is your consultation fee?',
                'What is your price?',
                'What is your cost?',
                'What is your rate?',
                'What is your charge?',
                'What is your payment method?',
                'What is your insurance policy?',
                'What is your refund policy?',
                'What is your cancellation policy?',
                'What is your reschedule policy?',
                'What is your no-show policy?',
                'What is your late policy?',
                'What is your privacy policy?',
                'What is your terms and conditions?',
                'What is your FAQ?',
                'What is your help?',
                'What is your support?',
                'What is your assistance?',
                'What is your guidance?',
                'What is your information?',
                'What is your details?',
                'What is your description?',
                'What is your overview?',
                'What is your summary?',
                'What is your background?',
                'What is your history?',
                'What is your mission?',
                'What is your vision?',
                'What is your values?',
                'What is your goals?',
                'What is your objectives?',
                'What is your achievements?',
                'What is your awards?',
                'What is your recognition?',
                'What is your reputation?',
                'What is your experience?',
                'What is your expertise?',
                'What is your credentials?',
                'What is your qualifications?',
                'What is your certifications?',
                'What is your licenses?',
                'What is your accreditations?',
                'What is your affiliations?',
                'What is your memberships?',
                'What is your partnerships?',
                'What is your collaborations?',
                'What is your network?',
                'What is your team?',
                'What is your staff?',
                'What is your doctors?',
                'What is your specialists?',
                'What is your consultants?',
                'What is your nurses?',
                'What is your assistants?',
                'What is your receptionists?',
                'What is your managers?',
                'What is your administrators?',
                'What is your coordinators?',
                'What is your supervisors?',
                'What is your directors?',
                'What is your executives?',
                'What is your owners?',
                'What is your founders?',
                'What is your CEO?',
                'What is your president?',
                'What is your vice president?',
                'What is your chairman?',
                'What is your board?',
                'What is your committee?',
                'What is your council?',
                'What is your advisory board?',
                'What is your governing body?',
                'What is your leadership?',
                'What is your management?',
                'What is your administration?',
                'What is your organization?',
                'What is your company?',
                'What is your business?',
                'What is your firm?',
                'What is your practice?',
                'What is your group?',
                'What is your association?',
                'What is your society?',
                'What is your club?',
                'What is your foundation?',
                'What is your institution?',
                'What is your establishment?',
                'What is your enterprise?',
                'What is your venture?',
                'What is your startup?',
                'What is your project?',
                'What is your initiative?',
                'What is your program?',
                'What is your campaign?',
                'What is your event?',
                'What is your activity?',
                'What is your operation?',
                'What is your service?',
                'What is your product?',
                'What is your solution?',
                'What is your offering?',
                'What is your feature?',
                'What is your benefit?',
                'What is your advantage?',
                'What is your strength?',
                'What is your weakness?',
                'What is your opportunity?',
                'What is your threat?',
                'What is your risk?',
                'What is your challenge?',
                'What is your issue?',
                'What is your problem?',
                'What is your concern?',
                'What is your question?',
                'What is your inquiry?',
                'What is your request?',
                'What is your suggestion?',
                'What is your feedback?',
                'What is your comment?',
                'What is your review?',
                'What is your rating?',
                'What is your testimonial?',
                'What is your reference?',
                'What is your case study?',
                'What is your example?',
                'What is your sample?',
                'What is your template?',
                'What is your form?',
                'What is your document?',
                'What is your file?',
                'What is your record?',
                'What is your report?',
                'What is your statement?',
                'What is your certificate?',
                'What is your license?',
                'What is your permit?',
                'What is your registration?',
                'What is your application?',
                'What is your submission?',
                'What is your approval?',
                'What is your rejection?',
                'What is your acceptance?',
                'What is your confirmation?',
                'What is your cancellation?',
                'What is your rescheduling?',
                'What is your postponement?',
                'What is your delay?',
                'What is your advance?',
                'What is your extension?',
                'What is your reduction?',
                'What is your increase?',
                'What is your decrease?',
                'What is your change?',
                'What is your update?',
                'What is your modification?',
                'What is your revision?',
                'What is your correction?',
                'What is your clarification?',
                'What is your explanation?',
                'What is your interpretation?',
                'What is your translation?',
                'What is your conversion?',
                'What is your adaptation?',
                'What is your adjustment?',
                'What is your customization?',
                'What is your personalization?',
                'What is your optimization?',
                'What is your automation?',
                'What is your integration?',
                'What is your migration?',
                'What is your implementation?',
                'What is your deployment?',
                'What is your installation?',
                'What is your configuration?',
                'What is your setup?',
                'What is your maintenance?',
                'What is your support?',
                'What is your troubleshooting?',
                'What is your repair?',
                'What is your replacement?',
                'What is your upgrade?',
                'What is your downgrade?',
                'What is your backup?',
                'What is your restore?',
                'What is your recovery?',
                'What is your reset?',
                'What is your restart?',
                'What is your shutdown?',
                'What is your startup?',
                'What is your reboot?',
                'What is your power?',
                'What is your battery?',
                'What is your charger?',
                'What is your cable?',
                'What is your adapter?',
                'What is your connector?',
                'What is your port?',
                'What is your slot?',
                'What is your socket?',
                'What is your plug?',
                'What is your outlet?',
                'What is your switch?',
                'What is your button?',
                'What is your key?',
                'What is your lock?',
                'What is your unlock?',
                'What is your open?',
                'What is your close?',
                'What is your start?',
                'What is your stop?',
                'What is your pause?',
                'What is your resume?',
                'What is your continue?',
                'What is your finish?',
                'What is your end?',
                'What is your complete?',
                'What is your incomplete?',
                'What is your pending?',
                'What is your active?',
                'What is your inactive?',
                'What is your available?',
                'What is your unavailable?',
                'What is your busy?',
                'What is your free?',
                'What is your occupied?',
                'What is your unoccupied?',
                'What is your reserved?',
                'What is your unreserved?',
                'What is your confirmed?',
                'What is your unconfirmed?',
                'What is your scheduled?',
                'What is your unscheduled?',
                'What is your planned?',
                'What is your unplanned?',
                'What is your expected?',
                'What is your unexpected?',
                'What is your required?',
                'What is your optional?',
                'What is your mandatory?',
                'What is your voluntary?',
                'What is your involuntary?',
                'What is your allowed?',
                'What is your not allowed?',
                'What is your permitted?',
                'What is your not permitted?',
                'What is your authorized?',
                'What is your unauthorized?',
                'What is your approved?',
                'What is your not approved?',
                'What is your accepted?',
                'What is your not accepted?',
                'What is your rejected?',
                'What is your not rejected?',
                'What is your valid?',
                'What is your invalid?',
                'What is your correct?',
                'What is your incorrect?',
                'What is your true?',
                'What is your false?',
                'What is your yes?',
                'What is your no?',
                'What is your maybe?',
                'What is your unsure?',
                'What is your certain?',
                'What is your uncertain?',
                'What is your sure?',
                'What is your unsure?',
                'What is your positive?',
                'What is your negative?',
                'What is your neutral?',
                'What is your satisfied?',
                'What is your dissatisfied?',
                'What is your happy?',
                'What is your unhappy?',
                'What is your pleased?',
                'What is your displeased?',
                'What is your content?',
                'What is your discontent?',
                'What is your grateful?',
                'What is your ungrateful?',
                'What is your thankful?',
                'What is your unthankful?',
                'What is your welcome?',
                'What is your unwelcome?',
                'What is your appreciated?',
                'What is your unappreciated?',
                'What is your respected?',
                'What is your disrespected?',
                'What is your trusted?',
                'What is your untrusted?',
                'What is your supported?',
                'What is your unsupported?',
                'What is your recommended?',
                'What is your not recommended?',
                'What is your preferred?',
                'What is your not preferred?',
                'What is your required?',
                'What is your not required?',
                'What is your optional?',
                'What is your not optional?',
                'What is your mandatory?',
                'What is your not mandatory?',
                'What is your voluntary?',
                'What is your not voluntary?',
                'What is your involuntary?',
                'What is your not involuntary?',
            ],
            'faq': [
                'Who are you?',
                'What is your purpose?',
                'Who created you?',
            ],
            'escalate': [
                'I want to talk to a human',
                'This is not helpful',
                'I want to make a complaint',
            ]
        }
        self.threshold = 0.35  # Lowered for even better recall
        self.fail_count = 0  # Track repeated fallback/failure
        self.dialog_state = {}  # Track dialog state per session (simple, single user)

    def route(self, user_input):
        user_input_lower = user_input.lower()
        # Only use semantic similarity for intent detection
        input_emb = self.model.encode(user_input, convert_to_tensor=True)
        best_intent = None
        best_sim = 0
        for intent in self.intents:
            phrase_embs = self.model.encode(self.intents[intent], convert_to_tensor=True)
            sim = util.pytorch_cos_sim(input_emb, phrase_embs).max().item()
            if sim > best_sim:
                best_sim = sim
                best_intent = intent
        if best_sim > self.threshold:
            self.fail_count = 0
            return self._handle_intent(best_intent, user_input)
        # Otherwise, treat as normal conversation (chit-chat, fallback)
        self.fail_count += 1
        if self.fail_count >= 2:
            self.fail_count = 0
            return self._handle_intent('escalate', user_input, reason='repeated_failure')
        return self._conversational_agent(user_input)

    def _handle_intent(self, intent, user_input, reason=None):
        if intent == "kb_query":
            # Always use RAG for KB queries
            return retriever.search(user_input, use_rag=True)
        elif intent == "schedule_appointment":
            # Multi-turn dialog: track and update state
            state = self.dialog_state.get('appointment', {})
            extracted = entity_extractor.extract(user_input)
            state.update({k: v for k, v in extracted.items() if v})
            required = ['name', 'contact', 'date', 'time', 'service']
            missing = [field for field in required if field not in state or not state[field]]
            self.dialog_state['appointment'] = state
            if missing:
                prompts = {
                    'name': "What is your name?",
                    'contact': "May I have your contact information (email or phone)?",
                    'date': "What date would you like the appointment?",
                    'time': "What time is convenient for you?",
                    'service': "What type of service or consultation do you need?"
                }
                return prompts[missing[0]]
            # All info collected, schedule and clear state
            confirmation = dialog_flow.schedule_appointment(state)
            self.dialog_state['appointment'] = {}  # Reset for next dialog
            return confirmation
        elif intent == "faq":
            return retriever.search(user_input, use_rag=True)
        elif intent == "escalate":
            if reason == 'repeated_failure':
                return escalation.escalate() + " (You have been transferred due to repeated misunderstanding.)"
            return escalation.escalate()
        else:
            return self._conversational_agent(user_input)

    # _faq_response removed: FAQ/self-description now routed to KB agent

    def _conversational_agent(self, user_input):
        # Handles greetings, chit-chat, and fallback
        response = flan_llm_func(user_input)
        if not response or not response.strip():
            return "Hello! How can I help you today?"
        return response
