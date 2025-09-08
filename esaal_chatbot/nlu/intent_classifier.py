# Simple intent classifier using keywords and semantic similarity
from sentence_transformers import SentenceTransformer, util

class IntentClassifier:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intents = {
            'greeting': [
                'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'howdy', 'hiya', 'yo'
            ],
            'schedule_appointment': [
                'schedule an appointment', 'book appointment', 'make appointment',
                'book a doctor', 'see a doctor', 'reserve a slot', 'appointment', 'booking',
                'I want to see a doctor', 'I need to book', 'I want to schedule', 'doctor appointment', 'visit doctor'
            ],
            'kb_query': [
                'information', 'tell me about', 'what is', 'how do I', 'details', 'services', 'clinic', 'doctor', 'hours', 'location',
                'I want some information', 'give me info', 'can you tell me', 'I want to know', 'explain', 'help me understand', 'about the clinic', 'about services', 'about doctors'
            ],
        }
        self.threshold = 0.45  # Lowered threshold for more flexibility

    def classify(self, user_input):
        user_input_lower = user_input.lower()
        # Keyword-based check
        for intent, keywords in self.intents.items():
            for kw in keywords:
                if kw in user_input_lower:
                    return intent
        # Semantic similarity check
        input_emb = self.model.encode(user_input, convert_to_tensor=True)
        best_intent = None
        best_sim = 0
        for intent, phrases in self.intents.items():
            phrase_embs = self.model.encode(phrases, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(input_emb, phrase_embs).max().item()
            if sim > best_sim:
                best_sim = sim
                best_intent = intent
        if best_sim > self.threshold:
            return best_intent
    # Fallback: treat as KB query if not recognized, only escalate if repeated failure
