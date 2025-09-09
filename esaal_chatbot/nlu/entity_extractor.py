# Basic entity extractor for date, time, service, name, and contact
import re
from datetime import datetime

class EntityExtractor:
    def extract(self, user_input):
        entities = {}
        # Date extraction (simple, e.g., 'September 10', '10/09/2025', 'next Tuesday')
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\d{1,2}-\d{1,2}-\d{2,4})',
            r'(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b \d{1,2})',
            r'(next \w+)',
        ]
        for pat in date_patterns:
            m = re.search(pat, user_input, re.IGNORECASE)
            if m:
                entities['date'] = m.group(0)
                break
        # Time extraction (e.g., '2 PM', '14:00', '3:30pm')
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(am|pm)?)',
            r'(\d{1,2}\s*(am|pm))',
        ]
        for pat in time_patterns:
            m = re.search(pat, user_input, re.IGNORECASE)
            if m:
                entities['time'] = m.group(0)
                break
        # Service extraction (improved: fuzzy match and noun phrase extraction)
        services = ['consultation', 'checkup', 'follow-up', 'vaccination', 'test', 'therapy', 'diagnosis', 'appointment', 'meeting', 'visit', 'session', 'treatment', 'procedure', 'exam', 'screening', 'injection', 'immunization', 'assessment', 'review', 'referral', 'counseling', 'advice', 'support']
        user_lower = user_input.lower()
        found_service = None
        for s in services:
            if s in user_lower:
                found_service = s
                break
        if not found_service:
            # Try to extract noun phrases as fallback (very basic)
            m = re.search(r'for a ([a-z ]+)', user_lower)
            if m:
                found_service = m.group(1).strip()
        if not found_service:
            m = re.search(r'for ([a-z ]+)', user_lower)
            if m:
                found_service = m.group(1).strip()
        if found_service:
            entities['service'] = found_service
        # Name extraction (e.g., 'my name is ...')
        m = re.search(r'my name is ([A-Za-z ]+)', user_input, re.IGNORECASE)
        if m:
            entities['name'] = m.group(1).strip()
        # Contact extraction (email or phone)
        m = re.search(r'([\w\.-]+@[\w\.-]+)', user_input)
        if m:
            entities['contact'] = m.group(1)
        m = re.search(r'(\+?\d{10,15})', user_input)
        if m:
            entities['contact'] = m.group(1)
        return entities
