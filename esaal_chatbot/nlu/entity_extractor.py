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
        # Service extraction (simple keyword match)
        services = ['consultation', 'checkup', 'follow-up', 'vaccination', 'test', 'therapy', 'diagnosis']
        for s in services:
            if s in user_input.lower():
                entities['service'] = s
                break
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
