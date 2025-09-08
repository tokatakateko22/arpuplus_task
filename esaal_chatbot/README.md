# ESAAL Chatbot Agent

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
python main.py
```

## Project Structure
- `kb/`: Knowledge base PDF extraction and retrieval
- `scheduler/`: Appointment scheduling logic (CSV)
- `nlu/`: Intent classification and entity extraction
- `dialog_manager.py`: Conversation state management
- `escalation.py`: Human agent escalation logic
- `utils.py`: Common utilities
