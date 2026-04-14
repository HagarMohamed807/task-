# Academic Advisor Agent 🎓

A FastAPI service that recommends one of 12 programming tracks to a student
through a **10-question adaptive MCQ interview** powered by OpenAI.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
Open `config.py` and replace the placeholder:
```python
OPENAI_API_KEY = "sk-..."   # your real key
```

### 3. Run the server
```bash
uvicorn main:app --reload
```

The API docs are available at: **http://127.0.0.1:8000/docs**

---

## API Reference

### `POST /start`
Starts a new interview session and returns the first question.

**Response:**
```json
{
  "session_id": "uuid-string",
  "question": "Scenario-based question text ...",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "question_number": 1,
  "is_finished": false,
  "recommendation": null
}
```

---

### `POST /answer`
Submits the student's answer and returns the next question or the final recommendation.

**Request body:**
```json
{
  "session_id": "uuid-string",
  "user_answer": "B"
}
```

**Response (mid-interview):**
```json
{
  "session_id": "uuid-string",
  "question": "Next question ...",
  "options": { "A": "...", "B": "...", "C": "...", "D": "..." },
  "question_number": 2,
  "is_finished": false,
  "recommendation": null
}
```

**Response (after 10th answer):**
```json
{
  "session_id": "uuid-string",
  "question": null,
  "options": null,
  "question_number": null,
  "is_finished": true,
  "recommendation": {
    "track_name": "Data Science & Analysis",
    "reasoning": "Based on your answers..."
  }
}
```

---

### `GET /health`
```json
{ "status": "ok", "active_sessions": 3 }
```

---

## Available Tracks
1. Frontend Development  
2. Backend Development  
3. Mobile Development  
4. Data Science & Analysis  
5. Data Engineering (ETL, Pipelines, Big Data)  
6. Computer Vision (Image Processing, Deep Learning)  
7. Internet of Things (IoT & Hardware)  
8. Cloud Engineering & DevOps  
9. Cybersecurity  
10. Game Development  
11. Artificial Intelligence (General ML/DL)  
12. Embedded Systems  
