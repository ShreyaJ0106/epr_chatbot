# EPR Chatbot

## Overview
This repository contains the complete source code for the EPR Chatbot.

## Deployment Status
The application is currently experiencing a deployment-related runtime issue on Hugging Face Spaces due to vector database initialization. The codebase and application logic are complete and functional in a local environment.

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/ShreyaJ0106/epr_chatbot.git
cd epr_chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file and add the required API keys.

### 5. Run the Application

```bash
python app.py
```

or

```bash
streamlit run app.py
```

(depending on the application entry point)

## Notes

The deployment issue is specific to the Hugging Face Spaces environment and does not affect the local execution of the chatbot.

If required, I would be happy to provide a live demonstration of the chatbot through a meeting or screen-sharing session.
