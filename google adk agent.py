import os
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types
import warnings
warnings.filterwarnings("ignore")
# === Config ===
APP_NAME = "med_advisor_app"
USER_ID = "patient_001"
SESSION_ID = "session_001"
os.environ['GOOGLE_API_KEY'] = 'your api key'  # Replace with your actual key

# === Escalation Tool for Critical Symptoms (silent) ===
def escalate_if_critical(query: str, tool_context: ToolContext) -> None:
    critical_keywords = [
        "emergency", "urgent", "chest pain", "severe", "fainting",
        "shortness of breath", "bleeding", "unconscious", "heart attack"
    ]
    if any(word in query.lower() for word in critical_keywords):
        tool_context.actions.transfer_to_agent = "backup_doctor"

# === Escalation Tool for Follow-Up ("yes" trigger) ===
def route_on_yes(query: str, tool_context: ToolContext) -> None:
    yes_words = ["yes", "yeah", "sure", "okay", "please do", "i want to", "go ahead"]
    if any(word in query.lower() for word in yes_words):
        tool_context.actions.transfer_to_agent = "followup_agent"

# === Tools ===
critical_tool = FunctionTool(func=escalate_if_critical)
yes_response_tool = FunctionTool(func=route_on_yes)

# === Main Medical Advisor Agent ===
med_advisor = Agent(
    model="gemini-2.0-flash-exp",
    name="med_advisor",
    instruction="""
    You are a trusted virtual medical advisor.
    Offer medicine suggestions, care routines, and general advice based on symptoms.
    Do not ask follow-up questions.
    If the user indicates consent (like 'yes'), use the 'route_on_yes' tool silently to pass control.
    If symptoms are severe, silently use the 'escalate_if_critical' tool.
    You do everything calmly and confidently.
    """,
    tools=[critical_tool, yes_response_tool]
)

# === Backup Doctor for Emergencies ===
backup_doctor = Agent(
    model="gemini-2.0-flash-exp",
    name="backup_doctor",
    instruction="""
    You are a senior medical advisor handling urgent symptoms.
    Give emergency instructions calmly. Do not ask questions or mention escalation.
    your main response to recommend drugs(medicines)
    """
)

# === Follow-Up Agent ===
followup_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="followup_agent",
    instruction="""
    You are the follow-up assistant.
    Based on previous advice, help the user take the next steps—like guiding on prescriptions, next checkups, or scheduling a doctor visit.
    Do not ask questions. Assume all context is known. Just provide helpful next actions.
    """
)

# === Link Sub-Agents ===
med_advisor.sub_agents = [backup_doctor, followup_agent]

# === Session and Runner Setup ===
session_service = InMemorySessionService()
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=med_advisor, app_name=APP_NAME, session_service=session_service)

# === Agent Interaction Function ===
def call_med_advisor(message: str):
    content = types.Content(role='user', parts=[types.Part(text=message)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            print("💊 Response:", event.content.parts[0].text)

# === Example Flow ===

# Step 1: User describes symptoms
call_med_advisor("I'm feeling cough with a mild fever.")

# Step 2: User replies to proceed
call_med_advisor("Yes, I want to continue with your recommendation.")
