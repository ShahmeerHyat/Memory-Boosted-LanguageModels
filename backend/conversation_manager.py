import time
from typing import List, Dict
from models import Message, MessageRole

SYSTEM_PROMPT = """You are PharmaChat, a professional pharmacy information assistant for MediPlus Pharmacy.

Your role:
- Provide accurate information about medications, dosages, and side effects
- Explain potential drug interactions clearly
- Guide proper medication administration and storage
- Help users understand when they must consult a doctor or pharmacist

Your strict policies:
- Provide INFORMATION ONLY — never diagnose or prescribe
- If a user mentions chest pain, difficulty breathing, or any emergency — immediately tell them to call emergency services (115 in Pakistan / 911)
- Never recommend stopping a prescribed medication without doctor guidance
- Always flag serious drug interactions with a clear WARNING

Communication style:
- Professional yet warm and approachable
- Concise answers — avoid overwhelming the user
- Ask one clarifying question at a time when needed (age, allergies, current meds, pregnancy)
- Use simple language, avoid excessive medical jargon

Disclaimer: Always remind users that your information is educational only, not a substitute for professional medical advice.
"""

# Keywords indicating high-value patient context worth retaining across turns
IMPORTANT_KEYWORDS = [
    "allerg", "pregnant", "breastfeed", "nursing", "years old", "age",
    "diabete", "hypertension", "heart", "kidney", "liver", "asthma",
    "blood pressure", "taking", "currently on", "prescribed", "mg",
    "daily", "twice", "dose", "react", "side effect", "intoleran"
]

# Max turns to keep verbatim (most recent)
MAX_RECENT_TURNS = 8
# Hard cap on total messages before pruning old low-importance turns
MAX_TOTAL_TURNS = 20


class ConversationManager:
    def __init__(self):
        # session_id -> list of Message objects
        self.sessions: Dict[str, List[Message]] = {}

    def get_or_create_session(self, session_id: str) -> List[Message]:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: MessageRole, content: str):
        session = self.get_or_create_session(session_id)
        session.append(Message(role=role, content=content, timestamp=time.time()))

    def _importance_score(self, message: Message) -> int:
        """
        Context Window Algorithm — Signal vs Noise Filter.

        Scores each message by how much critical patient context it contains.
        High-scoring old messages are retained as a compressed context block
        rather than being dropped entirely when the window is full.

        Returns an integer score (0 = noise, >0 = signal worth keeping).
        """
        content_lower = message.content.lower()
        score = sum(1 for kw in IMPORTANT_KEYWORDS if kw in content_lower)
        return score

    def build_messages(self, session_id: str) -> List[dict]:
        """
        Build the full message list to send to the LLM.

        Sliding Window Strategy:
        1. Always inject the system prompt first.
        2. If history fits within MAX_RECENT_TURNS — send everything verbatim.
        3. If history is longer:
           a. Split into old_turns (everything before the last MAX_RECENT_TURNS)
              and recent_turns (last MAX_RECENT_TURNS).
           b. Score each old turn for importance.
           c. Compress the top-3 important old turns into a single system-context
              block ("Key patient context from earlier").
           d. Send: [system_prompt] + [context_block?] + [recent_turns].

        This ensures:
        - Recent context is always fully preserved.
        - Important older facts (allergies, conditions, current meds) survive pruning.
        - Token budget stays manageable on a local laptop GPU.
        """
        history = self.get_or_create_session(session_id)
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}

        if len(history) <= MAX_RECENT_TURNS:
            return [system_msg] + [
                {"role": m.role.value, "content": m.content} for m in history
            ]

        old_turns = history[:-MAX_RECENT_TURNS]
        recent_turns = history[-MAX_RECENT_TURNS:]

        # Score and sort old turns by importance, keep top 3
        scored = sorted(old_turns, key=self._importance_score, reverse=True)
        top_important = [t for t in scored[:3] if self._importance_score(t) > 0]

        messages = [system_msg]

        if top_important:
            facts = " | ".join(
                f"{t.role.value.upper()}: {t.content[:120].strip()}"
                for t in top_important
            )
            context_block = {
                "role": "system",
                "content": f"[Key patient context retained from earlier in conversation]: {facts}"
            }
            messages.append(context_block)

        messages += [
            {"role": m.role.value, "content": m.content} for m in recent_turns
        ]
        return messages

    def reset_session(self, session_id: str):
        self.sessions[session_id] = []

    def get_session_stats(self, session_id: str) -> dict:
        history = self.get_or_create_session(session_id)
        total = len(history)
        retained = min(total, MAX_RECENT_TURNS)
        return {
            "session_id": session_id,
            "turn_count": total,
            "retained_turns": retained,
            "window_strategy": "sliding_window_importance_filter"
        }
