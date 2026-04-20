def classify_query(query: str):
    query = query.lower()

    violence = [
        "bomb", "explosive", "hijack", "kidnap", "murder", "kill someone"
    ]

    self_harm = [
        "suicide", "kill myself", "end my life", "i want to die"
    ]

    emotional = [
        "breakup", "sad", "depressed", "lonely", "heartbroken"
    ]

    if any(word in query for word in violence):
        return "violence"
    elif any(word in query for word in self_harm):
        return "self_harm"
    elif any(word in query for word in emotional):
        return "emotional"
    else:
        return "safe"
    
def violence_response():
    return """
⚠️ I cannot assist with harmful or illegal activities.

If this is an emergency, contact authorities:
📞 Emergency: 112
"""

def self_harm_response():
    return """
💛 I'm really sorry you're feeling this way.

You’re not alone, and things can improve with support.

📞 India Mental Health Helpline: 9152987821  
📞 Emergency: 112  

Please consider reaching out to someone you trust.
"""

def emotional_response():
    return """
💬 It sounds like you're going through a tough time.

Breakups and sadness can feel overwhelming, but they do pass.

Try to:
- Talk to a friend
- Take rest
- Do something you enjoy

I'm here if you want to talk more.
"""