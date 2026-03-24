def rewrite_query(query, history):
    query = query.lower().strip()

    followups = ["more", "more info", "explain", "details", "tell me more"]

    if any(f in query for f in followups) and history:
        last_user = None

        for msg in reversed(history):
            if msg.type == "human":
                last_user = msg.content
                break

        if last_user:
            return f"Explain in detail about: {last_user}"

    return query