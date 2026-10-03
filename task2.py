import re

# ============================================================
# CONVERSATIONAL CHATBOT USING PATTERN MATCHING
# ============================================================

print("==============================================")
print("        SIMPLE CONVERSATIONAL CHATBOT")
print("==============================================")
print("Type 'bye' or 'exit' to end the conversation.\n")


# ============================================================
# 1. INTENT KEYWORD MAPPING
# ============================================================

intent_keywords = {

    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ],

    "help": [
        "help",
        "support",
        "assist",
        "what can you do",
        "how can you help",
        "how may you help"
    ],

    "technology": [
        "python",
        "computer",
        "technology",
        "programming",
        "coding",
        "software",
        "artificial intelligence",
        "ai",
        "machine learning"
    ],

    "study": [
        "study",
        "exam",
        "learn",
        "learning",
        "course",
        "education",
        "homework"
    ],

    "goodbye": [
        "bye",
        "goodbye",
        "exit",
        "quit",
        "see you",
        "see ya"
    ]
}


# ============================================================
# 2. RESPONSES FOR EACH INTENT
# ============================================================

responses = {

    "greeting": [
        "Hello! Nice to meet you.",
        "Hi! How can I help you today?",
        "Hey! What would you like to talk about?"
    ],

    "help": [
        "I can help you with technology, programming, studying, and general questions.",
        "I can answer questions about Python, technology, and learning.",
        "Sure! Tell me what you need help with."
    ],

    "technology": [
        "Technology is a broad field involving computers, software, AI, and digital systems.",
        "Python is a popular programming language used for web development, data science, AI, and automation.",
        "Artificial intelligence allows computers to perform tasks that normally require human intelligence."
    ],

    "study": [
        "A good study routine includes regular practice, short breaks, and revision.",
        "I can help you understand programming and other study topics.",
        "Try breaking a difficult topic into smaller concepts and practice each one."
    ],

    "goodbye": [
        "Goodbye! Have a great day!",
        "See you later!",
        "Bye! It was nice talking with you."
    ]
}


# ============================================================
# 3. INTENT RECOGNITION USING REGEX + KEYWORD MATCHING
# ============================================================

def recognize_intent(user_input):

    # Convert input to lowercase
    text = user_input.lower().strip()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Check each intent
    for intent, keywords in intent_keywords.items():

        for keyword in keywords:

            # Escape keyword so special characters don't cause
            # problems in the regular expression
            pattern = r"\b" + re.escape(keyword) + r"\b"

            if re.search(pattern, text):
                return intent

    # No keyword matched
    return "fallback"


# ============================================================
# 4. FALLBACK RESPONSES
# ============================================================

fallback_responses = [
    "I'm sorry, I didn't understand that. Could you rephrase it?",
    "I'm not sure I understand. Try asking me about technology, programming, or studying.",
    "I don't have an answer for that yet. Could you ask another question?",
    "Sorry, I couldn't identify what you are asking about."
]


# ============================================================
# 5. CHATBOT RESPONSE FUNCTION
# ============================================================

def chatbot_response(user_input):

    intent = recognize_intent(user_input)

    # Handle unknown input
    if intent == "fallback":
        return fallback_responses[0]

    # Return a response for the recognized intent
    return responses[intent][0]


# ============================================================
# 6. START CHAT
# ============================================================

while True:

    user_input = input("You: ")

    # Empty input handling
    if not user_input.strip():
        print("Bot: Please type something so I can respond.")
        continue

    intent = recognize_intent(user_input)

    # Goodbye intent ends the conversation
    if intent == "goodbye":
        print("Bot:", responses["goodbye"][0])
        break

    # Generate response
    response = chatbot_response(user_input)

    print("Bot:", response)