import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.services.intent import get_intent, get_llm_client

TEST_DATA = {
    # ── original cases ────────────────────────────────────────────────────────

    # pure greetings
    "Hello!": "greeting",
    "hi": "greeting",
    "Good morning, is anyone there?": "greeting",

    # pure goodbyes
    "Goodbye for now": "goodbye",
    "bye bye": "goodbye",
    "I have to go now, talk later": "goodbye",

    # pure gratitude
    "Thank you so much!": "gratitude",
    "Thanks for your help": "gratitude",

    # out of scope
    "Write a python script to reverse a linked list.": "out_of_scope",
    "What is the capital of France?": "out_of_scope",
    "My stomach hurts and i have a fever, what medicine should i take?": "out_of_scope",

    # mental health questions
    "I have been feeling completely hopeless and sad for weeks.": "asking_mental_health_question",
    "How do i cope with a sudden panic attack?": "asking_mental_health_question",
    "My anxiety is making it impossible to sleep at night.": "asking_mental_health_question",

    # mixed intents (should override and pick mental health)
    "Hi there! I am feeling really depressed today and need advice.": "asking_mental_health_question",
    "Hello, can you help me? I am having a lot of relationship stress.": "asking_mental_health_question",
    "Thank you for being here. I am struggling with severe grief right now.": "asking_mental_health_question",

    # crisis / extreme distress
    "I can't take this life anymore, i want to end it all.": "asking_mental_health_question",

    # ── extended greetings ────────────────────────────────────────────────────
    "Hey, how are you doing today?": "greeting",
    "Good evening!": "greeting",
    "Ahlan! (Hello in Arabic)": "greeting",
    "Heyyyy it's me again": "greeting",
    "Morning. Hope you're doing well.": "greeting",
    "What's up?": "greeting",
    "Yo, I'm back": "greeting",

    # ── extended goodbyes ─────────────────────────────────────────────────────
    "See you later!": "goodbye",
    "I'm logging off, take care": "goodbye",
    "Alright, gotta run": "goodbye",
    "Peace out": "goodbye",
    "Thanks, I'm done for now": "goodbye",
    "That's all I needed, bye!": "goodbye",

    # ── extended gratitude ────────────────────────────────────────────────────
    "I really appreciate your support": "gratitude",
    "You've been very helpful, thank you!": "gratitude",
    "Wow, that was exactly what I needed. Thanks!": "gratitude",
    "Much appreciated.": "gratitude",
    "You're amazing, seriously.": "gratitude",

    # ── out of scope — diverse topics ─────────────────────────────────────────
    "Can you recommend a good sci-fi movie?": "out_of_scope",
    "What's 2 to the power of 10?": "out_of_scope",
    "Translate 'good morning' to French.": "out_of_scope",
    "How do I make pasta carbonara?": "out_of_scope",
    "What time is it in Tokyo right now?": "out_of_scope",
    "Give me the lyrics to a famous song.": "out_of_scope",
    "Explain quantum entanglement to me.": "out_of_scope",
    "Help me debug this SQL query.": "out_of_scope",
    "What's the weather like in Cairo?": "out_of_scope",

    # ── mental health — direct ────────────────────────────────────────────────
    "I feel empty inside all the time.": "asking_mental_health_question",
    "What are some grounding techniques for anxiety?": "asking_mental_health_question",
    "I can't stop crying and I don't even know why.": "asking_mental_health_question",
    "I think I might have bipolar disorder, what should I do?": "asking_mental_health_question",
    "How do I deal with intrusive thoughts?": "asking_mental_health_question",
    "I've been having panic attacks at work almost every day.": "asking_mental_health_question",
    "Is it normal to feel numb after a breakup?": "asking_mental_health_question",
    "What's the difference between sadness and clinical depression?": "asking_mental_health_question",
    "I've been feeling burned out for months and can't recover.": "asking_mental_health_question",

    # ── mental health — oblique / low signal ─────────────────────────────────
    # Tests whether the classifier can pick up implied distress without keywords
    "I just feel like everything is pointless lately.": "asking_mental_health_question",
    "My mind won't stop racing at night.": "asking_mental_health_question",
    "I don't feel like myself anymore.": "asking_mental_health_question",
    "Things have been really hard, I don't know how to explain it.": "asking_mental_health_question",
    "I used to enjoy things but now nothing feels good.": "asking_mental_health_question",

    # ── crisis / high distress ────────────────────────────────────────────────
    "I don't see the point in going on anymore.": "asking_mental_health_question",
    "I've been thinking about hurting myself.": "asking_mental_health_question",
    "Everything feels so dark and I have no one to talk to.": "asking_mental_health_question",
    "I'm scared of my own thoughts right now.": "asking_mental_health_question",

    # ── mixed intent — greeting + mental health ───────────────────────────────
    "Hi, I've been struggling with loneliness for a long time.": "asking_mental_health_question",
    "Hey there, I'm feeling really overwhelmed and anxious.": "asking_mental_health_question",
    "Good morning. I had another nightmare and my PTSD is flaring up.": "asking_mental_health_question",

    # ── mixed intent — gratitude + mental health ──────────────────────────────
    "Thanks for being here. I've been feeling very isolated lately.": "asking_mental_health_question",
    "I appreciate you. Honestly, talking to you is the only thing helping my depression.": "asking_mental_health_question",

    # ── mixed intent — goodbye + gratitude ───────────────────────────────────
    # Goodbye should win since the conversation is wrapping up
    "Thank you so much, bye now!": "goodbye",
    "You've been wonderful. Take care, I'm heading out.": "goodbye",
    "Thanks for everything, talk soon!": "goodbye",

    # ── ambiguous / tricky edge cases ────────────────────────────────────────
    "I'm done.": "goodbye",
    "I can't anymore.": "asking_mental_health_question",
    "Everything hurts.": "asking_mental_health_question",
    "I feel nothing.": "asking_mental_health_question",
    "That helped, thanks. See you.": "goodbye",
    "Okay.": "greeting",
    "Not great, honestly.": "asking_mental_health_question",

    # ── looks like MHQ but is out of scope ───────────────────────────────────
    # Tests whether the classifier conflates topic mention with user intent
    "Write a short story about a character dealing with depression.": "out_of_scope",
    "What percentage of people suffer from anxiety globally?": "out_of_scope",
    "Is SSRI medication addictive? (asking for a school project)": "out_of_scope",

    # ── sarcasm / informal tone ───────────────────────────────────────────────
    # Tests robustness to indirect expression of distress
    "oh great another day of existing": "asking_mental_health_question",
    "yeah im totally fine lol": "asking_mental_health_question",
    "super fun having anxiety 24/7": "asking_mental_health_question",
    "wow I love not sleeping ever": "asking_mental_health_question",
}


def run_intent_tests():
    print("connecting to llm client...")
    try:
        client = get_llm_client()
    except Exception as e:
        print(f"failed to initialize client: {e}")
        return

    passed = 0
    failed = 0

    # group results by category for a cleaner report
    results_by_category = {}

    print("\nrunning intent prompt evaluation...\n")
    print(f"{'User Message':<65} | {'Expected':<30} | {'Predicted':<30} | Status")
    print("-" * 135)

    for i, (message, expected) in enumerate(TEST_DATA.items()):
        try:
            predicted = get_intent(message, client)
            status = "PASS" if predicted == expected else "FAIL"
            if predicted == expected:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            predicted = f"ERROR: {str(e)[:20]}"
            status = "FAIL"
            failed += 1

        display_msg = message if len(message) <= 63 else message[:60] + "..."
        print(f"{display_msg:<65} | {expected:<30} | {str(predicted):<30} | {status}")
        
        if i < len(TEST_DATA) - 1:
            time.sleep(4)  # brief pause to avoid hitting rate limits

    print("-" * 135)
    total = len(TEST_DATA)
    accuracy = (passed / total) * 100
    print(f"\nresults -> passed: {passed} / {total} | failed: {failed} / {total} | accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    run_intent_tests()